# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import itertools
import logging
import numpy as np
import pickle
import random
import torch
import torch.utils.data as data
 
from torch.utils.data.sampler import Sampler

from detectron2.utils.serialize import PicklableWrapper

__all__ = ["MapDataset", "DatasetFromList", "AspectRatioGroupedDataset", "ToIterableDataset"]


class MapDataset(data.Dataset):
    """
    Map a function over the elements in a dataset.

    Args:
        dataset: a dataset where map function is applied.
        map_func: a callable which maps the element in dataset. map_func is
            responsible for error handling, when error happens, it needs to
            return None so the MapDataset will randomly use other
            elements from the dataset.
    """

    def __init__(self, dataset, map_func, offset=0, start_idx=0, training = True):
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))
        self._training = training
        self._offset = offset
        self.start = start_idx

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        retry_count = 0
        if self._training:
                data_list = []
                idx = idx[:-1]
                idx = [int(i) for i in idx]
        
                for cur_idx in idx: 
                  while True:
                        data = self._map_func(self._dataset[cur_idx]) 
                        if data is not None:
                            self._fallback_candidates.add(cur_idx)  
                            data_list.append(data) 
                            break  
                        else:
                            retry_count += 1
                            self._fallback_candidates.discard(cur_idx)
                            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

                            if retry_count >= 3:  
                                logger = logging.getLogger(__name__)
                                logger.warning(
                                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                                        idx, retry_count
                                    )
                                )
                #file_name = data_list[-1]['file_name'].split("_")[-1]

                third_idx = int(data_list[-1]['file_name'].split("_")[-2])
                i = torch.randint(0, 2, (1,)).item()
                # i = 0
                # third_idx = 0
                if i == 1:
                    third_idx = self.start + (idx[-1] - self.start) % self._offset 
                data = self._map_func(self._dataset[third_idx])  
                if data is not None:
                    self._fallback_candidates.add(third_idx)  
                    data_list.append(data)
                     
                return data_list  
        
        else:
                cur_idx = int(idx)

                while True:
                    data = self._map_func(self._dataset[cur_idx])
                    if data is not None:
                        self._fallback_candidates.add(cur_idx)
                        return data

                    # _map_func fails for this idx, use a random new index from the pool
                    retry_count += 1
                    self._fallback_candidates.discard(cur_idx)
                    cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

                    if retry_count >= 3:
                        logger = logging.getLogger(__name__)
                        logger.warning(
                            "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                                idx, retry_count
                            )
                        )


class DatasetFromList(data.Dataset):
    """
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(self, lst: list, copy: bool = True, serialize: bool = True):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
            serialize (bool): whether to hold memory using serialized objects, when
                enabled, data loader workers can use shared RAM from master
                process instead of making a copy.
        """
        self._lst = lst
        self._copy = copy
        self._serialize = serialize

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            logger = logging.getLogger(__name__)
            logger.info(
                "Serializing {} elements to byte tensors and concatenating them all ...".format(
                    len(self._lst)
                )
            )
            self._lst = [_serialize(x) for x in self._lst]
            self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)
            logger.info("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024 ** 2))

    def __len__(self):
        if self._serialize:
            return len(self._addr)
        else:
            return len(self._lst)

    def __getitem__(self, idx):
        if self._serialize:
            start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
            end_addr = self._addr[idx].item()
            bytes = memoryview(self._lst[start_addr:end_addr])
            return pickle.loads(bytes)
        elif self._copy:
            return copy.deepcopy(self._lst[idx])
        else:
            return self._lst[idx]


class ToIterableDataset(data.IterableDataset):
    """
    Convert an old indices-based (also called map-style) dataset
    to an iterable-style dataset.
    """

    def __init__(self, dataset, sampler):
        """
        Args:
            dataset (torch.utils.data.Dataset): an old-style dataset with ``__getitem__``
            sampler (torch.utils.data.sampler.Sampler): a cheap iterable that produces indices
                to be applied on ``dataset``.
        """
        assert not isinstance(dataset, data.IterableDataset), dataset
        assert isinstance(sampler, Sampler), sampler
        self.dataset = dataset
        self.sampler = sampler

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is None or worker_info.num_workers == 1:
            for idx in self.sampler:
                yield self.dataset[idx]
        else:
            # With map-style dataset, `DataLoader(dataset, sampler)` runs the
            # sampler in main process only. But `DataLoader(ToIterableDataset(dataset, sampler))`
            # will run sampler in every of the N worker and only keep 1/N of the ids on each
            # worker. The assumption is that sampler is cheap to iterate and it's fine to discard
            # ids in workers.
            for idx in itertools.islice(
                self.sampler, worker_info.id, None, worker_info.num_workers
            ):
                yield self.dataset[idx]


class AspectRatioGroupedDataset(data.IterableDataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.

    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: an iterable. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            batch_size (int):
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.group = batch_size // 3
        self._buckets = [[[] for _ in range(2)] for _ in range(3)]

    def __iter__(self):
        for d_list in self.dataset:
            for idx, d in enumerate(d_list):
                if idx == 0:
                    w, h = d["width"], d["height"]
                else:
                    box = d["instances"].gt_boxes.tensor[0]
                    w, h = (box[2]-box[0]).item(), (box[3]-box[1]).item()
                #w, h = d["width"], d["height"]
                bucket_id = 0 if w > h else 1  
                bucket = self._buckets[idx][bucket_id]
                bucket.append(d)
                if idx == 1:
                    self._buckets[-1][bucket_id].append(d_list[-1])
                    break
                # for d in d_list:
                #     bucket.append(d)
                    
            #for buck in self._buckets:
            bucket1 =[i for i in range(2) if len(self._buckets[0][i]) >= self.group]
            bucket2 = [i for i in range(2) if len(self._buckets[1][i]) >= self.group]
            if len(bucket1) > 0 and len(bucket2) > 0:
                idx1 = bucket1[0]
                idx2 = bucket2[0]
                bucket1 = self._buckets[0][idx1]
                bucket2 = self._buckets[1][idx2]
                bucket3 = self._buckets[2][idx2]
                
                yield bucket1[:self.group] + bucket2[:self.group] +bucket3[:self.group]

                del bucket1[:self.group]
                del bucket2[:self.group]
                del bucket3[:self.group]
                
              