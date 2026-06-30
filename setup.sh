#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$ROOT_DIR/generate"
TARGET_DIR="$ROOT_DIR/dreamgaussian"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Project B does not exist: $TARGET_DIR"
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: generate directory does not exist: $SOURCE_DIR"
    exit 1
fi

find "$SOURCE_DIR" -maxdepth 1 -type f -print0 |
while IFS= read -r -d '' file; do
    filename="$(basename "$file")"
    cp "$file" "$TARGET_DIR/$filename"
    echo "Installed: generate/$filename -> B/$filename"
done

echo "Setup completed."