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

find "$SOURCE_DIR" -type f -print0 | while IFS= read -r -d '' file; do
    rel_path="${file#$SOURCE_DIR/}"
    target_path="$TARGET_DIR/$rel_path"

    mkdir -p "$(dirname "$target_path")"

    cp "$file" "$target_path"
    echo "Installed: generate/$rel_path -> dreamgaussian/$rel_path"
done

echo "Setup completed."