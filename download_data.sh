#!/bin/bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

MIPNERF360_DIR="$SCRIPT_DIR/data/mipnerf360"
mkdir -p "$MIPNERF360_DIR"

# counter
if [ ! -d "$MIPNERF360_DIR/counter" ]; then
    wget -q --show-progress -O "$MIPNERF360_DIR/counter.zip" \
        "https://huggingface.co/loryruta/recogs/resolve/main/data/mipnerf360/counter.zip?download=true"
    unzip -q "$MIPNERF360_DIR/counter.zip" -d "$MIPNERF360_DIR"
    rm "$MIPNERF360_DIR/counter.zip"
fi

# room
if [ ! -d "$MIPNERF360_DIR/room" ]; then
    wget -q --show-progress -O "$MIPNERF360_DIR/room.zip" \
        "https://huggingface.co/loryruta/recogs/resolve/main/data/mipnerf360/room.zip?download=true"
    unzip -q "$MIPNERF360_DIR/room.zip" -d "$MIPNERF360_DIR"
    rm "$MIPNERF360_DIR/room.zip"
fi

# train
if [ ! -d "$MIPNERF360_DIR/train" ]; then
    wget -q --show-progress -O "$MIPNERF360_DIR/train.zip" \
        "https://huggingface.co/loryruta/recogs/resolve/main/data/mipnerf360/train.zip?download=true"
    unzip -q "$MIPNERF360_DIR/train.zip" -d "$MIPNERF360_DIR"
    rm "$MIPNERF360_DIR/train.zip"
fi

# kitchen
if [ ! -d "$MIPNERF360_DIR/kitchen" ]; then
    wget -q --show-progress -O "$MIPNERF360_DIR/kitchen.zip" \
        "https://huggingface.co/loryruta/recogs/resolve/main/data/mipnerf360/kitchen.zip?download=true"
    unzip -q "$MIPNERF360_DIR/kitchen.zip" -d "$MIPNERF360_DIR"
    rm "$MIPNERF360_DIR/kitchen.zip"
fi

# room
if [ ! -d "$MIPNERF360_DIR/room" ]; then
    wget -q --show-progress -O "$MIPNERF360_DIR/room.zip" \
        "https://huggingface.co/loryruta/recogs/resolve/main/data/mipnerf360/room.zip?download=true"
    unzip -q "$MIPNERF360_DIR/room.zip" -d "$MIPNERF360_DIR"
    rm "$MIPNERF360_DIR/room.zip"
fi

echo "Done!"
