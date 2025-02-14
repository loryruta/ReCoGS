#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ARCHIVE_FILENAME="onnxruntime-linux-x64-gpu-1.20.1"
URL="https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/$ARCHIVE_FILENAME.tgz"

cd "$SCRIPT_DIR" || true
wget -q "$URL" -O "onnxruntime.tgz"
tar -xf "onnxruntime.tgz"
if [ ! -d "$ARCHIVE_FILENAME" ]; then
  echo "ERROR: onnxruntime.tgz doesn't contain $ARCHIVE_FILENAME"
fi
rm "onnxruntime.tgz"
# Remove any previously created onnxruntime directory
if [ -d "onnxruntime" ]; then
  rm -rd "onnxruntime"
fi
mv "$ARCHIVE_FILENAME" "onnxruntime"

# Patch for 1.20.1: libonnxruntime.so.1.20.1 is searched in lib64 folder instead of lib
mv "onnxruntime/lib" "onnxruntime/lib64"
