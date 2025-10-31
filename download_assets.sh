#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

wget -q --show-progress -O "$SCRIPT_DIR/assets/pcvnet.onnx" \
  https://huggingface.co/loryruta/recogs/resolve/main/pcvnet.onnx?download=true

wget -q --show-progress -O "$SCRIPT_DIR/assets/pcvnet_quant.onnx" \
  https://huggingface.co/loryruta/recogs/resolve/main/pcvnet_quant.onnx?download=true
