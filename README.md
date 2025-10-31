<div align="center">
<h1>:art: ReCoGS: Real-time ReColoring for Gaussian Splatting scenes</h1>

<p align="center">
  <a href="https://google.com?q=TODO">
    <img src="https://img.shields.io/badge/arXiv-ReCoGS-red?logo=arxiv" alt="Paper PDF">
  </a>
</p>

**University of Modena and Reggio Emilia**

[Lorenzo Rutayisire](https://loryruta.github.io/),
[Nicola Capodieci](https://scholar.google.com/citations?user=0_onpPkAAAAJ&hl=it),
[Fabio Pellacini](https://xelatihy.github.io/)

</div>

## Overview

TODO

## How to run (Linux)

Requirements:
- A **NVIDIA GPU** with Compute Capability 7.5 or higher
- [vcpkg](https://github.com/microsoft/vcpkg) installation: make sure `VCPKG_ROOT` is defined.
- CUDA 12.6
- TensorRT 10.8.0.43

Clone and build the repository:

```bash
git clone https://github.com/loryruta/recogs
bash ./download_assets.sh

mkdir build
cd build
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . --target recogs
cd ..
```

Download sample scenes from the MipNeRF360 dataset (any colmap dataset can be used as well):

```
bash ./download_data.sh
```

Run recogs:

```
./build/recogs ./data/mipnerf360/kitchen/point_cloud/iteration_30000/point_cloud.ply
```

> :warning: **NOTE: on the first launch, recogs will take a long time to bootstrap (more than 10 minutes).**
> This caused by TensorRT having to compile the Stereo Matching model to an optimized version tailored
> to your specific device.

## Gallery

## License

This code is distributed under the MIT license.

## Citation

If you find our work helpful, you can cite us as:

```
TODO
```
