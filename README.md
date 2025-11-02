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

ReCoGS is a pipeline and a tool for interactively ReColoring a pre-trained gaussian splatting scene.

<p align="center">  
    <img src="https://github.com/user-attachments/assets/6e585aa8-fde9-4816-a829-ccacecc1fc8d" alt="ReCoGS demo treehill"/>
</p>

<details>
<summary>Additional scenes</summary>
<p align="center">  
  <img src="https://github.com/user-attachments/assets/d068c3b9-bf92-413b-8a19-f3f3b3201435" alt="ReCoGS demo train"/>
  <img src="https://github.com/user-attachments/assets/953b8cbc-aceb-49b8-8358-820b969b61c8" alt="ReCoGS demo train"/>
  <img src="https://github.com/user-attachments/assets/c60c3cef-2ab3-4e5e-a968-8a316a819e0e" alt="RecoGS demo flowers" />
</p>
</details>

## How to run

#### Linux

Requirements:
- A **NVIDIA GPU** with Compute Capability 7.5 or higher
- [vcpkg](https://github.com/microsoft/vcpkg) installation: make sure `VCPKG_ROOT` is defined.
- CUDA 12.6
- TensorRT 10.8.0.43

Clone and build the repository:

```bash
git clone https://github.com/loryruta/recogs update --init --recursive
cd recogs
# Download large assets
bash ./download_assets.sh
# Build
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
> This is caused by TensorRT having to compile the Stereo Matching model to an optimized version tailored
> to your specific device.

## License

This code is distributed under the MIT license.

## Citation

If you find our work helpful, you can cite us as:

```
TODO
```
