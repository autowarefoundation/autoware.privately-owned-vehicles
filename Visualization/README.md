## Prerequisites

- **CMake** 3.20 or higher
- **Qt6** (Widgets module)
- **OpenCV**
- **C++23 compatible compiler** (GCC 7+, Clang 5+, or MSVC 2017+)

## Setup

### 1. Install Dependencies

#### Ubuntu 22
```bash
sudo apt update
sudo apt install cmake build-essential qt6-base-dev libopencv-dev
```

### 3. Prepare Dataset
Ensure the OpenLane dataset is placed in a `dataset/` directory:
https://github.com/OpenDriveLab/OpenLane/tree/main

```
dataset/
├── cipo/
└── images/
└── lane3d/
```

## Build

```bash
cmake -B build

cmake --build build
```

Build artifacts will be generated in a `build/` directory.

## Run

```bash
./build/AutowareHMI
```

## Project Structure

- **`src/`** - Source code (C++ headers and implementation files)
- **`dataset/`** - OpenLane dataset (images and JSON annotations)
