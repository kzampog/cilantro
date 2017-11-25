# cilantro
[![Build Status](https://travis-ci.org/kzampog/cilantro.svg?branch=master)](https://travis-ci.org/kzampog/cilantro)

`cilantro` is a lean C++ library for working with 3D point clouds. It implements a number of common operations, with emphasis given to minimizing the amount of code required by the user.

## Supported functionality
- Voxel grid based point cloud resampling
- kd-trees (using packaged [nanoflann](https://github.com/jlblancoc/nanoflann))
- Surface normal estimation from point clouds
- Convex hull computation (using packaged [Qhull](http://www.qhull.org/)) that allows easy switching between vertex and half-space representations for the resulting polytope
- A representation of space regions as unions of convex polytopes that implements set operations
- An Iterative Closest Point implementation for point-to-point and point-to-plane metrics that supports multiple correspondence types
- A generic RANSAC implementation and instantiations of it for robust plane estimation and rigid 6DOF point cloud registration
- Connected component based point cloud segmentation, with pairwise similarities capturing any combination of spatial proximity, normal smoothness, and color similarity
- k-means clustering
- Principal Component Analysis
- A fast, flexible and easy to use 3D visualizer
- RGBD images to/from point cloud utility functions
- Basic point cloud I/O to and from PLY files (using packaged [tinyply](https://github.com/ddiakopoulos/tinyply))

## Dependencies
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) (3.3 or newer)
- [Pangolin](https://github.com/stevenlovegrove/Pangolin)

## Building
`cilantro` is developed and has been tested on Ubuntu 14.04 and 16.04 variants using [CMake](https://cmake.org/).
Please note that you may have to manually set up a recent version of Eigen on Ubuntu 14.04, as the one provided in the official repo is outdated.

To clone and build the library (with bundled examples), execute the following in a terminal:

```
git clone https://github.com/kzampog/cilantro.git
cd cilantro
mkdir build
cd build
cmake ..
make -j
```

## Usage examples
Documentation is sparse at the moment, but the short provided examples cover a significant part of the library's functionality.
Most of them expect a single command-line argument (path to a point cloud PLY file); one is bundled in `examples/test_clouds` for quick testing.
