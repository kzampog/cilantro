# cilantro
[![Build Status](https://travis-ci.org/kzampog/cilantro.svg?branch=master)](https://travis-ci.org/kzampog/cilantro) [![Documentation Status](https://readthedocs.org/projects/cilantro/badge/?version=latest)](http://cilantro.readthedocs.io/en/latest/?badge=latest)

`cilantro` is a lean C++ library for working with point cloud data, with emphasis given to the 3D case. It implements a number of common operations, while attempting to minimize the amount of code required by the user.

## Supported functionality

#### Basic operations:
- General dimension kd-trees (using bundled [nanoflann](https://github.com/jlblancoc/nanoflann))
- Surface normal and curvature estimation from raw point clouds
- General dimension grid-based point cloud resampling
- Principal Component Analysis
- Classical Multidimensional Scaling (using bundled [Spectra](https://github.com/yixuan/spectra) for eigendecompositions)
- Basic I/O utilities for 3D point clouds (in PLY format, using bundled [tinyply](https://github.com/ddiakopoulos/tinyply)) and Eigen matrices

#### Convex hulls:
- A general dimension convex polytope representation that is computed (using bundled [Qhull](http://www.qhull.org/)) from either vertex or half-space intersection input and allows for easy switching between the respective representations
- A representation of generic (general dimension) space regions as unions of convex polytopes that implements set operations

#### Clustering:
- General dimension k-means clustering that supports all distance metrics supported by [nanoflann](https://github.com/jlblancoc/nanoflann)
- Spectral clustering based on various graph Laplacian types (using bundled [Spectra](https://github.com/yixuan/spectra))
- Flat kernel mean-shift clustering
- Connected component based point cloud segmentation, with pairwise similarities capturing any combination of spatial proximity, normal smoothness, and color similarity

#### Model estimation and point set registration:
- A RANSAC estimator template and instantiations thereof for robust plane estimation and rigid 6DOF point cloud registration
- Generic Iterative Closest Point implementations for point-to-point and point-to-plane metrics (and combinations thereof) that support arbitrary correspondence types on any kind of point features

#### Visualization:
- A fast, powerful, and easy to use 3D visualizer
- RGBD images to/from point cloud utility functions

## Dependencies
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) (3.3 or newer)
- [Pangolin](https://github.com/stevenlovegrove/Pangolin) (built with Eigen enabled)

## Building
`cilantro` is developed and tested on Ubuntu 14.04 and 16.04 variants using [CMake](https://cmake.org/).
Please note that you may have to manually set up a recent version of Eigen on Ubuntu 14.04, as the one provided in the official repos is outdated.
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
Most of them expect a single command-line argument (path to a point cloud file in PLY format). One such input is bundled in `examples/test_clouds` for quick testing.
