# cilantro
[![Build Status](https://travis-ci.org/kzampog/cilantro.svg?branch=master)](https://travis-ci.org/kzampog/cilantro)

`cilantro` is a lean C++ library for working with 3D point clouds. It implements a number of common operations, with emphasis given to minimizing the amount of code required by the user.

## Supported functionality
- Voxel grid based point cloud resampling
- kd-trees (using packaged [nanoflann](https://github.com/jlblancoc/nanoflann))
- Surface normal estimation from point clouds
- Convex hull computation (using packaged [Qhull](http://www.qhull.org/)) that allows easy switching between vertex and half-space representations for the resulting polytope
- A representation of space regions as unions of convex polytopes that implements set operations
- An Iterative Closest Point implementation (point-to-point and point-to-plane metrics are supported) that supports multiple correspondence types
- A generic RANSAC implementation and instantiations of it for robust plane estimation and rigid 6DOF point cloud registration
- Connected component based point cloud segmentation
- k-means clustering
- Principal Component Analysis
- A fast, flexible and easy to use 3D visualizer
- RGBD images to/from point cloud utility functions
- Basic point cloud I/O to and from PLY files (using packaged [tinyply](https://github.com/ddiakopoulos/tinyply))

## Dependencies
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) (3.3 or newer)
- [Pangolin](https://github.com/stevenlovegrove/Pangolin)

Developed and tested on Ubuntu 14.04 and 16.04 variants.
