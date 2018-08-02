# cilantro
[![Build Status](https://travis-ci.org/kzampog/cilantro.svg?branch=master)](https://travis-ci.org/kzampog/cilantro) [![Documentation Status](https://readthedocs.org/projects/cilantro/badge/?version=latest)](http://cilantro.readthedocs.io/en/latest/?badge=latest) [![Documentation](https://codedocs.xyz/kzampog/cilantro.svg)](https://codedocs.xyz/kzampog/cilantro/)

`cilantro` is a lean and fast C++ library for working with point cloud data, with emphasis given to the 3D case.
It includes efficient implementations for a variety of common operations, providing a clean API and attempting to minimize the amount of boilerplate code.
The library is extensively templated, enabling operations on point data of arbitrary numerical type and dimensionality (where applicable) and featuring a modular/extensible design of the more complex procedures, while, at the same time, providing convenience aliases/wrappers for the most common cases.
A high-level description of `cilantro` can be found in our [technical report](https://arxiv.org/abs/1807.00399).

## Supported functionality

#### Basic operations:
- General dimension kd-trees (using bundled [nanoflann](https://github.com/jlblancoc/nanoflann))
- Surface normal and curvature estimation from raw point clouds
- General dimension grid-based point cloud resampling
- Principal Component Analysis
- Basic I/O utilities for 3D point clouds (in PLY format, using bundled [tinyply](https://github.com/ddiakopoulos/tinyply)) and Eigen matrices
- RGBD image pair to/from point cloud conversion utilities

#### Convex hulls:
- A general dimension convex polytope representation that is computed (using bundled [Qhull](http://www.qhull.org/)) from either vertex or half-space intersection input and allows for easy switching between the respective representations
- A representation of generic (general dimension) space regions as unions of convex polytopes that implements set operations

#### Clustering:
- General dimension k-means clustering that supports all distance metrics supported by [nanoflann](https://github.com/jlblancoc/nanoflann)
- Spectral clustering based on various graph Laplacian types (using bundled [Spectra](https://github.com/yixuan/spectra))
- Mean-shift clustering with custom kernel support
- Connected component based point cloud segmentation that supports arbitrary point-wise similarity functions

#### Geometric registration:
- Multiple generic Iterative Closest Point implementations that support arbitrary correspondence search methods in arbitrary point feature spaces for:
    - **Rigid** alignment under the point-to-point metric (general dimension), point-to-plane metric (3D), or any combination thereof
    - **Non-rigid** alignment of 3D point sets, by means of a robustly regularized, locally rigid warp field, under any combination of the point-to-point and point-to-plane metrics; implementations for both *densely* and *sparsely* (similarly to [DynamicFusion](http://grail.cs.washington.edu/projects/dynamicfusion/)) supported warp fields are provided

#### Robust model estimation:
- A RANSAC estimator template and instantiations thereof for general dimension:
    - Robust hyperplane estimation
    - Rigid point cloud registration given noisy correspondences

#### Visualization:
- Classical Multidimensional Scaling (using bundled [Spectra](https://github.com/yixuan/spectra) for eigendecompositions)
- A powerful, extensible, and easy to use 3D visualizer

## Dependencies
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) (3.3 or newer)
- [Pangolin](https://github.com/stevenlovegrove/Pangolin) (built with Eigen enabled)

## Building
`cilantro` is developed and tested on Ubuntu 14.04, 16.04, and 18.04 variants using [CMake](https://cmake.org/).
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

## Documentation
Documentation ([readthedocs.io](http://cilantro.readthedocs.io/en/latest/?badge=latest), [Doxygen API documentation](https://codedocs.xyz/kzampog/cilantro/)) is a work in progress.
The short provided examples (built by default) cover a significant part of the library's functionality.
Most of them expect a single command-line argument (path to a point cloud file in PLY format).
One such input is bundled in `examples/test_clouds` for quick testing.

## License
The library is released under the [MIT license](https://github.com/kzampog/cilantro/blob/master/LICENSE).
If you use `cilantro` in your research, please cite our [technical report](https://arxiv.org/abs/1807.00399):
```
@article{cilantro,
    author = {Konstantinos Zampogiannis and Cornelia Ferm{\"u}ller and Yiannis Aloimonos},
    title = {cilantro: a lean, versatile, and efficient library for point cloud data processing},
    archivePrefix = "arXiv",
    eprint = {1807.00399},
    year = {2018}
}
```
