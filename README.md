# cilantro
`cilantro` is a lean C++ library for working with 3D point clouds. It implements a number of common operations, with emphasis given to minimizing the amount of code required by the user.

## Supported functionality
- Voxel grid based point cloud resampling
- kd-trees (using packaged [nanoflann](https://github.com/jlblancoc/nanoflann))
- Normal estimation from point clouds
- Convex hull computation (using packaged [Qhull](http://www.qhull.org/)) that allows easy switching between vertex and half-space representations
- Iterative Closest Point implementation (point-to-point and point-to-plane)
- A generic RANSAC implementation and instantiations of it for robust plane estimation and rigid registration
- Connected component based point cloud segmentation
- Principal Component Analysis
- RGBD images to/from point cloud utility functions
- A flexible and easy to use 3D visualizer
- Basic point cloud I/O to and from PLY files (using packaged [tinyply](https://github.com/ddiakopoulos/tinyply))

## Dependencies
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
- [Pangolin](https://github.com/stevenlovegrove/Pangolin)
