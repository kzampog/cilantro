# cilantro
A lean C++ library for working with 3D point clouds

## Supported functionality
`cilantro` has built-in support for the following:
- kd-trees (using packaged [nanoflann](https://github.com/jlblancoc/nanoflann))
- Voxel grid based point cloud resampling
- Normal estimation
- Convex hull computation (using packaged [qhull](https://github.com/qhull/qhull))
- Principal Component Analysis 
- RGBD to/from point cloud utility functions
- A generic RANSAC implementation and instantiations of it for robust plane estimation and rigid registration
- Iterative Closest Point implementation (point-to-point and point-to-plane)
- Connected component based point cloud segmentation
- A flexible 3D visualizer
- Basic point cloud I/O to and from PLY files (using packaged [tinyply](https://github.com/ddiakopoulos/tinyply))

## Dependencies
