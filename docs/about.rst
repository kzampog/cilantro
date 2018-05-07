=====
About
=====

What is cilantro?
=================
**cilantro** is a lean C++ library for working with point cloud data, with emphasis given to the 3D case. It implements a number of common operations, while attempting to minimize the amount of code required by the user.

Features
========

* **Core operations:**
    * General dimension kd-trees (using bundled nanoflann_)
    * Surface normal and curvature estimation from raw point clouds
    * General dimension grid-based point cloud resampling
    * Principal Component Analysis
    * Classical Multidimensional Scaling (using bundled Spectra_ for eigendecompositions)
    * Basic I/O utilities for 3D point clouds (in PLY format, using bundled tinyply_) and Eigen matrices

* **Convex hulls and spatial reasoning tools:**
    * A general dimension convex polytope representation that is computed (using bundled Qhull_) from either vertex or half-space intersection input and allows for easy switching between the respective representations
    * A representation of generic (general dimension) space regions as unions of convex polytopes that implements set operations

* **Clustering:**
    * General dimension k-means clustering that supports all distance metrics supported by nanoflann
    * Spectral clustering based on various graph Laplacian types (using bundled Spectra)
    * Flat kernel mean-shift clustering
    * Connected component based point cloud segmentation that supports arbitrary point-wise similarity functions

* **Model estimation and point set registration:**
    * A RANSAC estimator template and instantiations thereof for robust plane estimation and rigid 6DOF point cloud registration
    * Fully generic Iterative Closest Point implementations for point-to-point and point-to-plane metrics (and combinations thereof) that support arbitrary correspondence search methods in arbitrary point feature spaces

* **Visualization:**
    * A powerful, extensible, and easy to use 3D visualizer
    * RGBD images to/from point cloud utility functions

.. _nanoflann: https://github.com/jlblancoc/nanoflann
.. _Spectra: https://github.com/yixuan/spectra
.. _tinyply: https://github.com/ddiakopoulos/tinyply
.. _Qhull: http://www.qhull.org/
