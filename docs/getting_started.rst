===============
Getting started
===============

Dependencies
============

* Eigen_ (version 3.3 or newer) [**required**]
* Pangolin_ (built with Eigen enabled) [**optional**; needed for visualization modules and most examples]

How to build
============
**cilantro**  is developed and tested on Ubuntu variants (16.04 and newer) using CMake_. To clone and build the library (with bundled examples), execute the following in a terminal:

.. code-block:: bash

    git clone https://github.com/kzampog/cilantro.git
    cd cilantro
    mkdir build
    cd build
    cmake ..
    make -j

.. _Pangolin: https://github.com/stevenlovegrove/Pangolin
.. _Eigen: http://eigen.tuxfamily.org/index.php?title=Main_Page
.. _CMake: https://cmake.org/