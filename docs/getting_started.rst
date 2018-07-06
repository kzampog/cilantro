===============
Getting started
===============

Dependencies
============

* Eigen_ (3.3 or greater)
* Pangolin_ (built with Eigen enabled)

How to build
============
**cilantro**  is developed and tested on Ubuntu 14.04, 16.04, and 18.04 variants using CMake_. Please note that you may have to manually set up a recent version of Eigen on Ubuntu 14.04, as the one provided in the official repos is outdated. To clone and build the library (with bundled examples), execute the following in a terminal:

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