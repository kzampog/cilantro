#include <iostream>
#include <cilantro/utilities/io_utilities.hpp>

int main(int argc, char ** argv) {
    Eigen::MatrixXf mat0(3,4);
    mat0 << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

    std::cout << "Before:" << std::endl;
    std::cout << mat0 << std::endl;

    bool binary = false;

    cilantro::writeEigenMatrixToFile("mat.dat", mat0, binary);

    Eigen::MatrixXf mat1;

    cilantro::readEigenMatrixFromFile("mat.dat", mat1, binary);

    std::cout << "After:" << std::endl;
    std::cout << mat1 << std::endl;

    return 0;
}
