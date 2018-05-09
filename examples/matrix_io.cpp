#include <cilantro/io.hpp>
#include <iostream>

int main(int argc, char ** argv) {
    Eigen::MatrixXf dok(3,4);
    dok << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

    std::cout << "Before:" << std::endl;
    std::cout << dok << std::endl;

    bool binary = false;

    cilantro::writeEigenMatrixToFile("mat.dat", dok, binary);

    Eigen::MatrixXf dok2;

    cilantro::readEigenMatrixFromFile("mat.dat", dok2, binary);

    std::cout << "After:" << std::endl;
    std::cout << dok2 << std::endl;

    std::vector<float> v;
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    v.push_back(4);

    std::cout << "Before:" << std::endl;
    for (int i = 0; i < v.size(); i++) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;

    cilantro::writeVectorToFile("mat.dat", v, binary);

    std::vector<float> v2;

    cilantro::readVectorFromFile("mat.dat", v2, binary);

    std::cout << "After:" << std::endl;
    for (int i = 0; i < v2.size(); i++) {
        std::cout << v2[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
