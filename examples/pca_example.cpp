#include <sisyphus/pca.hpp>
#include <sisyphus/ply_io.hpp>
#include <iostream>

int main(int argc, char ** argv) {

    PointCloud cloud;

    cloud.points.push_back(Eigen::Vector3f(0, 0, 0));
    cloud.points.push_back(Eigen::Vector3f(1, 0, 0));
    cloud.points.push_back(Eigen::Vector3f(0, 100, 0));
    cloud.points.push_back(Eigen::Vector3f(0, 0, 1000));
    cloud.points.push_back(Eigen::Vector3f(0, 100, 1000));
    cloud.points.push_back(Eigen::Vector3f(1, 0, 1000));
    cloud.points.push_back(Eigen::Vector3f(1, 100, 0));
    cloud.points.push_back(Eigen::Vector3f(1, 100, 1000));

//    PointCloud cloud;
//    readPointCloudFromPLYFile(argv[1], cloud);

    PCA pca(cloud);

    std::cout << "Data mean: " << pca.getMean().transpose() << std::endl;
    std::cout << "Eigenvalues: " << pca.getEigenValues().transpose() << std::endl;
    std::cout << "Eigenvectors: " << std::endl << pca.getEigenVectors().transpose() << std::endl;

    std::cout << pca.getEigenVectors().determinant() << std::endl;

    return 0;
}
