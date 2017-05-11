#include <cilantro/pca.hpp>
#include <cilantro/ply_io.hpp>

#include <iostream>
#include <ctime>

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

    clock_t begin, end;
    double compute_time;

    begin = clock();
    PCA pca(cloud);
    end = clock();
    compute_time = 1000.0*double(end - begin) / CLOCKS_PER_SEC;

    std::cout << "Compute time: " << compute_time << std::endl;

    std::cout << "Data mean: " << pca.getMean().transpose() << std::endl;
    std::cout << "Eigenvalues: " << pca.getEigenValues().transpose() << std::endl;
    std::cout << "Eigenvectors: " << std::endl << pca.getEigenVectors() << std::endl;

    std::cout << "Determinant: " << pca.getEigenVectors().determinant() << std::endl;

    return 0;
}
