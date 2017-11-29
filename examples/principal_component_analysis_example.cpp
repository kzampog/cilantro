#include <cilantro/principal_component_analysis.hpp>
#include <cilantro/io.hpp>

#include <iostream>

int main(int argc, char ** argv) {
    cilantro::PointCloud cloud;
    cloud.points.emplace_back(0, 0, 0);
    cloud.points.emplace_back(1, 0, 0);
    cloud.points.emplace_back(0, 100, 0);
    cloud.points.emplace_back(0, 0, 1000);
    cloud.points.emplace_back(0, 100, 1000);
    cloud.points.emplace_back(1, 0, 1000);
    cloud.points.emplace_back(1, 100, 0);
    cloud.points.emplace_back(1, 100, 1000);

//    PointCloud cloud;
//    readPointCloudFromPLYFile(argv[1], cloud);

    cilantro::PrincipalComponentAnalysis3D pca(cloud.points);

    std::cout << "Data mean: " << pca.getDataMean().transpose() << std::endl;
    std::cout << "Eigenvalues: " << pca.getEigenValues().transpose() << std::endl;
    std::cout << "Eigenvectors: " << std::endl << pca.getEigenVectorsMatrix() << std::endl;

//    std::cout << "Determinant: " << pca.getEigenVectorsMatrix().determinant() << std::endl;

//    Eigen::MatrixXf pts(Eigen::MatrixXf::Map((float *)cloud.points.data(), 3, cloud.points.size()));
//
//    std::cout << "pts" << std::endl;
//    std::cout << pts << std::endl;
//
//    Eigen::MatrixXf ppts = pca.project(pts, 2);
//    std::cout << "ppts" << std::endl;
//    std::cout << ppts << std::endl;
//
//    std::vector<Eigen::Vector2f> sppts = pca.project<2>(cloud.points);
//    for (int i = 0; i < sppts.size(); i++) {
//        std::cout << sppts[i].transpose() << std::endl;
//    }
//
//    Eigen::MatrixXf rpts = pca.reconstruct(ppts);
//    std::cout << "rpts" << std::endl;
//    std::cout << rpts << std::endl;
//    std::vector<Eigen::Vector3f> srpts = pca.reconstruct<2>(sppts);
//    for (int i = 0; i < srpts.size(); i++) {
//        std::cout << srpts[i].transpose() << std::endl;
//    }

    return 0;
}
