#include <cilantro/core/principal_component_analysis.hpp>
#include <iostream>

int main(int argc, char ** argv) {
    std::vector<Eigen::Vector3f> points;
    points.emplace_back(0, 0, 0);
    points.emplace_back(1, 0, 0);
    points.emplace_back(0, 100, 0);
    points.emplace_back(0, 0, 1000);
    points.emplace_back(0, 100, 1000);
    points.emplace_back(1, 0, 1000);
    points.emplace_back(1, 100, 0);
    points.emplace_back(1, 100, 1000);

    cilantro::PrincipalComponentAnalysis3f pca(points);

    std::cout << "Data mean: " << pca.getDataMean().transpose() << std::endl;
    std::cout << "Eigenvalues: " << pca.getEigenValues().transpose() << std::endl;
    std::cout << "Eigenvectors: " << std::endl << pca.getEigenVectors() << std::endl;

    return 0;
}
