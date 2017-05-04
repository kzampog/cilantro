#include <sisyphus/pca.hpp>
#include <iostream>

int main(int argc, char ** argv) {

	std::vector<Eigen::Matrix<float, 3, 1> > points;
	points.push_back(Eigen::Vector3f(0, 0, 0));
	points.push_back(Eigen::Vector3f(1, 0, 0));
	points.push_back(Eigen::Vector3f(0, 100, 0));
	points.push_back(Eigen::Vector3f(0, 0, 1000));
	points.push_back(Eigen::Vector3f(0, 100, 1000));
	points.push_back(Eigen::Vector3f(1, 0, 1000));
	points.push_back(Eigen::Vector3f(1, 100, 0));
	points.push_back(Eigen::Vector3f(1, 100, 1000));

	PCA pca(points);

	std::cout << pca.getMean() << std::endl;
	std::cout << pca.getEigenValues() << std::endl;
	std::cout << pca.getEigenVectors() << std::endl;

	return 0;
}
