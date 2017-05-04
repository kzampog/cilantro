#include <sisyphus/pca.hpp>
#include <iostream>

int main(int argc, char ** argv) {

	std::vector<Eigen::Matrix<float, 3, 1> > points;
	points.push_back(Eigen::Vector3f(0, 0, 0));
	points.push_back(Eigen::Vector3f(1, 0, 0));
	points.push_back(Eigen::Vector3f(0, 1, 0));
	points.push_back(Eigen::Vector3f(0, 0, 1));
	points.push_back(Eigen::Vector3f(0, 1, 1));
	points.push_back(Eigen::Vector3f(1, 0, 1));
	points.push_back(Eigen::Vector3f(1, 1, 0));
	points.push_back(Eigen::Vector3f(1, 1, 1));

	PCA pca(points);

	std::cout << pca.getMean() << std::endl;


	return 0;
}
