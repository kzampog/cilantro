#include <iostream>
#include <cilantro/core/kd_tree.hpp>

int main(int argc, char ** argv) {
    std::vector<Eigen::Vector3f> points;
    points.emplace_back(0, 0, 0);
    points.emplace_back(1, 0, 0);
    points.emplace_back(0, 1, 0);
    points.emplace_back(0, 0, 1);
    points.emplace_back(0, 1, 1);
    points.emplace_back(1, 0, 1);
    points.emplace_back(1, 1, 0);
    points.emplace_back(1, 1, 1);

    cilantro::KDTree3f<> tree(points);

    cilantro::NeighborSet<float> nn = tree.kNNInRadiusSearch(Eigen::Vector3f(0.1, 0.1, 0.4), 2, 1.001);

    std::cout << "Neighbor indices: ";
    for (int i = 0; i < nn.size(); i++) {
        std::cout << nn[i].index << " ";
    }
    std::cout << std::endl;

    std::cout << "Neighbor distances: ";
    for (int i = 0; i < nn.size(); i++) {
        std::cout << nn[i].value << " ";
    }
    std::cout << std::endl;

    return 0;
}
