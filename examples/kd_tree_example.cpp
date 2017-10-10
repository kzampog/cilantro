#include <cilantro/kd_tree.hpp>
#include <cilantro/ply_io.hpp>

#include <iostream>


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

    KDTree3D tree(points);

    std::vector<size_t> ind;
    std::vector<float> dist;

    tree.kNNInRadiusSearch(Eigen::Vector3f(0.1, 0.1, 0.4), 2, 1.001, ind, dist);

    std::cout << "Neighbor indices: ";
    for (int i = 0; i < ind.size(); i++) {
        std::cout << ind[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Neighbor distances: ";
    for (int i = 0; i < dist.size(); i++) {
        std::cout << dist[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
