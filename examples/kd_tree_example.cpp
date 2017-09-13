#include <cilantro/kd_tree.hpp>
#include <cilantro/ply_io.hpp>

#include <iostream>
#include <ctime>

int main(int argc, char ** argv) {

    PointCloud cloud;
    cloud.points.push_back(Eigen::Vector3f(0, 0, 0));
    cloud.points.push_back(Eigen::Vector3f(1, 0, 0));
    cloud.points.push_back(Eigen::Vector3f(0, 1, 0));
    cloud.points.push_back(Eigen::Vector3f(0, 0, 1));
    cloud.points.push_back(Eigen::Vector3f(0, 1, 1));
    cloud.points.push_back(Eigen::Vector3f(1, 0, 1));
    cloud.points.push_back(Eigen::Vector3f(1, 1, 0));
    cloud.points.push_back(Eigen::Vector3f(1, 1, 1));

    KDTree3D tree(cloud.points);

    std::vector<size_t> ind;
    std::vector<float> dist;

    tree.kNNInRadiusSearch(Eigen::Vector3f(0, 0, 0), 2, 1.001, ind, dist);

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

//    PointCloud cloud;
//    readPointCloudFromPLYFile(argv[1], cloud);
//
//    clock_t begin, end;
//    double build_time;
//
//    begin = clock();
//    KDTree tree(cloud);
//    end = clock();
//    build_time = 1000.0*double(end - begin) / CLOCKS_PER_SEC;
//
//    std::cout << "Build time: " << build_time << std::endl;

    return 0;
}
