#include <cilantro/convex_polyhedron.hpp>
#include <cilantro/ply_io.hpp>

#include <iostream>

int main(int argc, char ** argv) {

//    PointCloud cloud;
//    readPointCloudFromPLYFile(argv[1], cloud);

    std::vector<Eigen::Vector3f> points;
    points.push_back(Eigen::Vector3f(0,0,0));
    points.push_back(Eigen::Vector3f(1,0,0));
    points.push_back(Eigen::Vector3f(0,1,0));
    points.push_back(Eigen::Vector3f(0,0,1));
//    points.push_back(Eigen::Vector3f(0,1,1));
//    points.push_back(Eigen::Vector3f(1,0,1));
//    points.push_back(Eigen::Vector3f(1,1,0));
//    points.push_back(Eigen::Vector3f(1,1,1));

    VtoH(points);


    return 0;
}
