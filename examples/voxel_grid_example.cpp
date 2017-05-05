#include <sisyphus/voxel_grid.hpp>
#include <sisyphus/ply_io.hpp>
#include <iostream>

#include <ctime>

int main(int argc, char ** argv) {
//    PointCloud cloud;
//    cloud.points.push_back(Eigen::Vector3f(0, 0, 0));
//    cloud.points.push_back(Eigen::Vector3f(1, 0, 0));
//    cloud.points.push_back(Eigen::Vector3f(0, 100, 0));
//    cloud.points.push_back(Eigen::Vector3f(0, 0, 1000));
//    cloud.points.push_back(Eigen::Vector3f(0, 100, 1000));
//    cloud.points.push_back(Eigen::Vector3f(1, 0, 1000));
//    cloud.points.push_back(Eigen::Vector3f(1, 100, 0));
//    cloud.points.push_back(Eigen::Vector3f(1, 100, 1000));
//
//    VoxelGrid vg(cloud, 150);
//    PointCloud cloud_d = vg.getDownsampledCloud();
//
//    std::cout << "Before: " << cloud.points.size() << std::endl;
//    std::cout << "After: " << cloud_d.points.size() << std::endl;

    clock_t begin, end;
    double build_time, ds_time;

    PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    begin = clock();
    VoxelGrid vg(cloud, 0.010f);
    end = clock();
    build_time = 1000.0*double(end - begin) / CLOCKS_PER_SEC;
    begin = clock();
    PointCloud cloud_d = vg.getDownsampledCloud();
    end = clock();
    ds_time = 1000.0*double(end - begin) / CLOCKS_PER_SEC;

    std::cout << "Before: " << cloud.points.size() << std::endl;
    std::cout << "After: " << cloud_d.points.size() << std::endl;

    std::cout << "Build time: " << build_time << std::endl;
    std::cout << "Downsampling time: " << ds_time << std::endl;

    writePointCloudToPLYFile(argv[2], cloud_d);

    return 0;
}
