#include <cilantro/voxel_grid.hpp>
#include <cilantro/ply_io.hpp>
#include <cilantro/visualizer.hpp>

int main(int argc, char ** argv) {

//    PointCloud cloud;
//    cloud.points.push_back(Eigen::Vector3f(0, 0, 0));
//    cloud.points.push_back(Eigen::Vector3f(1, 0, 0));
//    cloud.normals.push_back(Eigen::Vector3f(0, 0, 1));
//    cloud.normals.push_back(Eigen::Vector3f(0, 0, -1));
//
//    VoxelGrid vg(cloud, 10);
//    PointCloud cloud_d = vg.getDownsampledCloud();
//
//    std::cout << cloud_d.normals[0] << std::endl;

    cilantro::PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    auto start = std::chrono::high_resolution_clock::now();
    cilantro::VoxelGrid vg(cloud, 0.01f);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> build_time = end - start;

    start = std::chrono::high_resolution_clock::now();

//    PointCloud cloud_d;
//    cloud_d.points = vg.getDownsampledPoints();
//    cloud_d.normals = vg.getDownsampledNormals();
//    cloud_d.colors = vg.getDownsampledColors();

    cilantro::PointCloud cloud_d = vg.getDownsampledCloud();

    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ds_time = end - start;

    std::cout << "Before: " << cloud.size() << std::endl;
    std::cout << "After: " << cloud_d.size() << std::endl;

    std::cout << "Build time: " << build_time.count() << "ms" << std::endl;
    std::cout << "Downsampling time: " << ds_time.count() << "ms" << std::endl;

    cilantro::Visualizer viz("win", "disp");

    viz.addPointCloud("cloud_d", cloud_d);
    viz.addPointCloudNormals("normals_d", cloud_d);

    while (!viz.wasStopped()){
        viz.spinOnce();
    }

    return 0;
}
