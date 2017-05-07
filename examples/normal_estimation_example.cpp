#include <sisyphus/normal_estimation.hpp>
#include <sisyphus/ply_io.hpp>
#include <iostream>

#include <sisyphus/voxel_grid.hpp>

#include <ctime>

int main(int argc, char ** argv) {
    clock_t begin, end;
    double est_time;

    PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    VoxelGrid vg(cloud, 0.005);
    cloud = vg.getDownsampledCloud();

    cloud.normals.clear();
    begin = clock();
    estimatePointCloudNormals(cloud, 0.01, Eigen::Vector3f(0, 0, 0));
    end = clock();
    est_time = 1000.0*double(end - begin) / CLOCKS_PER_SEC;

    std::cout << "Estimation time: " << est_time << std::endl;

    writePointCloudToPLYFile(argv[2], cloud);

    return 0;
}
