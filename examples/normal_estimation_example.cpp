#include <sisyphus/normal_estimation.hpp>
#include <sisyphus/ply_io.hpp>
#include <iostream>

#include <sisyphus/voxel_grid.hpp>

#include <ctime>

int main(int argc, char ** argv) {
    clock_t begin, end;
    double kd_tree_time, estimation_time;

    PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    cloud.normals.clear();

    VoxelGrid vg(cloud, 0.005);
    cloud = vg.getDownsampledCloud();

    begin = clock();
    KDTree tree(cloud);
    NormalEstimation ne(cloud, tree);
    end = clock();
    kd_tree_time = 1000.0*double(end - begin) / CLOCKS_PER_SEC;

    begin = clock();
    ne.computeNormalsRadius(cloud, 0.01);
//    ne.computeNormalsKNN(cloud, 7);
    end = clock();
    estimation_time = 1000.0*double(end - begin) / CLOCKS_PER_SEC;

    std::cout << "kd-tree time: " << kd_tree_time << std::endl;
    std::cout << "Estimation time: " << estimation_time << std::endl;

    writePointCloudToPLYFile(argv[2], cloud);

    return 0;
}
