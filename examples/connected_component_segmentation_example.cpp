#include <cilantro/connected_component_segmentation.hpp>
#include <cilantro/ply_io.hpp>
#include <cilantro/visualizer.hpp>
#include <iostream>

#include <cilantro/voxel_grid.hpp>

#include <ctime>

int main(int argc, char ** argv) {
    clock_t begin, end;
    double kd_tree_time, estimation_time;

    PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    cloud.normals.clear();

    VoxelGrid vg(cloud, 0.005);
    cloud = vg.getDownsampledCloud();



    Visualizer viz("win", "disp");

    viz.addPointCloud("cloud", cloud);
    while (!viz.wasStopped()){
        viz.spinOnce();
    }

    return 0;
}
