#include <sisyphus/cloud_visualizer.hpp>
#include <sisyphus/ply_io.hpp>
#include <sisyphus/voxel_grid.hpp>

#include <iostream>

int main(int argc, char ** argv) {

    PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

//    VoxelGrid vg(cloud, 0.01);
//    cloud = vg.getDownsampledCloud();

    CloudVisualizer viz("win", "disp");

    viz.addPointCloud("pcd", cloud, CloudVisualizer::RenderingProperties().setPointSize(10));
    viz.addPointCloudNormals("nrm", cloud, CloudVisualizer::RenderingProperties().setNormalsPercentage(0.05).setDrawingColor(0,1,0));

    while (!pangolin::ShouldQuit()) {
        viz.render();
        pangolin::FinishFrame();
    }

    return 0;
}
