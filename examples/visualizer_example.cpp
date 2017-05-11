#include <sisyphus/visualizer.hpp>
#include <sisyphus/ply_io.hpp>
#include <sisyphus/voxel_grid.hpp>

#include <iostream>

int main(int argc, char ** argv) {

    PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    VoxelGrid vg(cloud, 0.01);
    cloud = vg.getDownsampledCloud();

    Visualizer viz("win", "disp");

    viz.addPointCloud("pcd", cloud, Visualizer::RenderingProperties().setPointSize(10));
    viz.addPointCloudNormals("nrm", cloud, Visualizer::RenderingProperties().setNormalsPercentage(0.05).setDrawingColor(0,1,0));

    while (!pangolin::ShouldQuit()) {
        viz.render();
        pangolin::FinishFrame();
    }

    return 0;
}
