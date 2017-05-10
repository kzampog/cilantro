#include <sisyphus/cloud_visualizer.hpp>
#include <sisyphus/ply_io.hpp>

#include <iostream>

int main(int argc, char ** argv) {

    PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    CloudVisualizer viz("win", "disp");

    viz.addPointCloud("pcd", cloud);
//    viz.addPointCloudNormals("nrm", cloud, 0.02f);

    while (!pangolin::ShouldQuit()) {
        viz.render();
        pangolin::FinishFrame();
    }

    return 0;
}
