#include <cilantro/visualizer.hpp>
#include <cilantro/ply_io.hpp>
#include <cilantro/voxel_grid.hpp>

#include <iostream>

void test(Visualizer &viz, int key, void *cookie) {
    std::cout << ((char)key) << std::endl;
}

int main(int argc, char ** argv) {

    PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    VoxelGrid vg(cloud, 0.01);
    cloud = vg.getDownsampledCloud();

    Visualizer viz("win", "disp");

    viz.addPointCloud("pcd", cloud);
    viz.addPointCloudNormals("nrm", cloud, Visualizer::RenderingProperties().setNormalsPercentage(0.05));

    std::vector<int> keys;
    keys.push_back('a');
    keys.push_back('z');

    viz.registerKeyboardCallback(keys, test, NULL);

    while (!pangolin::ShouldQuit()) {
        viz.render();
        pangolin::FinishFrame();
    }

    return 0;
}
