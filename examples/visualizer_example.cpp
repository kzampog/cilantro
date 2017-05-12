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

    PointCloud cloud2(cloud);
    for (size_t i = 0; i < cloud2.size(); i++) {
        cloud2.points[i] += Eigen::Vector3f(1.0, 0.0, 1.0);
    }

    Visualizer viz("win", "disp");

    viz.addPointCloud("pcd1", cloud, Visualizer::RenderingProperties().setDrawingColor(1,0,0).setOverrideColors(true).setOpacity(0.2));
    viz.addPointCloud("pcd2", cloud2, Visualizer::RenderingProperties().setDrawingColor(0,0,1).setOverrideColors(true));
    viz.addPointCloudNormals("nrm1", cloud, Visualizer::RenderingProperties().setCorrespondencesFraction(0.20).setDrawingColor(0,1,0).setOpacity(0.2));
    viz.addPointCloudNormals("nrm2", cloud2, Visualizer::RenderingProperties().setCorrespondencesFraction(0.20).setDrawingColor(0,1,0));
    viz.addPointCorrespondences("corr", cloud, cloud2, Visualizer::RenderingProperties().setCorrespondencesFraction(0.01));

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
