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

    // First
    Visualizer viz("win", "disp");

    std::vector<int> keys;
    keys.push_back('a');
    keys.push_back('z');
    viz.registerKeyboardCallback(keys, test, NULL);

    viz.addPointCloud("pcd", cloud);

    // Second
    PointCloud cloud2(cloud);
    for (size_t i = 0; i < cloud2.size(); i++) {
        cloud2.points[i] += Eigen::Vector3f(1.0, 0.0, 1.0);
    }

    Visualizer viz2("win2", "disp2");
    viz2.addPointCloud("pcd1", cloud, Visualizer::RenderingProperties().setDrawingColor(1,0,0).setOverrideColors(true).setOpacity(0.2));
    viz2.addPointCloud("pcd2", cloud2, Visualizer::RenderingProperties().setDrawingColor(0,0,1).setOverrideColors(true));
    viz2.addPointCloudNormals("nrm1", cloud, Visualizer::RenderingProperties().setCorrespondencesFraction(0.20).setDrawingColor(0,1,0).setOpacity(0.2));
    viz2.addPointCloudNormals("nrm2", cloud2, Visualizer::RenderingProperties().setCorrespondencesFraction(0.20).setDrawingColor(0,1,0));
    viz2.addPointCorrespondences("corr", cloud, cloud2, Visualizer::RenderingProperties().setCorrespondencesFraction(0.01));

    while (!viz.wasStopped() || !viz2.wasStopped()) {
        viz.spinOnce();
        viz2.spinOnce();
    }

    return 0;
}
