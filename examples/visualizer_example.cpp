#include <cilantro/visualizer.hpp>
#include <cilantro/ply_io.hpp>
#include <cilantro/voxel_grid.hpp>

#include <iostream>

void callback_test(Visualizer &viz, int key, void *cookie) {
    std::string name = *((std::string *)cookie);
    std::cout << ((char)key) << " " << name << std::endl;
    viz.toggleVisibility(name);
}

int main(int argc, char ** argv) {

    PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    VoxelGrid vg(cloud, 0.01);
    cloud = vg.getDownsampledCloud();

    // First
    Visualizer viz("win", "disp");

    std::vector<float> scalars (cloud.size());
    for (size_t i = 0; i < cloud.size(); i++)
        scalars[i] = cloud.points[i].norm();
    viz.addPointCloud("pcd", cloud, Visualizer::RenderingProperties().setColormapType(COLORMAP_JET));
    viz.addPointCloudValues("pcd", scalars);
    
//    viz.addPointCloud("pcd", cloud);
//    viz.addPointCloudNormals("nrm", cloud, Visualizer::RenderingProperties().setCorrespondencesFraction(0.20).setOpacity(0.5));

    viz.addCoordinateSystem("axis", 0.4f, Eigen::Matrix4f::Identity(), Visualizer::RenderingProperties().setLineWidth(10.0f));

    std::string name = "nrm";
    viz.registerKeyboardCallback(std::vector<int>(1,'n'), callback_test, &name);


    // Second
    PointCloud cloud2(cloud);
    for (size_t i = 0; i < cloud2.size(); i++) {
        cloud2.points[i] += Eigen::Vector3f(1.0, 0.0, 1.0);
    }

    Visualizer viz2("win2", "disp2");
    viz2.addPointCloud("pcd1", cloud, Visualizer::RenderingProperties().setDrawingColor(1,0,0).setOpacity(0.5));
    viz2.addPointCloud("pcd2", cloud2, Visualizer::RenderingProperties().setDrawingColor(0,0,1).setOpacity(0.4));
    viz2.addPointCorrespondences("corr", cloud, cloud2, Visualizer::RenderingProperties().setCorrespondencesFraction(0.01).setOpacity(0.4));
    viz2.addCoordinateSystem("axis", 0.4f, Eigen::Matrix4f::Identity(), Visualizer::RenderingProperties().setLineWidth(10.0f));

    std::string name2 = "corr";
    viz2.registerKeyboardCallback(std::vector<int>(1,'c'), callback_test, &name2);

    while (!viz.wasStopped() && !viz2.wasStopped()) {
        viz.spinOnce();
        viz2.spinOnce();
    }

    return 0;
}
