#include <cilantro/visualizer.hpp>
#include <cilantro/ply_io.hpp>
#include <cilantro/voxel_grid.hpp>

#include <iostream>
#include <ctime>

void callback_test(Visualizer &viz, int key, void *cookie) {
    std::string name = *((std::string *)cookie);
    std::cout << "Toggling visibility for " << name << std::endl;
    viz.toggleVisibilityStatus(name);
}

int main(int argc, char ** argv) {

    PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    VoxelGrid vg(cloud, 0.01);
    cloud = vg.getDownsampledCloud();

//    pangolin::CreateWindowAndBind("VIS_WIN",640,480);
//    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual)
//            .AddDisplay(pangolin::Display("disp1"))
//            .AddDisplay(pangolin::Display("disp2"));

    // First
    Visualizer viz("VIS_WIN", "disp1");

    std::vector<float> scalars (cloud.size());
    for (size_t i = 0; i < cloud.size(); i++)
        scalars[i] = cloud.points[i].norm();
    viz.addPointCloud("pcd", cloud, RenderingProperties().setColormapType(ColormapType::JET));
    viz.addPointCloudValues("pcd", scalars);
    viz.addPointCloudNormals("nrm", cloud, RenderingProperties().setCorrespondencesFraction(0.20).setOpacity(0.5));

    viz.addCoordinateSystem("axis", 0.4f, Eigen::Matrix4f::Identity(), RenderingProperties().setLineWidth(10.0f));

    std::string name = "nrm";
    viz.registerKeyboardCallback(std::vector<int>(1,'n'), callback_test, &name);

    // Second
    PointCloud cloud2(cloud);
    for (size_t i = 0; i < cloud2.size(); i++) {
        cloud2.points[i] += Eigen::Vector3f(1.0, 0.0, 1.0);
    }

    Visualizer viz2("VIS_WIN2", "disp2");
    viz2.addPointCloud("pcd1", cloud, RenderingProperties().setDrawingColor(1,0,0).setOpacity(0.5));
    viz2.addPointCloud("pcd2", cloud2, RenderingProperties().setDrawingColor(0,0,1).setOpacity(0.4));
    viz2.addPointCorrespondences("corr", cloud, cloud2, RenderingProperties().setCorrespondencesFraction(0.01).setOpacity(0.4));
    viz2.addCoordinateSystem("axis", 0.4f, Eigen::Matrix4f::Identity(), RenderingProperties().setLineWidth(10.0f));

    std::string name2 = "corr";
    viz2.registerKeyboardCallback(std::vector<int>(1,'c'), callback_test, &name2);

    clock_t t0 = clock();

    Eigen::Vector3f cam_pos(0,0,0), look_at(0,0,1), up_dir(0,-1,0);

    while (!viz.wasStopped() && !viz2.wasStopped()) {
        double t = 1000.0*double(clock() - t0)/CLOCKS_PER_SEC;

        cam_pos(0) = 0.0 + 2.0*std::cos(t/20.0);
        cam_pos(2) = 1.0 + 2.0*std::sin(t/20.0);
        cam_pos(1) = 0.3*std::sin(t/10.0);

        viz.setCameraPose(cam_pos, look_at, up_dir);

        viz.spinOnce();
        viz2.spinOnce();

    }

    return 0;
}
