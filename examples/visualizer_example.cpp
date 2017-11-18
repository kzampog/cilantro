#include <cilantro/visualizer.hpp>
#include <cilantro/io.hpp>
#include <cilantro/voxel_grid.hpp>

void callback_test(cilantro::Visualizer &viz, int key, void *cookie) {
    std::string name = *((std::string *)cookie);
    std::cout << "Toggling visibility for " << name << std::endl;
    viz.toggleVisibilityStatus(name);
}

int main(int argc, char ** argv) {

    cilantro::PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    cilantro::VoxelGrid vg(cloud, 0.01);
    cloud = vg.getDownsampledCloud();

//    pangolin::CreateWindowAndBind("VIS_WIN",640,480);
//    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual)
//            .AddDisplay(pangolin::Display("disp1"))
//            .AddDisplay(pangolin::Display("disp2"));

    // First
    cilantro::Visualizer viz("VIS_WIN", "disp1");

    std::vector<float> scalars (cloud.size());
    for (size_t i = 0; i < cloud.size(); i++)
        scalars[i] = cloud.points[i].norm();
    viz.addPointCloud("pcd", cloud, cilantro::RenderingProperties().setColormapType(cilantro::ColormapType::JET).setLineDensityFraction(0.2f).setOpacity(0.5).setDrawNormals(true));
    viz.addPointCloudValues("pcd", scalars);
    viz.addCoordinateFrame("axis", 0.4f, Eigen::Matrix4f::Identity(), cilantro::RenderingProperties().setLineWidth(10.0f));
    viz.addText("text", "Coordinate Frame", 0, 0, 0, cilantro::RenderingProperties().setFontSize(20.0f).setPointColor(1,1,0).setTextAnchorPoint(0.5,-1));

    // Second
    cilantro::PointCloud cloud2(cloud);
    for (size_t i = 0; i < cloud2.size(); i++) {
        cloud2.points[i] += Eigen::Vector3f(1.0, 0.0, 1.0);
    }

    cilantro::Visualizer viz2("VIS_WIN2", "disp2");
    viz2.addPointCloud("pcd1", cloud, cilantro::RenderingProperties().setPointColor(1,0,0).setOpacity(0.5));
    viz2.addPointCloud("pcd2", cloud2, cilantro::RenderingProperties().setPointColor(0,0,1).setOpacity(0.4));
    viz2.addPointCorrespondences("correspondences", cloud, cloud2, cilantro::RenderingProperties().setLineDensityFraction(0.01).setOpacity(0.4));
    viz2.addCoordinateFrame("axis", 0.4f, Eigen::Matrix4f::Identity(), cilantro::RenderingProperties().setLineWidth(10.0f));

    std::string name = "correspondences";
    viz2.registerKeyboardCallback(std::vector<int>(1,'c'), callback_test, &name);

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
