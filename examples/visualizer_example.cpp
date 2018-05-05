#include <cilantro/visualizer.hpp>
#include <cilantro/common_renderables.hpp>
#include <cilantro/io.hpp>

void callback(cilantro::Visualizer &viz, const std::string &name) {
    std::cout << "Toggling visibility for " << name << std::endl;
    viz.toggleVisibility(name);
}

int main(int argc, char ** argv) {
    cilantro::PointCloud3f cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

//    cilantro::VoxelGrid vg(cloud, 0.01);
//    cloud = vg.getDownsampledCloud();

//    pangolin::CreateWindowAndBind("VIS_WIN",640,480);
//    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual)
//            .AddDisplay(pangolin::Display("disp1"))
//            .AddDisplay(pangolin::Display("disp2"));

    // First
    cilantro::Visualizer viz1("Visualizer demo (window 1)", "disp");

    std::vector<float> scalars(cloud.size());
    for (size_t i = 0; i < cloud.size(); i++) {
        scalars[i] = cloud.points.col(i).norm();
    }

    viz1.addObject<cilantro::PointCloudRenderable>("pcd", cloud, cilantro::RenderingProperties().setColormapType(cilantro::ColormapType::JET).setLineDensityFraction(0.2f).setUseLighting(false))
            ->setPointValues(scalars);
    viz1.addObject<cilantro::CoordinateFrameRenderable>("axis", Eigen::Matrix4f::Identity(), 0.4f, cilantro::RenderingProperties().setLineWidth(5.0f));
    viz1.addObject<cilantro::TextRenderable>("text", "Coordinate Frame", 0, 0, 0, cilantro::RenderingProperties().setFontSize(20.0f).setPointColor(1.0f,1.0f,0.0f).setTextAnchorPoint(0.5f,-1.0f));

    // Second
    cilantro::PointCloud3f cloud2(cloud);
    for (size_t i = 0; i < cloud2.size(); i++) {
        cloud2.points.col(i) += Eigen::Vector3f(1.0, 0.0, 1.0);
    }

    cilantro::Visualizer viz2("Visualizer demo (window 2)", "disp");
    viz2.addObject<cilantro::PointCloudRenderable>("pcd1", cloud, cilantro::RenderingProperties().setPointColor(1.0f,0.0f,0.0f).setOpacity(0.4f));
    viz2.addObject<cilantro::PointCloudRenderable>("pcd2", cloud2, cilantro::RenderingProperties().setPointColor(0.0f,0.0f,1.0f).setOpacity(0.4f));
    viz2.addObject<cilantro::PointCorrespondencesRenderable>("correspondences", cloud, cloud2, cilantro::RenderingProperties().setLineDensityFraction(0.005).setOpacity(0.3f));
    viz2.addObject<cilantro::CoordinateFrameRenderable>("axis", Eigen::Matrix4f::Identity(), 0.4f, cilantro::RenderingProperties().setLineWidth(5.0f));

    viz2.registerKeyboardCallback('c', std::bind(callback, std::ref(viz2), "correspondences"));

    std::cout << "Press 'n' to toggle rendering of normals" << std::endl;
//    clock_t t0 = clock();
//    Eigen::Vector3f cam_pos(0,0,0), look_at(0,0,1), up_dir(0,-1,0);
    while (!viz1.wasStopped() && !viz2.wasStopped()) {
//        double t = 1000.0*double(clock() - t0)/CLOCKS_PER_SEC;
//
//        cam_pos(0) = 0.0 + 2.0*std::cos(t/20.0);
//        cam_pos(2) = 1.0 + 2.0*std::sin(t/20.0);
//        cam_pos(1) = 0.3*std::sin(t/10.0);
//
//        viz1.setCameraPose(cam_pos, look_at, up_dir);

        viz1.spinOnce();
        viz2.spinOnce();
    }

    return 0;
}
