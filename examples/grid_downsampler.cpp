//#include <cilantro/core/grid_downsampler.hpp>
#include <cilantro/utilities/point_cloud.hpp>
#include <cilantro/visualization.hpp>
#include <cilantro/utilities/timer.hpp>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cout << "Please provide path to PLY file." << std::endl;
        return 0;
    }

    cilantro::PointCloud3f cloud(argv[1]);

    if (cloud.isEmpty()) {
        std::cout << "Input cloud is empty!" << std::endl;
        return 0;
    }

    cilantro::Timer timer;
    timer.start();
    cilantro::PointCloud3f cloud_d(cloud.gridDownsampled(0.01f));
    timer.stop();

    std::cout << "Before: " << cloud.size() << std::endl;
    std::cout << "After: " << cloud_d.size() << std::endl;

    std::cout << "Downsampling time: " << timer.getElapsedTime() << "ms" << std::endl;

    pangolin::CreateWindowAndBind("VoxelGrid demo",1280,480);
    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual).AddDisplay(pangolin::Display("disp1")).AddDisplay(pangolin::Display("disp2"));

    cilantro::Visualizer viz1("VoxelGrid demo", "disp1");
    viz1.addObject<cilantro::PointCloudRenderable>("cloud", cloud, cilantro::RenderingProperties());

    cilantro::Visualizer viz2("VoxelGrid demo", "disp2");
    viz2.addObject<cilantro::PointCloudRenderable>("cloud_d", cloud_d, cilantro::RenderingProperties());

    // Keep viewpoints in sync
    viz2.setRenderState(viz1.getRenderState());

    std::cout << "Press 'n' to toggle rendering of normals" << std::endl;
    while (!viz1.wasStopped() && !viz2.wasStopped()) {
        viz1.clearRenderArea();
        viz1.render();
        viz2.render();
        pangolin::FinishFrame();
    }

    return 0;
}
