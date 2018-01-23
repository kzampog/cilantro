#include <cilantro/voxel_grid.hpp>
#include <cilantro/io.hpp>
#include <cilantro/visualizer.hpp>

int main(int argc, char ** argv) {
    cilantro::PointCloud3D cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    auto start = std::chrono::high_resolution_clock::now();
    cilantro::VoxelGrid vg(cloud, 0.01f);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> build_time = end - start;

    start = std::chrono::high_resolution_clock::now();
    cilantro::PointCloud3D cloud_d = vg.getDownsampledCloud();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ds_time = end - start;

    std::cout << "Before: " << cloud.size() << std::endl;
    std::cout << "After: " << cloud_d.size() << std::endl;

    std::cout << "Build time: " << build_time.count() << "ms" << std::endl;
    std::cout << "Downsampling time: " << ds_time.count() << "ms" << std::endl;

    pangolin::CreateWindowAndBind("VoxelGrid demo",1280,480);
    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual).AddDisplay(pangolin::Display("disp1")).AddDisplay(pangolin::Display("disp2"));

    cilantro::Visualizer viz1("VoxelGrid demo", "disp1");
    viz1.addPointCloud("cloud", cloud, cilantro::RenderingProperties());

    cilantro::Visualizer viz2("VoxelGrid demo", "disp2");
    viz2.addPointCloud("cloud_d", cloud_d, cilantro::RenderingProperties());

    std::cout << "Press 'n' to toggle rendering of normals" << std::endl;
    while (!viz1.wasStopped() && !viz2.wasStopped()) {
        viz1.clearRenderArea();
        viz1.render();
        viz2.render();
        pangolin::FinishFrame();
    }

    return 0;
}
