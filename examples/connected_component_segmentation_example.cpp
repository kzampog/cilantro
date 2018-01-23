#include <cilantro/connected_component_segmentation.hpp>
#include <cilantro/io.hpp>
#include <cilantro/visualizer.hpp>
#include <cilantro/voxel_grid.hpp>

int main(int argc, char ** argv) {
    cilantro::PointCloud3D cloud;
    cilantro::readPointCloudFromPLYFile(argv[1], cloud);

    cloud = cilantro::VoxelGrid(cloud,0.005).getDownsampledCloud().removeInvalidData();

    // Perform segmentation
    cilantro::ConnectedComponentSegmentation ccs(cloud.points, cloud.normals, cloud.colors);

    auto start = std::chrono::high_resolution_clock::now();
    ccs.segment(0.02, (float)(2.0*M_PI/180.0), 5.0, 100, cloud.size());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Segmentation time: " << elapsed.count() << "ms" << std::endl;
    std::cout << ccs.getComponentPointIndices().size() << " components found" << std::endl;

    // Build a color map
    size_t num_labels = ccs.getComponentPointIndices().size();
    std::vector<size_t> labels = ccs.getComponentIndexMap();

    std::vector<Eigen::Vector3f> color_map(num_labels+1);
    for (size_t i = 0; i < num_labels; i++) {
        color_map[i] = Eigen::Vector3f::Random().array().abs();
    }
    color_map[num_labels] = Eigen::Vector3f(0, 0, 0);   // No label

    cilantro::VectorSet<float,3> cols(3,labels.size());
    for (size_t i = 0; i < cols.cols(); i++) {
        cols.col(i) = color_map[labels[i]];
    }

    // Create a new colored cloud
    cilantro::PointCloud3D cloud_seg(cloud.points, cloud.normals, cols);

    // Visualize result
    pangolin::CreateWindowAndBind("ConnectedComponentSegmentation demo",1280,480);
    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual).AddDisplay(pangolin::Display("disp1")).AddDisplay(pangolin::Display("disp2"));

    cilantro::Visualizer viz1("ConnectedComponentSegmentation demo", "disp1");
    viz1.addPointCloud("cloud", cloud);

    cilantro::Visualizer viz2("ConnectedComponentSegmentation demo", "disp2");
    viz2.addPointCloud("cloud_seg", cloud_seg);

    while (!viz1.wasStopped() && !viz2.wasStopped()) {
        viz1.clearRenderArea();
        viz1.render();
        viz2.render();
        pangolin::FinishFrame();
    }

    return 0;
}
