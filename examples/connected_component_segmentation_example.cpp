#include <cilantro/connected_component_segmentation.hpp>
#include <cilantro/io.hpp>
#include <cilantro/visualizer.hpp>
#include <cilantro/voxel_grid.hpp>

int main(int argc, char ** argv) {

    cilantro::PointCloud cloud;
    cilantro::readPointCloudFromPLYFile(argv[1], cloud);

    cilantro::VoxelGrid vg(cloud, 0.005);
    cloud = vg.getDownsampledCloud().removeInvalidData();

//    NormalEstimation(cloud).estimateNormalsInPlaceRadius(0.02);

    std::cout << cloud.size() << std::endl;

    cilantro::ConnectedComponentSegmentation ccs(cloud);

    auto start = std::chrono::high_resolution_clock::now();

    ccs.segment(0.02, (float)(2.0*M_PI/180.0), 5.0, 100, cloud.size());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Segmentation time: " << elapsed.count() << "ms" << std::endl;

    std::cout << ccs.getComponentPointIndices().size() << " components found" << std::endl;

    size_t num_labels = ccs.getComponentPointIndices().size();
    std::vector<size_t> labels = ccs.getComponentIndexMap();

    std::vector<Eigen::Vector3f> color_map(num_labels+1);
    for (size_t i = 0; i < num_labels; i++) {
        color_map[i] = Eigen::Vector3f::Random().array().abs();
    }
    color_map[num_labels] = Eigen::Vector3f(0, 0, 0);

    std::vector<Eigen::Vector3f> cols(labels.size());
    for (size_t i = 0; i < cols.size(); i++) {
        cols[i] = color_map[labels[i]];
    }

    cilantro::Visualizer viz("win", "disp");

    viz.addPointCloud("cloud", cloud, cilantro::RenderingProperties().setColormapType(cilantro::ColormapType::JET));
    viz.addPointCloudColors("cloud", cols);

    while (!viz.wasStopped()) {
        viz.spinOnce();
    }

    return 0;
}
