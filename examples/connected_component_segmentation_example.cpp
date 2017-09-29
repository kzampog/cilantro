#include <cilantro/connected_component_segmentation.hpp>
#include <cilantro/ply_io.hpp>
#include <cilantro/visualizer.hpp>

#include <cilantro/voxel_grid.hpp>

#include <ctime>

int main(int argc, char ** argv) {

    PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    VoxelGrid vg(cloud, 0.005);
    cloud = vg.getDownsampledCloud();

//    NormalEstimation(cloud).estimateNormalsInPlaceRadius(0.02);

    std::cout << cloud.size() << std::endl;

    ConnectedComponentSegmentation ccs(cloud);

    clock_t begin, end;
    double elapsed_time;
    begin = clock();

//    ccs.segment(std::vector<size_t>(1, 10000), 0.02, (float)(2.0*M_PI/180.0), 5.0, 100, cloud.size(), 0.0);
    ccs.segment(0.02, (float)(2.0*M_PI/180.0), 5.0, 100, cloud.size());
//    ccs.segment(0.02, (float)(2.0*M_PI/180.0), 5.0, 0, cloud.size(), 0.0);

    end = clock();
    elapsed_time = 1000.0*double(end - begin) / CLOCKS_PER_SEC;

    std::cout << "Segmentation time: " << elapsed_time << std::endl;

    std::cout << ccs.getComponentPointIndices().size() << std::endl;

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

    Visualizer viz("win", "disp");

    viz.addPointCloud("cloud", cloud, RenderingProperties().setColormapType(ColormapType::JET));
    viz.addPointCloudColors("cloud", cols);

    while (!viz.wasStopped()) {
        viz.spinOnce();
    }

    return 0;
}
