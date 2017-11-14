#include <cilantro/normal_estimation.hpp>
#include <cilantro/io.hpp>
#include <cilantro/visualizer.hpp>
#include <cilantro/voxel_grid.hpp>

int main(int argc, char ** argv) {
    cilantro::PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    cloud.normals.clear();

    cilantro::VoxelGrid vg(cloud, 0.005);
    cloud = vg.getDownsampledCloud();

    auto start = std::chrono::high_resolution_clock::now();
    cilantro::KDTree3D tree(cloud.points);
    cilantro::NormalEstimation ne(cloud, tree);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> kd_tree_time = end - start;

    start = std::chrono::high_resolution_clock::now();
//    ne.estimateNormalsInPlace(KDTree3D::Neighborhood(KDTree3D::NeighborhoodType::KNN_IN_RADIUS, 7, 0.01));
//    ne.estimateNormalsInPlaceKNNInRadius(7, 0.01);
//    ne.estimateNormalsInPlaceRadius(0.01);
    ne.estimateNormalsInPlaceKNN(7);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> estimation_time = end - start;

    std::cout << "kd-tree time: " << kd_tree_time.count() << "ms" << std::endl;
    std::cout << "Estimation time: " << estimation_time.count() << "ms" << std::endl;

    cilantro::Visualizer viz("win", "disp");

    viz.addPointCloud("cloud_d", cloud, cilantro::RenderingProperties().setDrawNormals(true));

    while (!viz.wasStopped()){
        viz.spinOnce();
    }

    return 0;
}
