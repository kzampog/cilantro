#include <cilantro/normal_estimation.hpp>
#include <cilantro/io.hpp>
#include <cilantro/visualizer.hpp>
#include <cilantro/voxel_grid.hpp>

int main(int argc, char ** argv) {
    cilantro::PointCloud3D cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    // Clear input normals
    cloud.normals.resize(Eigen::NoChange, 0);

    cilantro::VoxelGrid vg(cloud, 0.005);
    cloud = vg.getDownsampledCloud();

    auto start = std::chrono::high_resolution_clock::now();
//    cilantro::KDTree3D tree(cloud.points);
//    cilantro::NormalEstimation3D ne(cloud.points, tree);
    cilantro::NormalEstimation3D ne(cloud.points);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> kd_tree_time = end - start;

    start = std::chrono::high_resolution_clock::now();
//    cloud.normals = ne.estimateNormals(cilantro::KDTree3D::Neighborhood(cilantro::KDTree3D::NeighborhoodType::KNN_IN_RADIUS, 7, 0.01));
//    cloud.normals = ne.estimateNormalsKNNInRadius(7, 0.01);
//    cloud.normals = ne.estimateNormalsRadius(0.01);
    cloud.normals = ne.estimateNormalsKNN(7);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> estimation_time = end - start;

    std::cout << "kd-tree time: " << kd_tree_time.count() << "ms" << std::endl;
    std::cout << "Estimation time: " << estimation_time.count() << "ms" << std::endl;

    cilantro::Visualizer viz("NormalEstimation example", "disp");

    viz.addPointCloud("cloud_d", cloud, cilantro::RenderingProperties().setDrawNormals(true));

    std::cout << "Press 'n' to toggle rendering of normals" << std::endl;
    while (!viz.wasStopped()){
        viz.spinOnce();
    }

    return 0;
}
