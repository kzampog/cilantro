#include <cilantro/kmeans.hpp>
#include <cilantro/ply_io.hpp>
#include <cilantro/visualizer.hpp>
#include <cilantro/voxel_grid.hpp>

int main(int argc, char ** argv) {

    PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    cloud = VoxelGrid(cloud, 0.005).getDownsampledCloud().removeInvalidData();

//    Eigen::MatrixXf data_points(6,cloud.size());
//    data_points.topRows(3) = cloud.pointsMatrixMap();
//    data_points.bottomRows(3) = 0.15*cloud.colorsMatrixMap();
//    KMeans<float, 6, KDTreeDistanceAdaptors::L2> kmc(data_points);

//    KMeans<float,3,KDTreeDistanceAdaptors::L1> kmc(cloud.points);
    KMeans3D kmc(cloud.points);

    size_t k = 250;
    size_t max_iter = 100;
    float tol = std::numeric_limits<float>::epsilon();
    bool use_kd_tree = true;


    auto start = std::chrono::high_resolution_clock::now();

//    kmc.cluster(k, 1000);
    kmc.cluster(k, max_iter, tol, use_kd_tree);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Clustering time: " << elapsed.count() << "ms" << std::endl;


    std::cout << "Performed iterations: " << kmc.getPerformedIterationsCount() << std::endl;

    const std::vector<std::vector<size_t> >& cpi(kmc.getClusterPointIndices());

    size_t mins = cloud.size(), maxs = 0;
    for (size_t i = 0; i < cpi.size(); i++) {
        if (cpi[i].size() < mins) mins = cpi[i].size();
        if (cpi[i].size() > maxs) maxs = cpi[i].size();
    }

    std::cout << "Cluster size range is: [" << mins << "," << maxs << "]" << std::endl;

    Visualizer viz("win", "disp");

    std::vector<Eigen::Vector3f> color_map(k);
    for (size_t i = 0; i < k; i++) {
        color_map[i] = Eigen::Vector3f::Random().array().abs();
    }

    const std::vector<size_t>& idx_map(kmc.getClusterIndexMap());

    std::vector<Eigen::Vector3f> cols(idx_map.size());
    for (size_t i = 0; i < cols.size(); i++) {
        cols[i] = color_map[idx_map[i]];
    }

    viz.addPointCloud("cloud", cloud);
    viz.addPointCloudColors("cloud", cols);

    viz.addPointCloud("centroids", kmc.getClusterCentroids(), RenderingProperties().setPointSize(5.0f));
    viz.addPointCloudColors("centroids", std::vector<Eigen::Vector3f>(kmc.getClusterCentroids().size(), Eigen::Vector3f(1,1,1)));

    while (!viz.wasStopped()) {
        viz.spinOnce();
    }

    return 0;
}
