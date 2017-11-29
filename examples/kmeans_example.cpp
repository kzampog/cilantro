#include <cilantro/kmeans.hpp>
#include <cilantro/io.hpp>
#include <cilantro/visualizer.hpp>
#include <cilantro/voxel_grid.hpp>

int main(int argc, char ** argv) {
    cilantro::PointCloud cloud;
    cilantro::readPointCloudFromPLYFile(argv[1], cloud);

    cloud = cilantro::VoxelGrid(cloud, 0.005).getDownsampledCloud().removeInvalidData();

//    Eigen::MatrixXf data_points(6,cloud.size());
//    data_points.topRows(3) = cloud.pointsMatrixMap();
//    data_points.bottomRows(3) = 0.15*cloud.colorsMatrixMap();
//    KMeans<float, 6, KDTreeDistanceAdaptors::L2> kmc(data_points);

//    KMeans<float,3,KDTreeDistanceAdaptors::L1> kmc(cloud.points);
    cilantro::KMeans3D kmc(cloud.points);

    size_t k = 250;
    size_t max_iter = 100;
    float tol = std::numeric_limits<float>::epsilon();
    bool use_kd_tree = true;


    auto start = std::chrono::high_resolution_clock::now();
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

    // Create a color map
    std::vector<Eigen::Vector3f> color_map(k);
    for (size_t i = 0; i < k; i++) {
        color_map[i] = Eigen::Vector3f::Random().array().abs();
    }

    const std::vector<size_t>& idx_map(kmc.getClusterIndexMap());

    std::vector<Eigen::Vector3f> cols(idx_map.size());
    for (size_t i = 0; i < cols.size(); i++) {
        cols[i] = color_map[idx_map[i]];
    }

    // Create a new colored cloud
    cilantro::PointCloud cloud_seg(cloud.points, cloud.normals, cols);

    // Visualize result
    pangolin::CreateWindowAndBind("KMeans demo",1280,480);
    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual).AddDisplay(pangolin::Display("disp1")).AddDisplay(pangolin::Display("disp2"));

    cilantro::Visualizer viz1("KMeans demo", "disp1");
    viz1.addPointCloud("cloud", cloud);

    cilantro::Visualizer viz2("KMeans demo", "disp2");
    viz2.addPointCloud("cloud_seg", cloud_seg);
    viz2.addPointCloud("centroids", kmc.getClusterCentroids(), cilantro::RenderingProperties().setPointSize(5.0f).setPointColor(1.0f,1.0f,1.0f));

    while (!viz1.wasStopped() && !viz2.wasStopped()) {
        viz1.clearRenderArea();
        viz1.render();
        viz2.render();
        pangolin::FinishFrame();
    }

    return 0;
}
