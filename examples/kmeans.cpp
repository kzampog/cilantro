#include <cilantro/clustering/kmeans.hpp>
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

    cloud.gridDownsample(0.005f).removeInvalidData();

//    Eigen::MatrixXf data_points(6,cloud.size());
//    data_points.topRows(3) = cloud.pointsMatrixMap();
//    data_points.bottomRows(3) = 0.15*cloud.colorsMatrixMap();
//    KMeans<float, 6, KDTreeDistanceAdaptors::L2> kmc(data_points);

//    KMeans<float,3,KDTreeDistanceAdaptors::L1> kmc(cloud.points);

    // k-means on point coordinates
    cilantro::KMeans3f<> kmc(cloud.points);

    size_t k = 250;
    size_t max_iter = 100;
    float tol = std::numeric_limits<float>::epsilon();
    bool use_kd_tree = true;

    cilantro::Timer timer;
    timer.start();
    kmc.cluster(k, max_iter, tol, use_kd_tree);
    timer.stop();

    std::cout << "Clustering time: " << timer.getElapsedTime() << "ms" << std::endl;
    std::cout << "Performed iterations: " << kmc.getNumberOfPerformedIterations() << std::endl;

    const auto& cpi = kmc.getClusterToPointIndicesMap();
    size_t mins = cloud.size(), maxs = 0;
    for (size_t i = 0; i < cpi.size(); i++) {
        if (cpi[i].size() < mins) mins = cpi[i].size();
        if (cpi[i].size() > maxs) maxs = cpi[i].size();
    }
    std::cout << "Cluster size range is: [" << mins << "," << maxs << "]" << std::endl;

    // Create a color map
    cilantro::VectorSet3f color_map(3, k);
    for (size_t i = 0; i < k; i++) {
        color_map.col(i) = Eigen::Vector3f::Random().cwiseAbs();
    }

    const auto& idx_map = kmc.getPointToClusterIndexMap();

    cilantro::VectorSet<float,3> colors(3,idx_map.size());
    for (size_t i = 0; i < colors.cols(); i++) {
        colors.col(i) = color_map.col(idx_map[i]);
    }

    // Create a new colored cloud
    cilantro::PointCloud3f cloud_seg(cloud.points, cloud.normals, colors);

    // Visualize result
    pangolin::CreateWindowAndBind("KMeans demo",1280,480);
    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual)
        .AddDisplay(pangolin::Display("disp1")).AddDisplay(pangolin::Display("disp2"));

    cilantro::Visualizer viz1("KMeans demo", "disp1");
    viz1.addObject<cilantro::PointCloudRenderable>("cloud", cloud);

    cilantro::Visualizer viz2("KMeans demo", "disp2");
    viz2.addObject<cilantro::PointCloudRenderable>("cloud_seg", cloud_seg);

    // Keep viewpoints in sync
    viz2.setRenderState(viz1.getRenderState());

    while (!viz1.wasStopped() && !viz2.wasStopped()) {
        viz1.clearRenderArea();
        viz1.render();
        viz2.render();
        pangolin::FinishFrame();
    }

    return 0;
}
