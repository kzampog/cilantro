#include <cilantro/spectral_clustering.hpp>
#include <cilantro/point_cloud.hpp>
#include <cilantro/visualizer.hpp>
#include <cilantro/voxel_grid.hpp>

Eigen::MatrixXf build_dense_knn_affinity_graph(const cilantro::ConstVectorSetMatrixMap<float,3> &points, size_t k) {
    Eigen::MatrixXf graph(Eigen::MatrixXf::Zero(points.cols(), points.cols()));

    cilantro::VectorSet<float,3> tree_data = points.topRows(3);
    cilantro::KDTree<float,3> tree(tree_data);
    std::vector<size_t> neighbors;
    std::vector<float> distances;
#pragma omp parallel for shared (graph) private (neighbors, distances)
    for (size_t i = 0; i < points.cols(); i++) {
        tree.kNNSearch(points.col(i).head(3), k+1, neighbors, distances);
        for (size_t j = 0; j < neighbors.size(); j++) {
//            float val = std::exp(-1.0f*(points.col(i) - points.col(neighbors[j])).squaredNorm());
            float val = 1.0f;
            graph(i,neighbors[j]) = val;
            graph(neighbors[j],i) = val;
        }
    }

//    std::cout << (graph - graph.transpose()).cwiseAbs().maxCoeff() << std::endl;

    return graph;
}

Eigen::MatrixXf build_dense_radius_affinity_graph(const cilantro::ConstVectorSetMatrixMap<float,3> &points, float radius) {
    Eigen::MatrixXf graph(Eigen::MatrixXf::Zero(points.cols(), points.cols()));

    float radius_sq = radius*radius;

    cilantro::VectorSet<float,3> tree_data = points.topRows(3);
    cilantro::KDTree<float,3> tree(tree_data);
    std::vector<size_t> neighbors;
    std::vector<float> distances;
#pragma omp parallel for shared (graph) private (neighbors, distances)
    for (size_t i = 0; i < points.cols(); i++) {
        tree.radiusSearch(points.col(i).head(3), radius_sq, neighbors, distances);
        for (size_t j = 0; j < neighbors.size(); j++) {
//            float val = std::exp(-1.0f*(points.col(i) - points.col(neighbors[j])).squaredNorm());
            float val = 1.0f;
            graph(i,neighbors[j]) = val;
            graph(neighbors[j],i) = val;
        }
    }

//    std::cout << (graph - graph.transpose()).cwiseAbs().maxCoeff() << std::endl;

    return graph;
}

int main(int argc, char ** argv) {

    cilantro::VectorSet<float,3> points(3, 1700);
    for (size_t i = 0; i < 1500; i++) {
        points.col(i).setRandom().normalize();
    }
    for (size_t i = 1500; i < 1700; i++) {
        points.col(i).setRandom().normalize();
        points.col(i) *= 0.3f;
    }
    points.row(2).array() += 4.0f;

//    Eigen::MatrixXf data0 = build_dense_radius_affinity_graph(cloud.points, 0.6);
    Eigen::MatrixXf data0 = build_dense_knn_affinity_graph(points, 30);

    Eigen::SparseMatrix<float> data = data0.sparseView();

    std::cout << "Number of points: " << points.cols() << std::endl;

    size_t max_num_clusters = 4;

    auto start = std::chrono::high_resolution_clock::now();

//    cilantro::SpectralClustering<float,2> sc(data);
//    cilantro::SpectralClustering<float,Eigen::Dynamic> sc(data, max_num_clusters, true, cilantro::GraphLaplacianType::UNNORMALIZED);
//    cilantro::SpectralClustering<float,Eigen::Dynamic> sc(data, max_num_clusters, true, cilantro::GraphLaplacianType::NORMALIZED_SYMMETRIC);
    cilantro::SpectralClustering<float,Eigen::Dynamic> sc(data, max_num_clusters, true, cilantro::GraphLaplacianType::NORMALIZED_RANDOM_WALK);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Clustering time: " << elapsed.count() << "ms" << std::endl;
    std::cout << "Number of clusters: " << sc.getNumberOfClusters() << std::endl;
    std::cout << "Performed k-means iterations: " << sc.getClusterer().getPerformedIterationsCount() << std::endl;

    const std::vector<std::vector<size_t> >& cpi(sc.getClusterPointIndices());
    size_t mins = points.cols(), maxs = 0;
    for (size_t i = 0; i < cpi.size(); i++) {
        if (cpi[i].size() < mins) mins = cpi[i].size();
        if (cpi[i].size() > maxs) maxs = cpi[i].size();
    }
    std::cout << "Cluster size range is: [" << mins << "," << maxs << "]" << std::endl;

    // Create a color map
    std::vector<Eigen::Vector3f> color_map(max_num_clusters);
    for (size_t i = 0; i < max_num_clusters; i++) {
        color_map[i] = Eigen::Vector3f::Random().array().abs();
    }

    const std::vector<size_t>& idx_map(sc.getClusterIndexMap());

    std::vector<Eigen::Vector3f> cols(idx_map.size());
    for (size_t i = 0; i < cols.size(); i++) {
        cols[i] = color_map[idx_map[i]];
    }

    // Visualize result
    pangolin::CreateWindowAndBind("SpectralClustering demo",1280,480);
    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual).AddDisplay(pangolin::Display("disp1")).AddDisplay(pangolin::Display("disp2"));

    cilantro::Visualizer viz1("SpectralClustering demo", "disp1");
    viz1.addPointCloud("cloud", points, cilantro::RenderingProperties().setPointSize(5.0f));

    cilantro::Visualizer viz2("SpectralClustering demo", "disp2");
    viz2.addPointCloud("cloud_seg", points, cilantro::RenderingProperties().setPointSize(5.0f));
    viz2.addPointCloudColors("cloud_seg", cols);

    while (!viz1.wasStopped() && !viz2.wasStopped()) {
        viz1.clearRenderArea();
        viz1.render();
        viz2.render();
        pangolin::FinishFrame();
    }

    return 0;
}
