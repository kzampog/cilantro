#include <cilantro/nearest_neighbor_graph_utilities.hpp>
#include <cilantro/spectral_clustering.hpp>
#include <cilantro/point_cloud.hpp>
#include <cilantro/visualizer.hpp>
#include <cilantro/common_renderables.hpp>
#include <cilantro/timer.hpp>

struct AffinityEvaluator {
    inline float operator()(size_t i, size_t j, float dist) const {
        return std::exp(-dist/2.0f);
    }
};

int main(int argc, char ** argv) {
    // Generate dataset
    cilantro::VectorSet<float,3> points(3, 1700);
    for (size_t i = 0; i < 1500; i++) {
        points.col(i).setRandom().normalize();
    }
    for (size_t i = 1500; i < 1700; i++) {
        points.col(i).setRandom().normalize();
        points.col(i) *= 0.3f;
    }
    points.row(2).array() += 4.0f;

    std::cout << "Number of points: " << points.cols() << std::endl;

    // Build neighborhood graph
    // radius of 0.6
//    cilantro::NeighborhoodSpecification<float> nh(cilantro::NeighborhoodType::RADIUS, 0, 0.6f*0.6f);
    // 30 neighbors
    cilantro::NeighborhoodSpecification<float> nh(cilantro::NeighborhoodType::KNN, 30, 0.0f);

    std::vector<cilantro::NearestNeighborSearchResultSet<float>> nn;

    cilantro::KDTree3f(points).search(points, nh, nn);
    Eigen::SparseMatrix<float> data = cilantro::getNNGraphFunctionValueSparseMatrix(nn, AffinityEvaluator(), true);

    size_t max_num_clusters = 4;

    cilantro::Timer timer;
    timer.start();

//    cilantro::SpectralClustering<float,2> sc(data);
//    cilantro::SpectralClustering<float,Eigen::Dynamic> sc(data, max_num_clusters, true, cilantro::GraphLaplacianType::UNNORMALIZED);
//    cilantro::SpectralClustering<float,Eigen::Dynamic> sc(data, max_num_clusters, true, cilantro::GraphLaplacianType::NORMALIZED_SYMMETRIC);
    cilantro::SpectralClustering<float> sc(data, max_num_clusters, true, cilantro::GraphLaplacianType::NORMALIZED_RANDOM_WALK);

    timer.stop();
    std::cout << "Clustering time: " << timer.getElapsedTime() << "ms" << std::endl;
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
    viz1.addObject<cilantro::PointCloudRenderable>("cloud", points, cilantro::RenderingProperties().setPointSize(5.0f));

    cilantro::Visualizer viz2("SpectralClustering demo", "disp2");
    viz2.addObject<cilantro::PointCloudRenderable>("cloud_seg", points, cilantro::RenderingProperties().setPointSize(5.0f))
            ->setPointColors(cols);

    while (!viz1.wasStopped() && !viz2.wasStopped()) {
        viz1.clearRenderArea();
        viz1.render();
        viz2.render();
        pangolin::FinishFrame();
    }

    return 0;
}
