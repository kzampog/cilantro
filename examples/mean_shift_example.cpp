#include <random>
#include <cilantro/mean_shift.hpp>
#include <cilantro/visualizer.hpp>

int main(int argc, char ** argv) {
    // Generate random points
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f,1.0f);

    size_t c_sz = 500;
    size_t c_n = 3;
    cilantro::VectorSet<float,3> points(3, c_n*c_sz);
    cilantro::VectorSet<float,3> offsets(3, c_n);

    for (size_t j = 0; j < points.cols(); j++) {
        for (size_t i = 0; i < points.rows(); i++) {
            points(i,j) = distribution(generator);
        }
    }
    points.row(2).array() += 10.0f;

    for (size_t j = 0; j < offsets.cols(); j++) {
        for (size_t i = 0; i < offsets.rows(); i++) {
            offsets(i,j) = distribution(generator);
        }
        offsets.col(j) = 2.5f*offsets.col(j).normalized();
    }

    for (size_t i = 0; i < c_n; i++) {
        for (size_t j = 0; j < c_sz; j++) {
            points.col(i*c_sz + j) += offsets.col(i);
        }
    }

    std::cout << "Number of points: " << points.cols() << std::endl;

    cilantro::MeanShift3f ms(points);

    auto start = std::chrono::high_resolution_clock::now();
    ms.cluster(2.0f, 5000, 0.2, 1e-7);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Clustering time: " << elapsed.count() << "ms" << std::endl;
    std::cout << "Number of clusters: " << ms.getNumberOfClusters() << std::endl;
    std::cout << "Performed mean shift iterations: " << ms.getPerformedIterationsCount() << std::endl;

    const std::vector<std::vector<size_t> >& cpi(ms.getClusterPointIndices());
    size_t mins = points.cols(), maxs = 0;
    for (size_t i = 0; i < cpi.size(); i++) {
        if (cpi[i].size() < mins) mins = cpi[i].size();
        if (cpi[i].size() > maxs) maxs = cpi[i].size();
    }
    std::cout << "Cluster size range is: [" << mins << "," << maxs << "]" << std::endl;

    // Create a color map
    std::vector<Eigen::Vector3f> color_map(cpi.size());
    for (size_t i = 0; i < cpi.size(); i++) {
        color_map[i] = Eigen::Vector3f::Random().array().abs();
    }

    const std::vector<size_t>& idx_map(ms.getClusterIndexMap());

    std::vector<Eigen::Vector3f> cols(idx_map.size());
    for (size_t i = 0; i < cols.size(); i++) {
        cols[i] = color_map[idx_map[i]];
    }

    // Visualize result
    pangolin::CreateWindowAndBind("MeanShift demo",1280,480);
    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual).AddDisplay(pangolin::Display("disp1")).AddDisplay(pangolin::Display("disp2"));

    cilantro::Visualizer viz1("MeanShift demo", "disp1");
    viz1.addPointCloud("cloud", points, cilantro::RenderingProperties().setPointSize(5.0f));

    cilantro::Visualizer viz2("MeanShift demo", "disp2");
    viz2.addPointCloud("cloud_seg", points, cilantro::RenderingProperties().setPointSize(5.0f));
    viz2.addPointCloudColors("cloud_seg", cols);
    viz2.addPointCloud("modes", ms.getClusterModes(), cilantro::RenderingProperties().setPointSize(20.0f));
    viz2.addPointCloudColors("modes", color_map);

    while (!viz1.wasStopped() && !viz2.wasStopped()) {
        viz1.clearRenderArea();
        viz1.render();
        viz2.render();
        pangolin::FinishFrame();
    }

    return 0;
}
