#include <random>
#include <cilantro/clustering/mean_shift.hpp>
#include <cilantro/visualization.hpp>
#include <cilantro/utilities/timer.hpp>

cilantro::VectorSet3f generate_input_data() {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f,1.0f);

    size_t cluster_size = 500;
    size_t cluster_num = 3;
    cilantro::VectorSet<float,3> points(3, cluster_num*cluster_size);
    cilantro::VectorSet<float,3> offsets(3, cluster_num);

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

    for (size_t i = 0; i < cluster_num; i++) {
        for (size_t j = 0; j < cluster_size; j++) {
            points.col(i*cluster_size + j) += offsets.col(i);
        }
    }

    return points;
}

int main(int argc, char ** argv) {
    // Generate random points
    cilantro::VectorSet3f points = generate_input_data();

    std::cout << "Number of points: " << points.cols() << std::endl;

    cilantro::MeanShift3f<> ms(points);

    cilantro::Timer timer;
    timer.start();
    // Flat kernel
    ms.cluster(2.0f, 5000, 0.2, 1e-7, cilantro::UnityWeightEvaluator<float>());
    timer.stop();

    std::cout << "Clustering time: " << timer.getElapsedTime() << "ms" << std::endl;
    std::cout << "Number of clusters: " << ms.getNumberOfClusters() << std::endl;
    std::cout << "Performed mean shift iterations: " << ms.getNumberOfPerformedIterations() << std::endl;

    const auto& cpi = ms.getClusterToPointIndicesMap();
    size_t mins = points.cols(), maxs = 0;
    for (size_t i = 0; i < cpi.size(); i++) {
        if (cpi[i].size() < mins) mins = cpi[i].size();
        if (cpi[i].size() > maxs) maxs = cpi[i].size();
    }
    std::cout << "Cluster size range is: [" << mins << "," << maxs << "]" << std::endl;

    // Create a color map
    cilantro::VectorSet3f color_map(3, cpi.size());
    for (size_t i = 0; i < cpi.size(); i++) {
        color_map.col(i) = Eigen::Vector3f::Random().cwiseAbs();
    }

    const auto& idx_map = ms.getPointToClusterIndexMap();
    cilantro::VectorSet3f colors(3, idx_map.size());
    for (size_t i = 0; i < colors.cols(); i++) {
        colors.col(i) = color_map.col(idx_map[i]);
    }

    // Visualize result
    pangolin::CreateWindowAndBind("MeanShift demo",1280,480);
    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual)
        .AddDisplay(pangolin::Display("disp1")).AddDisplay(pangolin::Display("disp2"));

    cilantro::Visualizer viz1("MeanShift demo", "disp1");
    viz1.addObject<cilantro::PointCloudRenderable>("cloud", points, cilantro::RenderingProperties().setPointSize(5.0f));

    cilantro::Visualizer viz2("MeanShift demo", "disp2");
    viz2.addObject<cilantro::PointCloudRenderable>("cloud_seg", points, cilantro::RenderingProperties().setPointSize(5.0f))
            ->setPointColors(colors);

    viz2.addObject<cilantro::PointCloudRenderable>("modes", ms.getClusterModes(), cilantro::RenderingProperties().setPointSize(20.0f))
            ->setPointColors(color_map);

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
