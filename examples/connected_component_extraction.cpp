#include <cilantro/clustering/connected_component_extraction.hpp>
#include <cilantro/utilities/point_cloud.hpp>
#include <cilantro/utilities/timer.hpp>
#include <cilantro/visualization.hpp>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cout << "Please provide path to PLY file." << std::endl;
        return 0;
    }

    cilantro::PointCloud3f cloud(argv[1]);
    cloud.gridDownsample(0.005f).removeInvalidData();

    if (!cloud.hasNormals()) {
        std::cout << "Input cloud does not have normals!" << std::endl;
        return 0;
    }

    // Perform segmentation
    cilantro::Timer timer;
    timer.start();

    cilantro::RadiusNeighborhoodSpecification<float> nh(0.02f*0.02f);
    cilantro::NormalsProximityEvaluator<float,3> ev(cloud.normals, (float)(2.0*M_PI/180.0));

    cilantro::ConnectedComponentExtraction3f<> cce(cloud.points);
    cce.segment(nh, ev, 100, cloud.size());

    timer.stop();

    std::cout << "Segmentation time: " << timer.getElapsedTime() << "ms" << std::endl;
    std::cout << cce.getNumberOfClusters() << " components found" << std::endl;

    // Build a color map
    size_t num_labels = cce.getNumberOfClusters();
    const auto& labels = cce.getPointToClusterIndexMap();

    cilantro::VectorSet3f color_map(3, num_labels+1);
    for (size_t i = 0; i < num_labels; i++) {
        color_map.col(i) = Eigen::Vector3f::Random().cwiseAbs();
    }
    // No label
    color_map.col(num_labels).setZero();

    cilantro::VectorSet3f colors(3, labels.size());
    for (size_t i = 0; i < colors.cols(); i++) {
        colors.col(i) = color_map.col(labels[i]);
    }

    // Create a new colored cloud
    cilantro::PointCloud3f cloud_seg(cloud.points, cloud.normals, colors);

    // Visualize result
    pangolin::CreateWindowAndBind("ConnectedComponentSegmentation demo", 1280, 480);
    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual)
        .AddDisplay(pangolin::Display("disp1")).AddDisplay(pangolin::Display("disp2"));

    cilantro::Visualizer viz1("ConnectedComponentSegmentation demo", "disp1");
    viz1.addObject<cilantro::PointCloudRenderable>("cloud", cloud);

    cilantro::Visualizer viz2("ConnectedComponentSegmentation demo", "disp2");
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
