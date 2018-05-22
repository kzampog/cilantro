#include <cilantro/common_similarity_evaluators.hpp>
#include <cilantro/connected_component_segmentation.hpp>
#include <cilantro/point_cloud.hpp>
#include <cilantro/visualizer.hpp>
#include <cilantro/common_renderables.hpp>
#include <cilantro/timer.hpp>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cout << "Please provide path to PLY file" << std::endl;
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

    cilantro::ConnectedComponentSegmentation ccs;

    cilantro::NeighborhoodSpecification<float> nh(cilantro::NeighborhoodType::RADIUS, 32, 0.02f*0.02f);
//    cilantro::NormalsColorsProximityEvaluator<float,3> ev(cloud.normals, cloud.colors, (float)(2.0*M_PI/180.0), 0.1f);
    cilantro::NormalsProximityEvaluator<float,3> ev(cloud.normals, (float)(2.0*M_PI/180.0));
    ccs.segment<float,3>(cloud.points, nh, ev, 100, cloud.size());

//    std::vector<cilantro::NearestNeighborSearchResultSet<float>> nn;
//    cilantro::KDTree3f(cloud.points).radiusSearch(cloud.points, 0.02f*0.02f, nn);
//    ccs.segment(nn);
    timer.stop();

    std::cout << "Segmentation time: " << timer.getElapsedTime() << "ms" << std::endl;
    std::cout << ccs.getComponentPointIndices().size() << " components found" << std::endl;

    // Build a color map
    size_t num_labels = ccs.getComponentPointIndices().size();
    std::vector<size_t> labels = ccs.getComponentIndexMap();

    std::vector<Eigen::Vector3f> color_map(num_labels+1);
    for (size_t i = 0; i < num_labels; i++) {
        color_map[i] = Eigen::Vector3f::Random().array().abs();
    }
    color_map[num_labels] = Eigen::Vector3f(0, 0, 0);   // No label

    cilantro::VectorSet<float,3> cols(3,labels.size());
    for (size_t i = 0; i < cols.cols(); i++) {
        cols.col(i) = color_map[labels[i]];
    }

    // Create a new colored cloud
    cilantro::PointCloud3f cloud_seg(cloud.points, cloud.normals, cols);

    // Visualize result
    pangolin::CreateWindowAndBind("ConnectedComponentSegmentation demo",1280,480);
    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual).AddDisplay(pangolin::Display("disp1")).AddDisplay(pangolin::Display("disp2"));

    cilantro::Visualizer viz1("ConnectedComponentSegmentation demo", "disp1");
    viz1.addObject<cilantro::PointCloudRenderable>("cloud", cloud);

    cilantro::Visualizer viz2("ConnectedComponentSegmentation demo", "disp2");
    viz2.addObject<cilantro::PointCloudRenderable>("cloud_seg", cloud_seg);

    while (!viz1.wasStopped() && !viz2.wasStopped()) {
        viz1.clearRenderArea();
        viz1.render();
        viz2.render();
        pangolin::FinishFrame();
    }

    return 0;
}
