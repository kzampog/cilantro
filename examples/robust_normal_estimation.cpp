// #include <cilantro/core/normal_estimation.hpp>
#include <cilantro/utilities/point_cloud.hpp>
#include <cilantro/visualization.hpp>
#include <cilantro/utilities/timer.hpp>

void toggle_invalid(cilantro::Visualizer &viz) {
    viz.getObject("cloud_i")->toggleVisibility();
}


cilantro::PointCloud3f invalidNormalsCloud(const cilantro::PointCloud3f& cloud) {
    std::vector<size_t> ind_to_remove;
    ind_to_remove.reserve(cloud.normals.cols());
    for (size_t i = 0; i < cloud.normals.cols(); i++) {
        if (!cloud.normals.col(i).allFinite()) ind_to_remove.emplace_back(i);
    }
    cilantro::PointCloud3f invalid_cloud(cloud, ind_to_remove);
    invalid_cloud.normals.resize(Eigen::NoChange, 0);
    return invalid_cloud;
}

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

    // Clear input normals
    cloud.normals.resize(Eigen::NoChange, 0);

    cloud.gridDownsample(0.005f);

    cilantro::Timer tree_timer;
    tree_timer.start();
    cilantro::KDTree3f tree(cloud.points);
    tree_timer.stop();

    cilantro::Timer ne_timer;
    ne_timer.start();
    cilantro::NormalEstimation<float, 3, cilantro::MinimumCovarianceDeterminant<float, 3>> ne(tree);
    ne.setViewPoint(Eigen::Vector3f::Zero());
    ne.covarianceMethod().setChiSquareThreshold(6.25).setNumberOfTrials(2).setNumberOfRefinements(1);  // 90% confidence ellipsoid
    cloud.normals = ne.getNormalsKNN(12);

    cilantro::PointCloud3f invalid_cloud = invalidNormalsCloud(cloud);
    cloud.removeInvalidNormals();

    ne_timer.stop();

    std::cout << "kd-tree time: " << tree_timer.getElapsedTime() << "ms" << std::endl;
    std::cout << "Estimation time: " << ne_timer.getElapsedTime() << "ms" << std::endl;

    cilantro::Visualizer viz("NormalEstimation example", "disp");
    viz.registerKeyboardCallback('i', std::bind(toggle_invalid, std::ref(viz)));

    viz.addObject<cilantro::PointCloudRenderable>("cloud_d", cloud, cilantro::RenderingProperties().setDrawNormals(true));
    viz.addObject<cilantro::PointCloudRenderable>("cloud_i", invalid_cloud, cilantro::RenderingProperties().setPointColor(255, 0, 0));

    std::cout << "Press 'n' to toggle rendering of normals" << std::endl;
    std::cout << "Press 'i' to toggle rendering of outliers" << std::endl;
    while (!viz.wasStopped()){
        viz.spinOnce();
    }

    return 0;
}
