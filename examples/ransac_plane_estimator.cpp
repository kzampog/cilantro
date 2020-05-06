#include <cilantro/utilities/point_cloud.hpp>
#include <cilantro/model_estimation/ransac_hyperplane_estimator.hpp>
#include <cilantro/visualization.hpp>

void callback(bool &re_estimate) {
    re_estimate = true;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Please provide path to PLY file." << std::endl;
        return 0;
    }

    cilantro::PointCloud3f cloud(argv[1]);

    if (cloud.isEmpty()) {
        std::cout << "Input cloud is empty!" << std::endl;
        return 0;
    }

    cilantro::Visualizer viz("HyperplaneRANSACEstimator example", "disp");
    bool re_estimate = false;
    viz.registerKeyboardCallback('a', std::bind(callback, std::ref(re_estimate)));

    std::cout << "Press 'a' for a new estimate" << std::endl;

    cilantro::PointCloud3f planar_cloud;

    viz.addObject<cilantro::PointCloudRenderable>("cloud", cloud);
    while (!viz.wasStopped()) {
        if (re_estimate) {
            re_estimate = false;

            cilantro::PlaneRANSACEstimator3f<> pe(cloud.points);
            pe.setMaxInlierResidual(0.01f).setTargetInlierCount((size_t)(0.15*cloud.size()))
                .setMaxNumberOfIterations(250).setReEstimationStep(true);

            Eigen::Hyperplane<float,3> plane = pe.estimate().getModel();
            const auto& inliers = pe.getModelInliers();

            std::cout << "RANSAC iterations: " << pe.getNumberOfPerformedIterations() << ", inlier count: " << pe.getNumberOfInliers() << std::endl;

            planar_cloud = cilantro::PointCloud3f(cloud, inliers);
            viz.addObject<cilantro::PointCloudRenderable>("plane", planar_cloud.points, cilantro::RenderingProperties().setPointColor(1,0,0).setPointSize(3.0));

            std::cout << "Press 'a' for a new estimate" << std::endl;
        }
        viz.spinOnce();
    }

    return 0;
}
