#include <cilantro/point_cloud.hpp>
#include <cilantro/plane_estimator.hpp>
#include <cilantro/io.hpp>
#include <cilantro/visualizer.hpp>

void callback(bool &re_estimate) {
    re_estimate = true;
}

int main(int argc, char **argv) {
    cilantro::PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    cilantro::Visualizer viz("PlaneEstimator example", "disp");
    bool re_estimate = false;
    viz.registerKeyboardCallback('a', std::bind(callback, std::ref(re_estimate)));

    std::cout << "Press 'a' for a new estimate" << std::endl;

    viz.addPointCloud("cloud", cloud);
    while (!viz.wasStopped()) {
        if (re_estimate) {
            re_estimate = false;

            cilantro::PlaneEstimator pe(cloud);
            pe.setMaxInlierResidual(0.01).setTargetInlierCount((size_t)(0.15*cloud.size())).setMaxNumberOfIterations(250).setReEstimationStep(true);
            cilantro::PlaneParameters plane = pe.getModelParameters();
            std::vector<size_t> inliers = pe.getModelInliers();
            std::cout << "RANSAC iterations: " << pe.getPerformedIterationsCount() << ", inlier count: " << pe.getNumberOfInliers() << std::endl;

            cilantro::PointCloud planar_cloud(cloud, inliers);
            viz.addPointCloud("plane", planar_cloud, cilantro::RenderingProperties().setPointColor(1,0,0).setPointSize(3.0));

            std::cout << "Press 'a' for a new estimate" << std::endl;
        }
        viz.spinOnce();
    }

    return 0;
}
