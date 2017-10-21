#include <cilantro/point_cloud.hpp>
#include <cilantro/plane_estimator.hpp>
#include <cilantro/ply_io.hpp>
#include <cilantro/visualizer.hpp>

bool re_estimate = false;

void callback(cilantro::Visualizer &viz, int key, void *cookie) {
    if (key == 'a') {
        re_estimate = true;
    }
}

int main(int argc, char **argv) {

    cilantro::PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    cilantro::Visualizer viz("win", "disp");
    viz.registerKeyboardCallback(std::vector<int>(1,'a'), callback, NULL);

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
            viz.addPointCloud("plane", planar_cloud, cilantro::RenderingProperties().setDrawingColor(1,0,0).setPointSize(3.0));

            std::cout << "Press 'a' for a new estimate" << std::endl;
        }
        viz.spinOnce();
    }

    return 0;
}
