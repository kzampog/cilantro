#include <cilantro/point_cloud.hpp>
#include <cilantro/plane_estimator.hpp>

#include <cilantro/ply_io.hpp>
#include <cilantro/visualizer.hpp>

#include <iostream>

bool re_estimate = false;

void callback(Visualizer &viz, int key, void *cookie) {
    if (key == 'a') {
        re_estimate = true;
    }
}

int main(int argc, char **argv) {

    PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    Visualizer viz("win", "disp");
    viz.registerKeyboardCallback(std::vector<int>(1,'a'), callback, NULL);

    PlaneParameters plane;
    std::vector<size_t> inliers;

    viz.addPointCloud("cloud", cloud);
    while (!viz.wasStopped()) {
        if (re_estimate) {
            re_estimate = false;

            PlaneEstimator pe(cloud);
            pe.setMaxInlierResidual(0.01).setTargetInlierCount((size_t)(0.15*cloud.size())).setMaxNumberOfIterations(250).setReEstimationStep(true);
            pe.getModel(plane, inliers);
            std::cout << "RANSAC iterations: " << pe.getPerformedIterationsCount() << ", inlier count: " << pe.getNumberOfInliers() << std::endl;

            PointCloud planar_cloud(cloud, inliers);
            viz.addPointCloud("plane", planar_cloud, Visualizer::RenderingProperties().setDrawingColor(1,0,0).setPointSize(5.0));
        }
        viz.spinOnce();
    }

    return 0;
}
