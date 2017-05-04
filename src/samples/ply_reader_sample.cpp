#include "../include/vis.hpp"

#include "ply_reader.h"

using namespace ply_reader;

int main(int /*argc*/, char ** /*argv*/ ) {
    // Instantiate PLY reader
    ply_reader::PlyFile read_ply;

    // Create a pointcloud instance
    ply_reader::PointCloud pointCloud;

    //  Read dem cloudz
    read_ply.read_ply_file("vase-v2.ply", pointCloud);

    // Initialize visualization
    Vis vis;
    while (!pangolin::ShouldQuit()) {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Axis
        pangolin::glDrawAxis(1.0f);

        // Colored points
        pangolin::glDrawColoredVertices<Eigen::Vector3f, Eigen::Vector3f>(pointCloud.num_points_,
                                                                          pointCloud.points_.data(),
                                                                          pointCloud.colors_.data(),
                                                                          GL_POINTS);

        // Swap frames and Process Events
        pangolin::FinishFrame();
    }

    return 0;
}
