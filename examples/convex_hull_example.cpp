#include <cilantro/convex_hull.hpp>
#include <cilantro/ply_io.hpp>
#include <cilantro/visualizer.hpp>

#include <iostream>

int main(int argc, char ** argv) {

    std::vector<Eigen::Vector3f> points1;
    points1.push_back(Eigen::Vector3f(0,0,0));
    points1.push_back(Eigen::Vector3f(1,0,0));
    points1.push_back(Eigen::Vector3f(0,1,0));
    points1.push_back(Eigen::Vector3f(0,0,1));
    points1.push_back(Eigen::Vector3f(0,1,1));
    points1.push_back(Eigen::Vector3f(1,0,1));
    points1.push_back(Eigen::Vector3f(1,1,0));
    points1.push_back(Eigen::Vector3f(1,1,1));
    points1.push_back(Eigen::Vector3f(0.5,0.5,0.5));

    std::cout << "V to H:" << std::endl;
    std::vector<Eigen::Vector3f> hull_points1;
    std::vector<Eigen::Vector4f> halfspaces1;
    std::vector<size_t> faces1;
    std::vector<size_t> hull_pt_indices1;

    VtoH<Eigen::Vector3f,Eigen::Vector3f,Eigen::Vector4f>(points1, hull_points1, halfspaces1, faces1, hull_pt_indices1, true);

    for (size_t i = 0; i < halfspaces1.size(); ++i) {
        std::cout << halfspaces1[i].transpose() << std::endl;
    }

//    std::vector<Eigen::Vector4f> faces2;
//    faces2.push_back(Eigen::Vector4f(-1,0,0,0));
//    faces2.push_back(Eigen::Vector4f(0,-1,0,0));
//    faces2.push_back(Eigen::Vector4f(0,0,-1,0));
//    faces2.push_back(Eigen::Vector4f(1,1,1,-1));

//    Eigen::Vector3f interior_pt(0.1, 0.1, 0.1);

//    Eigen::Vector3f interior_pt(0.5, 0.5, 0.5);
//
//    std::cout << "H to V:" << std::endl;
//    std::vector<Eigen::Vector3f> points2 = HtoV<Eigen::Vector4f,Eigen::Vector3f,Eigen::Vector3f>(faces1, interior_pt);
//    for (size_t i = 0; i < points2.size(); ++i) {
//        std::cout << points2[i].transpose() << std::endl;
//    }

////    Visualizer vis("win", "disp");
////    vis.addPointCloud("initial", points1, Visualizer::RenderingProperties().setDrawingColor(1,0,0));
////    vis.addPointCloud("final", points2, Visualizer::RenderingProperties().setDrawingColor(0,0,1));
////
////    while (!vis.wasStopped()) {
////        vis.spinOnce();
////    }

    return 0;
}
