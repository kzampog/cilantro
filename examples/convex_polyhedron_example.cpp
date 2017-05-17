#include <cilantro/convex_polyhedron.hpp>
#include <cilantro/ply_io.hpp>
#include <cilantro/visualizer.hpp>

#include <iostream>

int main(int argc, char ** argv) {

    PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    std::vector<Eigen::Vector4f> faces1 = VtoH<Eigen::Vector3f,Eigen::Vector4f>(cloud.points);
    std::vector<Eigen::Vector3f> points2 = HtoV<Eigen::Vector4f,Eigen::Vector3f,Eigen::Vector3f>(faces1, Eigen::Vector3f(0,0,0.8));

    Visualizer vis("win", "disp");
    vis.addPointCloud("initial", cloud, Visualizer::RenderingProperties().setDrawingColor(1,0,0).setOverrideColors(true).setOpacity(0.5).setPointSize(1.0));
    vis.addPointCloud("final", points2, Visualizer::RenderingProperties().setDrawingColor(0,0,1));

    while (!vis.wasStopped()) {
        vis.spinOnce();
    }

//    std::vector<Eigen::Vector3f> points1;
//    points1.push_back(Eigen::Vector3f(0,0,0));
//    points1.push_back(Eigen::Vector3f(1,0,0));
//    points1.push_back(Eigen::Vector3f(0,1,0));
//    points1.push_back(Eigen::Vector3f(0,0,1));
//    points1.push_back(Eigen::Vector3f(0,1,1));
//    points1.push_back(Eigen::Vector3f(1,0,1));
//    points1.push_back(Eigen::Vector3f(1,1,0));
//    points1.push_back(Eigen::Vector3f(1,1,1));
//
//    std::cout << "V to H:" << std::endl;
//    std::vector<Eigen::Vector4f> faces1 = VtoH<Eigen::Vector3f,Eigen::Vector4f>(points1);
//    for (size_t i = 0; i < faces1.size(); ++i) {
//        std::cout << faces1[i].transpose() << std::endl;
//    }
//
////    std::vector<Eigen::Vector4f> faces2;
////    faces2.push_back(Eigen::Vector4f(-1,0,0,0));
////    faces2.push_back(Eigen::Vector4f(0,-1,0,0));
////    faces2.push_back(Eigen::Vector4f(0,0,-1,0));
////    faces2.push_back(Eigen::Vector4f(1,1,1,-1));
//
////    Eigen::Vector3f interior_pt(0.1, 0.1, 0.1);
//
//    Eigen::Vector3f interior_pt(0.5, 0.5, 0.5);
//
//    std::cout << "H to V:" << std::endl;
//    std::vector<Eigen::Vector3f> points2 = HtoV<Eigen::Vector4f,Eigen::Vector3f,Eigen::Vector3f>(faces1, interior_pt);
//    for (size_t i = 0; i < points2.size(); ++i) {
//        std::cout << points2[i].transpose() << std::endl;
//    }
//
////    Visualizer vis("win", "disp");
////    vis.addPointCloud("initial", points1, Visualizer::RenderingProperties().setDrawingColor(1,0,0));
////    vis.addPointCloud("final", points2, Visualizer::RenderingProperties().setDrawingColor(0,0,1));
////
////    while (!vis.wasStopped()) {
////        vis.spinOnce();
////    }

    return 0;
}
