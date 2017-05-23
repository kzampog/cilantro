#include <cilantro/convex_hull.hpp>
#include <cilantro/ply_io.hpp>
#include <cilantro/visualizer.hpp>

#include <iostream>

int main(int argc, char ** argv) {

//    std::vector<Eigen::Vector3f> points;
//    points.push_back(Eigen::Vector3f(0,0,0));
//    points.push_back(Eigen::Vector3f(1,0,0));
//    points.push_back(Eigen::Vector3f(0,1,0));
//    points.push_back(Eigen::Vector3f(0,0,1));
//    points.push_back(Eigen::Vector3f(0,1,1));
//    points.push_back(Eigen::Vector3f(1,0,1));
//    points.push_back(Eigen::Vector3f(1,1,0));
//    points.push_back(Eigen::Vector3f(1,1,1));
//    points.push_back(Eigen::Vector3f(0.5,0.5,0.5));
//
//    std::cout << "V to H:" << std::endl;
//    std::vector<Eigen::Vector3f> hull_points;
//    std::vector<Eigen::Vector4f> halfspaces;
//    std::vector<std::vector<size_t> > faces;
//    std::vector<size_t> hull_pt_indices;
//    double area, volume;
//
//    convexHullFromPoints<Eigen::Vector3f,Eigen::Vector3f,Eigen::Vector4f>(points, hull_points, halfspaces, faces, hull_pt_indices, area, volume);
//
//    std::cout << "Input points:" << std::endl;
//    for (size_t i = 0; i < points.size(); ++i) {
//        std::cout << points[i].transpose() << std::endl;
//    }
//
//    std::cout << "Hull points:" << std::endl;
//    for (size_t i = 0; i < hull_points.size(); ++i) {
//        std::cout << hull_points[i].transpose() << std::endl;
//    }
//
//    std::cout << "Halfspaces:" << std::endl;
//    for (size_t i = 0; i < halfspaces.size(); ++i) {
//        std::cout << halfspaces[i].transpose() << std::endl;
//    }
//
//    std::cout << "Faces:" << std::endl;
//    for (size_t i = 0; i < faces.size(); ++i) {
//        for (size_t j = 0; j < faces[i].size(); ++j) {
//            std::cout << faces[i][j] << " ";
//        }
//        std::cout << std::endl;
//    }
//
//    std::cout << "Hull point indices:" << std::endl;
//    for (size_t i = 0; i < hull_pt_indices.size(); ++i) {
//        std::cout << hull_pt_indices[i] << " " << std::endl;
//    }
//    std::cout << "Area: " << area << std::endl;
//    std::cout << "Volume: " << volume << std::endl;
//
//    std::cout << std::endl;
//
//    std::cout << "H to V:" << std::endl;
//    std::vector<Eigen::Vector3f> hull_points2;
//    std::vector<Eigen::Vector4f> halfspaces2;
//    std::vector<std::vector<size_t> > faces2;
//    std::vector<size_t> hull_pt_indices2;
//    double area2, volume2;
//
//    convexHullFromHalfspaces<Eigen::Vector4f,Eigen::Vector3f,Eigen::Vector4f>(halfspaces, hull_points2,halfspaces2,faces2,hull_pt_indices2, area2, volume2);
//
//    std::cout << "Input halfspaces:" << std::endl;
//    for (size_t i = 0; i < halfspaces.size(); ++i) {
//        std::cout << halfspaces[i].transpose() << std::endl;
//    }
//
//    std::cout << "Hull points:" << std::endl;
//    for (size_t i = 0; i < hull_points2.size(); ++i) {
//        std::cout << hull_points2[i].transpose() << std::endl;
//    }
//
//    std::cout << "Halfspaces:" << std::endl;
//    for (size_t i = 0; i < halfspaces2.size(); ++i) {
//        std::cout << halfspaces2[i].transpose() << std::endl;
//    }
//
//    std::cout << "Faces:" << std::endl;
//    for (size_t i = 0; i < faces2.size(); ++i) {
//        for (size_t j = 0; j < faces2[i].size(); ++j) {
//            std::cout << faces2[i][j] << " ";
//        }
//        std::cout << std::endl;
//    }
//
//    std::cout << "Hull point indices:" << std::endl;
//    for (size_t i = 0; i < hull_pt_indices2.size(); ++i) {
//        std::cout << hull_pt_indices2[i] << " " << std::endl;
//    }
//
//    std::cout << "Area: " << area2 << std::endl;
//    std::cout << "Volume: " << volume2 << std::endl;

    PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

//    std::vector<Eigen::Vector3f> hull_points;
//    std::vector<Eigen::Vector4f> halfspaces;
//    std::vector<std::vector<size_t> > faces;
//    std::vector<size_t> hull_pt_indices;
//    double area, volume;
//
//    convexHullFromPoints<float,float,3>(cloud.points, hull_points, halfspaces, faces, hull_pt_indices, area, volume);
////    convexHullFromHalfspaces<float,float,3>(halfspaces, hull_points, halfspaces, faces, hull_pt_indices, area, volume, true, 0.000001);
//
//    std::cout << hull_points.size() << std::endl;

    ConvexHull3D ch(cloud);

    Visualizer viz("win", "disp");

    viz.addPointCloud("cloud", cloud, Visualizer::RenderingProperties().setOpacity(1.0));
    viz.addTriangleMesh("mesh", ch.getVertices(), ch.getFacetVertexIndices(), Visualizer::RenderingProperties().setDrawingColor(1,0,0).setUseFaceNormals(true).setOpacity(0.8));

    while (!viz.wasStopped()) {
        viz.spinOnce();
    }

    return 0;
}
