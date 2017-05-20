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
//
//    VtoH<Eigen::Vector3f,Eigen::Vector3f,Eigen::Vector4f>(points, hull_points, halfspaces, faces, hull_pt_indices, true);
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
//
//    std::cout << std::endl;
//
//    std::cout << "H to V:" << std::endl;
//    std::vector<Eigen::Vector3f> hull_points2;
//
//    HtoV<Eigen::Vector4f,Eigen::Vector3f,Eigen::Vector3f>(halfspaces, Eigen::Vector3f(0.5,0.5,0.5), hull_points2);
//
//    std::cout << "Hull points:" << std::endl;
//    for (size_t i = 0; i < hull_points2.size(); ++i) {
//        std::cout << hull_points2[i].transpose() << std::endl;
//    }

    PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    std::vector<Eigen::Vector3f> hull_points;
    std::vector<Eigen::Vector4f> halfspaces;
    std::vector<std::vector<size_t> > faces;
    std::vector<size_t> hull_pt_indices;

    VtoH<Eigen::Vector3f,Eigen::Vector3f,Eigen::Vector4f>(cloud.points, hull_points, halfspaces, faces, hull_pt_indices, true);

    Visualizer viz("win", "disp");

    PointCloud hull_cloud(cloud, hull_pt_indices);

    viz.addPointCloud("cloud", cloud, Visualizer::RenderingProperties().setOpacity(1.0));
//    viz.addTriangleMesh("mesh", hull_points, faces, Visualizer::RenderingProperties().setDrawingColor(1,0,0).setUseFaceNormals(true).setOpacity(0.8));

    std::vector<Eigen::Vector3f> face_normals(halfspaces.size());
    for (size_t i = 0; i < face_normals.size(); ++i) {
        face_normals[i] = halfspaces[i].head(3);
    }

    viz.addTriangleMesh("mesh", hull_cloud, faces, face_normals, Visualizer::RenderingProperties().setDrawingColor(1,0,0).setUseFaceNormals(true).setOpacity(0.8));

    while (!viz.wasStopped()) {
        viz.spinOnce();
    }

    return 0;
}
