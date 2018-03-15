#include <cilantro/flat_convex_hull_3d.hpp>
#include <cilantro/io.hpp>
#include <cilantro/visualizer.hpp>

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
//    ConvexHull3D ch(points);
//
//    std::cout << "Vertices:" << std::endl;
//    for (size_t i = 0; i < ch.getVertices().size(); i++) {
//        std::cout << ch.getVertices()[i].transpose() << std::endl;
//    }
//
//    std::cout << "Faces:" << std::endl;
//    for (size_t i = 0; i < ch.getFacetVertexIndices().size(); i++) {
//        for (size_t j = 0; j < ch.getFacetVertexIndices()[i].size(); j++) {
//            std::cout << ch.getFacetVertexIndices()[i][j] << " ";
//        }
//        std::cout << std::endl;
//    }
//
//    std::cout << "Vertex neighbor faces:" << std::endl;
//    for (size_t i = 0; i < ch.getVertexNeighborFacets().size(); i++) {
//        for (size_t j = 0; j < ch.getVertexNeighborFacets()[i].size(); j++) {
//            std::cout << ch.getVertexNeighborFacets()[i][j] << " ";
//        }
//        std::cout << std::endl;
//    }
//
//    std::cout << "Face neighbor faces:" << std::endl;
//    for (size_t i = 0; i < ch.getFacetNeighborFacets().size(); i++) {
//        for (size_t j = 0; j < ch.getFacetNeighborFacets()[i].size(); j++) {
//            std::cout << ch.getFacetNeighborFacets()[i][j] << " ";
//        }
//        std::cout << std::endl;
//    }


    cilantro::PointCloud3f cloud;
    cilantro::readPointCloudFromPLYFile(argv[1], cloud);

    cilantro::ConvexHull3f ch(cloud.points, true, true);
    cilantro::PointCloud3f hc(cloud, ch.getVertexPointIndices());

    std::vector<Eigen::Vector3f> v_colors(ch.getVertices().cols());
    std::vector<float> v_vals(ch.getVertices().cols());
    for (size_t i = 0; i < ch.getVertices().cols(); i++) {
        if (i%3 == 0) v_colors[i] = Eigen::Vector3f(1,0,0);
        if (i%3 == 1) v_colors[i] = Eigen::Vector3f(0,1,0);
        if (i%3 == 2) v_colors[i] = Eigen::Vector3f(0,0,1);
        v_vals[i] = ch.getVertices().col(i).norm();
    }

    std::vector<Eigen::Vector3f> f_colors(ch.getFacetVertexIndices().size());
    std::vector<float> f_vals(ch.getFacetVertexIndices().size());
    for (size_t i = 0; i < ch.getFacetVertexIndices().size(); i++) {
        if (i%3 == 0) f_colors[i] = Eigen::Vector3f(1,0,0);
        if (i%3 == 1) f_colors[i] = Eigen::Vector3f(0,1,0);
        if (i%3 == 2) f_colors[i] = Eigen::Vector3f(0,0,1);
        f_vals[i] = (ch.getVertices().col(ch.getFacetVertexIndices()[i][0]) +
                     ch.getVertices().col(ch.getFacetVertexIndices()[i][1]) +
                     ch.getVertices().col(ch.getFacetVertexIndices()[i][2])).rowwise().mean().norm();
    }

    cilantro::Visualizer viz1("3D convex hull", "disp");

    viz1.addPointCloud("cloud", cloud, cilantro::RenderingProperties().setOpacity(1.0));

    viz1.addTriangleMesh("mesh", ch.getVertices(), ch.getFacetVertexIndices());
    viz1.addTriangleMeshVertexNormals("mesh", hc.normals);
    viz1.addTriangleMeshVertexColors("mesh", v_colors);
    viz1.addTriangleMeshFaceColors("mesh", f_colors);
    viz1.addTriangleMeshVertexValues("mesh", v_vals);
    viz1.addTriangleMeshFaceValues("mesh", f_vals);

    cilantro::RenderingProperties rp = viz1.getRenderingProperties("mesh");
    rp.setUseFaceNormals(true).setUseFaceColors(false).setOpacity(0.8).setColormapType(cilantro::ColormapType::BLUE2RED);
    viz1.setRenderingProperties("mesh", rp);

    if (cloud.points.size() < 3) return 0;

    // PointCloudHullFlat also inherits from PrincipalComponentAnalysis
    cilantro::FlatConvexHull3f ch2d(cloud.points, true, true);
    cloud.points = ch2d.reconstruct<2>(ch2d.project<2>(cloud.points));
    cloud.normals.resize(Eigen::NoChange, 0);

    std::vector<std::vector<size_t> > face_v_ind = ch2d.getFacetVertexIndices();
    std::vector<Eigen::Vector3f> p_src(face_v_ind.size()), p_dst(face_v_ind.size());
    for (size_t i = 0; i < face_v_ind.size(); i++) {
        p_src[i] = ch2d.getVertices3D().col(face_v_ind[i][0]);
        p_dst[i] = ch2d.getVertices3D().col(face_v_ind[i][1]);
    }

    cilantro::Visualizer viz2("2D convex hull in 3D space", "disp");
    viz2.addPointCloud("cloud", cloud, cilantro::RenderingProperties().setOpacity(0.5));
    viz2.addPointCloud("hull_cloud", ch2d.getVertices3D(), cilantro::RenderingProperties().setPointColor(1,0,0).setPointSize(10.0));
    viz2.addPointCorrespondences("hull_lines", p_src, p_dst, cilantro::RenderingProperties().setLineColor(0,0,1).setLineWidth(5.0).setLineDensityFraction(1.0));

    while (!viz1.wasStopped() && !viz2.wasStopped()) {
        viz1.spinOnce();
        viz2.spinOnce();
    }

    return 0;
}
