#include <cilantro/convex_polytope.hpp>
#include <cilantro/point_cloud.hpp>
#include <cilantro/visualizer.hpp>
#include <cilantro/common_renderables.hpp>

void run_demo() {
    std::vector<Eigen::Vector3f> points;
    points.emplace_back(0,0,0);
    points.emplace_back(1,0,0);
    points.emplace_back(0,1,0);
    points.emplace_back(0,0,1);
    points.emplace_back(0,1,1);
    points.emplace_back(1,0,1);
    points.emplace_back(1,1,0);
    points.emplace_back(1,1,1);
    points.emplace_back(0.5,0.5,0.5);

    cilantro::ConvexHull3f ch(points, true);

    std::cout << "Vertices:" << std::endl;
    for (size_t i = 0; i < ch.getVertices().cols(); i++) {
        std::cout << ch.getVertices().col(i).transpose() << std::endl;
    }

    std::cout << "Faces:" << std::endl;
    for (size_t i = 0; i < ch.getFacetVertexIndices().size(); i++) {
        for (size_t j = 0; j < ch.getFacetVertexIndices()[i].size(); j++) {
            std::cout << ch.getFacetVertexIndices()[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Vertex neighbor faces:" << std::endl;
    for (size_t i = 0; i < ch.getVertexNeighborFacets().size(); i++) {
        for (size_t j = 0; j < ch.getVertexNeighborFacets()[i].size(); j++) {
            std::cout << ch.getVertexNeighborFacets()[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Face neighbor faces:" << std::endl;
    for (size_t i = 0; i < ch.getFacetNeighborFacets().size(); i++) {
        for (size_t j = 0; j < ch.getFacetNeighborFacets()[i].size(); j++) {
            std::cout << ch.getFacetNeighborFacets()[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cout << "No input PLY file path provided, running simple demo." << std::endl;
        run_demo();
        return 0;
    }

    cilantro::PointCloud3f cloud(argv[1]);

    if (cloud.isEmpty()) {
        std::cout << "Input cloud is empty!" << std::endl;
        return 0;
    }

    cilantro::ConvexHull3f ch(cloud.points, true, true);
    cilantro::PointCloud3f hull_cloud(cloud, ch.getVertexPointIndices());

    cilantro::VectorSet3f vertex_colors(3, ch.getVertices().cols());
    std::vector<float> vertex_values(ch.getVertices().cols());
    for (size_t i = 0; i < ch.getVertices().cols(); i++) {
        if (i%3 == 0) vertex_colors.col(i) = Eigen::Vector3f(1,0,0);
        if (i%3 == 1) vertex_colors.col(i) = Eigen::Vector3f(0,1,0);
        if (i%3 == 2) vertex_colors.col(i) = Eigen::Vector3f(0,0,1);
        vertex_values[i] = ch.getVertices().col(i).norm();
    }

    cilantro::VectorSet3f face_colors(3, ch.getFacetVertexIndices().size());
    std::vector<float> face_values(ch.getFacetVertexIndices().size());
    for (size_t i = 0; i < ch.getFacetVertexIndices().size(); i++) {
        if (i%3 == 0) face_colors.col(i) = Eigen::Vector3f(1,0,0);
        if (i%3 == 1) face_colors.col(i) = Eigen::Vector3f(0,1,0);
        if (i%3 == 2) face_colors.col(i) = Eigen::Vector3f(0,0,1);
        face_values[i] = (ch.getVertices().col(ch.getFacetVertexIndices()[i][0]) +
                     ch.getVertices().col(ch.getFacetVertexIndices()[i][1]) +
                     ch.getVertices().col(ch.getFacetVertexIndices()[i][2])).rowwise().mean().norm();
    }

    cilantro::Visualizer viz("3D convex hull", "disp");
    viz.addObject<cilantro::PointCloudRenderable>("cloud", cloud, cilantro::RenderingProperties().setOpacity(1.0));
    viz.addObject<cilantro::TriangleMeshRenderable>("mesh", ch.getVertices(), ch.getFacetVertexIndices())
            ->setVertexNormals(hull_cloud.normals).setVertexColors(vertex_colors).setVertexValues(vertex_values)
             .setFaceColors(face_colors).setFaceValues(face_values);

    cilantro::RenderingProperties rp = viz.getRenderingProperties("mesh");
    rp.setUseFaceNormals(true).setUseFaceColors(false).setOpacity(0.8).setColormapType(cilantro::ColormapType::BLUE2RED);
    viz.setRenderingProperties("mesh", rp);

    viz.spin();

    return 0;
}
