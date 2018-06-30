#include <cilantro/space_region.hpp>
#include <cilantro/point_cloud.hpp>
#include <cilantro/visualizer.hpp>
#include <cilantro/common_renderables.hpp>
#include <cilantro/timer.hpp>

void callback(cilantro::Visualizer &viz, int key) {
    if (key == 'a') {
        viz.toggleVisibility("hull1");
    } else if (key == 's') {
        viz.toggleVisibility("hull2");
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Please provide path to PLY file." << std::endl;
        return 0;
    }

    cilantro::PointCloud3f cloud1(argv[1]);

    if (cloud1.isEmpty()) {
        std::cout << "Input cloud is empty!" << std::endl;
        return 0;
    }

    // Shift to the right
    cilantro::PointCloud3f cloud2(cloud1);
    cloud2.points.colwise() += Eigen::Vector3f(1,0,0);

    // Compute convex hulls as SpaceRegion objects
    // bool flags enable topology computation (for visualization)
    cilantro::SpaceRegion3f sr1(cloud1.points, true, true);
    cilantro::SpaceRegion3f sr2(cloud2.points, true, true);

    // Compute a spatial expression
    cilantro::Timer timer;
    timer.start();
//    cilantro::SpaceRegion3f sr = sr1.relativeComplement(sr2, true, true);
//    cilantro::SpaceRegion3f sr = sr1.intersectionWith(sr2, true, true);
//    cilantro::SpaceRegion3f sr = sr1.intersectionWith(sr2).complement(true, true);
    cilantro::SpaceRegion3f sr = sr1.complement().unionWith(sr2.complement()).complement(true, true);
    timer.stop();
    std::cout << "Build time: " << timer.getElapsedTime() << "ms" << std::endl;

    std::cout << "Number of polytopes in union: " << sr.getConvexPolytopes().size() << std::endl;

    // Find the points of the concatenation of the clouds that satisfy the spatial expression
    cilantro::PointCloud3f interior_cloud(cloud1);
    interior_cloud.append(cloud2);
    interior_cloud = cilantro::PointCloud3f(interior_cloud, sr.getInteriorPointIndices(interior_cloud.points, 0.005f));

    // Visualize results
    cilantro::Visualizer viz("SpaceRegion3D example", "disp");
    viz.registerKeyboardCallback('a', std::bind(callback, std::ref(viz), 'a'));
    viz.registerKeyboardCallback('s', std::bind(callback, std::ref(viz), 's'));

    viz.addObject<cilantro::PointCloudRenderable>("cloud1", cloud1, cilantro::RenderingProperties().setOpacity(0.3));
    viz.addObject<cilantro::PointCloudRenderable>("cloud2", cloud2, cilantro::RenderingProperties().setOpacity(0.3));
    viz.addObject<cilantro::PointCloudRenderable>("interior_cloud", interior_cloud, cilantro::RenderingProperties().setOpacity(1.0).setPointSize(2.5f).setPointColor(0.8,0.8,0.8));

    const cilantro::ConvexPolytope3f& cp1(sr1.getConvexPolytopes()[0]);
    const cilantro::ConvexPolytope3f& cp2(sr2.getConvexPolytopes()[0]);
    viz.addObject<cilantro::TriangleMeshRenderable>("hull1", cp1.getVertices(), cp1.getFacetVertexIndices(), cilantro::RenderingProperties().setPointColor(1,0,0).setDrawWireframe(true).setUseFaceNormals(true).setLineWidth(2.0));
    viz.addObject<cilantro::TriangleMeshRenderable>("hull2", cp2.getVertices(), cp2.getFacetVertexIndices(), cilantro::RenderingProperties().setPointColor(0,0,1).setDrawWireframe(true).setUseFaceNormals(true).setLineWidth(2.0));

//    viz.setVisibilityStatus("hull1", false);
//    viz.setVisibilityStatus("hull2", false);

    const auto& polys(sr.getConvexPolytopes());
    for (size_t i = 0; i < polys.size(); i++) {
        viz.addObject<cilantro::TriangleMeshRenderable>("sr_" + std::to_string(i), polys[i].getVertices(), polys[i].getFacetVertexIndices(), cilantro::RenderingProperties().setOpacity(0.9).setUseFaceNormals(true).setLineWidth(2.0).setPointColor(0.8,0.8,0.8));
    }

    std::cout << "Press 'a' or 's' to toggle visibility of original hulls" << std::endl;
    std::cout << "Press 'w' to toggle wireframe" << std::endl;
    while (!viz.wasStopped()) {
        viz.spinOnce();
    }

    return 0;
}
