#include <cilantro/convex_hull.hpp>
#include <cilantro/space_region.hpp>
#include <cilantro/io.hpp>
#include <cilantro/visualizer.hpp>

#include <ctime>

void callback(cilantro::Visualizer &viz, int key) {
    if (key == 'a') {
        viz.toggleVisibilityStatus("hull1");
    } else if (key == 's') {
        viz.toggleVisibilityStatus("hull2");
    }
}

int main(int argc, char* argv[]) {
    cilantro::PointCloud3D cloud1;
    cilantro::readPointCloudFromPLYFile(argv[1], cloud1);

    // Shift to the right
    cilantro::PointCloud3D cloud2(cloud1);
    cloud2.points.colwise() += Eigen::Vector3f(1,0,0);

    // Compute convex hulls as SpaceRegion objects
    // bool flags enable topology computation (for visualization)
    cilantro::SpaceRegion3D sr1(cloud1.points, true, true);
    cilantro::SpaceRegion3D sr2(cloud2.points, true, true);

    // Compute a spatial expression
    auto start = std::chrono::high_resolution_clock::now();
//    cilantro::SpaceRegion3D sr = sr1.relativeComplement(sr2, true, true);
//    cilantro::SpaceRegion3D sr = sr1.intersectionWith(sr2, true, true);
//    cilantro::SpaceRegion3D sr = sr1.intersectionWith(sr2).complement(true, true);
    cilantro::SpaceRegion3D sr = sr1.complement().unionWith(sr2.complement()).complement(true, true);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Build time: " << elapsed.count() << "ms" << std::endl;

    std::cout << "Number of polytopes in union: " << sr.getConvexPolytopes().size() << std::endl;

    // Find the points of the concatenation of the clouds that satisfy the spatial expression
    cilantro::PointCloud3D interior_cloud(cloud1);
    interior_cloud.append(cloud2);
    interior_cloud = cilantro::PointCloud3D(interior_cloud, sr.getInteriorPointIndices(interior_cloud.points, 0.005f));

    // Visualize results
    cilantro::Visualizer viz("SpaceRegion3D example", "disp");
    viz.registerKeyboardCallback('a', std::bind(callback, std::ref(viz), 'a'));
    viz.registerKeyboardCallback('s', std::bind(callback, std::ref(viz), 's'));

    viz.addPointCloud("cloud1", cloud1, cilantro::RenderingProperties().setOpacity(0.3));
    viz.addPointCloud("cloud2", cloud2, cilantro::RenderingProperties().setOpacity(0.3));
    viz.addPointCloud("interior_cloud", interior_cloud, cilantro::RenderingProperties().setOpacity(1.0).setPointSize(2.5f).setPointColor(0.8,0.8,0.8));

    const cilantro::ConvexPolytope3D& cp1(sr1.getConvexPolytopes()[0]);
    const cilantro::ConvexPolytope3D& cp2(sr2.getConvexPolytopes()[0]);
    viz.addTriangleMesh("hull1", cp1.getVertices(), cp1.getFacetVertexIndices(), cilantro::RenderingProperties().setPointColor(1,0,0).setDrawWireframe(true).setUseFaceNormals(true).setLineWidth(2.0));
    viz.addTriangleMesh("hull2", cp2.getVertices(), cp2.getFacetVertexIndices(), cilantro::RenderingProperties().setPointColor(0,0,1).setDrawWireframe(true).setUseFaceNormals(true).setLineWidth(2.0));

//    viz.setVisibilityStatus("hull1", false);
//    viz.setVisibilityStatus("hull2", false);

    const auto& polys(sr.getConvexPolytopes());
    for (size_t i = 0; i < polys.size(); i++) {
        viz.addTriangleMesh("sr_" + std::to_string(i), polys[i].getVertices(), polys[i].getFacetVertexIndices(), cilantro::RenderingProperties().setOpacity(0.9).setUseFaceNormals(true).setLineWidth(2.0).setPointColor(0.8,0.8,0.8));
    }

    std::cout << "Press 'a' or 's' to toggle visibility of original hulls" << std::endl;
    std::cout << "Press 'w' to toggle wireframe" << std::endl;
    while (!viz.wasStopped()) {
        viz.spinOnce();
    }

    return 0;
}
