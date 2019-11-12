#include <cilantro/spatial/flat_convex_hull_3d.hpp>
#include <cilantro/utilities/point_cloud.hpp>
#include <cilantro/visualization.hpp>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cout << "Please provide path to PLY file." << std::endl;
        return 0;
    }

    cilantro::PointCloud3f cloud(argv[1]);
    cloud.normals.resize(3, 0);

    if (cloud.size() < 3) {
        std::cout << "Input cloud empty or too small!" << std::endl;
        return 0;
    }

    // PointCloudHullFlat also inherits from PrincipalComponentAnalysis
    cilantro::FlatConvexHull3f flat_hull(cloud.points, true, true);
    cloud.points = flat_hull.reconstruct<2>(flat_hull.project<2>(cloud.points));

    const auto& face_v_ind = flat_hull.getFacetVertexIndices();
    cilantro::VectorSet3f src_points(3, face_v_ind.size());
    cilantro::VectorSet3f dst_points(3, face_v_ind.size());
    for (size_t i = 0; i < face_v_ind.size(); i++) {
        src_points.col(i) = flat_hull.getVertices3D().col(face_v_ind[i][0]);
        dst_points.col(i) = flat_hull.getVertices3D().col(face_v_ind[i][1]);
    }

    cilantro::Visualizer viz("2D convex hull in 3D space", "disp");
    viz.addObject<cilantro::PointCloudRenderable>("cloud", cloud, cilantro::RenderingProperties().setOpacity(0.5));
    viz.addObject<cilantro::PointCloudRenderable>("hull_cloud", flat_hull.getVertices3D(), cilantro::RenderingProperties().setPointColor(1,0,0).setPointSize(10.0));
    viz.addObject<cilantro::PointCorrespondencesRenderable>("hull_lines", src_points, dst_points, cilantro::RenderingProperties().setLineColor(0,0,1).setLineWidth(5.0).setLineDensityFraction(1.0));
    viz.spin();

    return 0;
}
