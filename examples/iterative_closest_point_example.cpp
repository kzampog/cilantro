#include <cilantro/icp_rigid_point_to_point.hpp>
#include <cilantro/icp_rigid_combined_metric_3d.hpp>
#include <cilantro/icp_simple_feature_adaptors.hpp>
#include <cilantro/io.hpp>
#include <cilantro/voxel_grid.hpp>
#include <cilantro/visualizer.hpp>

void callback(bool &proceed) {
    proceed = true;
}

int main(int argc, char ** argv) {
    cilantro::PointCloud3f dst, src;
    cilantro::readPointCloudFromPLYFile(argv[1], dst);

    dst = cilantro::VoxelGrid(dst, 0.005).getDownsampledCloud();

    // Create a distorted and transformed version of the point cloud
    src = dst;
    for (size_t i = 0; i < src.size(); i++) {
        src.points.col(i) += 0.01f*Eigen::Vector3f::Random();
        src.normals.col(i) += 0.02f*Eigen::Vector3f::Random();
        src.normals.col(i).normalize();
        src.colors.col(i) += 0.05f*Eigen::Vector3f::Random();
    }

    cilantro::PointCloud3f dst2;
    dst2.points.resize(Eigen::NoChange,dst.size());
    dst2.normals.resize(Eigen::NoChange,dst.size());
    dst2.colors.resize(Eigen::NoChange,dst.size());
    size_t k = 0;
    for (size_t i = 0; i < dst.size(); i++) {
        if (dst.points(0,i) > -0.4) {
            dst2.points.col(k) = dst.points.col(i);
            dst2.normals.col(k) = dst.normals.col(i);
            dst2.colors.col(k) = dst.colors.col(i);
            k++;
        }
    }
    dst2.points.conservativeResize(Eigen::NoChange,k);
    dst2.normals.conservativeResize(Eigen::NoChange,k);
    dst2.colors.conservativeResize(Eigen::NoChange,k);

    dst = dst2;

    cilantro::RigidTransformation3f tf_ref;
    Eigen::Matrix3f tmp;
    tmp = Eigen::AngleAxisf(-0.1f ,Eigen::Vector3f::UnitZ()) *
          Eigen::AngleAxisf(0.1f, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(-0.1f, Eigen::Vector3f::UnitX());
    tf_ref.linear() = tmp;
    tf_ref.translation() = Eigen::Vector3f(-0.20f, -0.05f, 0.10f);

    src.points = tf_ref*src.points;
    src.normals = tf_ref.linear()*src.normals;

    // Visualize initial configuration
    cilantro::Visualizer viz("IterativeClosestPoint example", "disp");
    bool proceed = false;
    viz.registerKeyboardCallback('a', std::bind(callback, std::ref(proceed)));

    viz.addPointCloud("dst", dst, cilantro::RenderingProperties().setPointColor(0,0,1));
    viz.addPointCloud("src", src, cilantro::RenderingProperties().setPointColor(1,0,0));

    std::cout << "Press 'a' to compute transformation" << std::endl;
    while (!proceed && !viz.wasStopped()) {
        if (viz.wasStopped()) return 0;
        viz.spinOnce();
    }
    proceed = false;

    // Perform ICP registration
    auto start = std::chrono::high_resolution_clock::now();

    cilantro::RigidTransformation3f tf_est;

    // Point features adaptors for correspondence search
//    cilantro::PointsNormalsColorsAdaptor<float,3> dst_feat(dst.points, dst.normals, dst.colors, 0.5, 5.0);
//    cilantro::PointsNormalsColorsAdaptor<float,3> src_feat(src.points, src.normals, src.colors, 0.5, 5.0);
    cilantro::PointsAdaptor<float,3> dst_feat(dst.points);
    cilantro::PointsAdaptor<float,3> src_feat(src.points);

    // Point-to-point
//    cilantro::PointToPointRigidICP<float,3,cilantro::PointsNormalsColorsAdaptor<float,3>> icp(dst.points, src.points, dst_feat, src_feat);
//    cilantro::PointToPointRigidICP<float,3,cilantro::PointsAdaptor<float,3>> icp(dst.points, src.points, dst_feat, src_feat);

    // Weighted combination of point-to-point and point-to-plane
//    cilantro::CombinedMetricRigidICP3D<float,cilantro::PointsNormalsColorsAdaptor<float,3>> icp(dst.points, dst.normals, src.points, dst_feat, src_feat);
    cilantro::CombinedMetricRigidICP3D<float,cilantro::PointsAdaptor<float,3>> icp(dst.points, dst.normals, src.points, dst_feat, src_feat);

    icp.setMaxCorrespondenceDistance(0.1f*0.1f).setConvergenceTolerance(1e-3f).setMaxNumberOfIterations(100);
    icp.setMaxNumberOfOptimizationStepIterations(1).setPointToPointMetricWeight(0.0).setPointToPlaneMetricWeight(1.0);

    tf_est = icp.estimateTransformation().getTransformation();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Registration time: " << elapsed.count() << "ms" << std::endl;

    std::cout << "Iterations performed: " << icp.getNumberOfPerformedIterations() << std::endl;
    std::cout << "Has converged: " << icp.hasConverged() << std::endl;

    std::cout << "TRUE transformation:" << std::endl << tf_ref.inverse().matrix() << std::endl;
    std::cout << "ESTIMATED transformation R:" << std::endl << tf_est.matrix() << std::endl;

    // Visualize registration results
    cilantro::PointCloud3f src_trans(src);

    src_trans.points = tf_est*src_trans.points;
    src_trans.normals = tf_est.linear()*src_trans.normals;

    viz.addPointCloud("dst", dst, cilantro::RenderingProperties().setPointColor(0,0,1));
    viz.addPointCloud("src", src_trans, cilantro::RenderingProperties().setPointColor(1,0,0));

    std::cout << "Press 'a' to compute residuals" << std::endl;
    while (!proceed && !viz.wasStopped()) {
        if (viz.wasStopped()) return 0;
        viz.spinOnce();
    }
    proceed = false;

    start = std::chrono::high_resolution_clock::now();
    auto residuals = icp.getResiduals();
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Residual computation time: " << elapsed.count() << "ms" << std::endl;

    viz.clear();
    viz.addPointCloud("src", src_trans, cilantro::RenderingProperties().setUseLighting(false)).addPointCloudValues("src", residuals);

    while (!proceed && !viz.wasStopped()) {
        if (viz.wasStopped()) return 0;
        viz.spinOnce();
    }

    return 0;
}
