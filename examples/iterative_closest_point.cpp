#include <cilantro/icp_common_instances.hpp>
#include <cilantro/point_cloud.hpp>
#include <cilantro/visualizer.hpp>
#include <cilantro/common_renderables.hpp>
#include <cilantro/timer.hpp>

void callback(bool &proceed) {
    proceed = true;
}

void generate_input_data(cilantro::PointCloud3f &dst,
                         cilantro::PointCloud3f &src,
                         cilantro::RigidTransformation3f &tf_ref)
{
    dst.gridDownsample(0.005f);

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

    Eigen::Matrix3f tmp;
    tmp = Eigen::AngleAxisf(-0.1f ,Eigen::Vector3f::UnitZ()) *
          Eigen::AngleAxisf(0.1f, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(-0.1f, Eigen::Vector3f::UnitX());
    tf_ref.linear() = tmp;
    tf_ref.translation() = Eigen::Vector3f(-0.20f, -0.05f, 0.10f);

    src.transform(tf_ref);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cout << "Please provide path to PLY file." << std::endl;
        return 0;
    }

    cilantro::PointCloud3f dst(argv[1]), src;
    cilantro::RigidTransformation3f tf_ref;

    if (!dst.hasNormals()) {
        std::cout << "Input cloud does not have normals!" << std::endl;
        return 0;
    }

    generate_input_data(dst, src, tf_ref);

    // Visualize initial configuration
    cilantro::Visualizer viz("IterativeClosestPoint example", "disp");
    bool proceed = false;
    viz.registerKeyboardCallback('a', std::bind(callback, std::ref(proceed)));

    viz.addObject<cilantro::PointCloudRenderable>("dst", dst, cilantro::RenderingProperties().setPointColor(0,0,1));
    viz.addObject<cilantro::PointCloudRenderable>("src", src, cilantro::RenderingProperties().setPointColor(1,0,0));

    std::cout << "Press 'a' to compute transformation" << std::endl;
    while (!proceed && !viz.wasStopped()) {
        if (viz.wasStopped()) return 0;
        viz.spinOnce();
    }
    proceed = false;

    // Perform ICP registration
    cilantro::Timer timer;
    timer.start();

//    // Custom examples
//    // Point features adaptors for correspondence search
////    cilantro::PointNormalColorFeaturesAdaptor3f dst_feat(dst.points, dst.normals, dst.colors, 0.5, 5.0);
////    cilantro::PointNormalColorFeaturesAdaptor3f src_feat(src.points, src.normals, src.colors, 0.5, 5.0);
//
//    cilantro::PointFeaturesAdaptor3f dst_feat(dst.points);
//    cilantro::PointFeaturesAdaptor3f src_feat(src.points);
//
//    cilantro::CorrespondenceDistanceEvaluator<float> eval;
//    cilantro::ICPCorrespondenceSearchKDTree<decltype(dst_feat)> corr_engine(dst_feat, src_feat, eval);
//
//    // Metric optimizers
////    cilantro::RigidICPPointToPointMetricOptimizer<float,3> metric_opt(dst.points, src.points);
//    cilantro::RigidICPCombinedMetricOptimizer3<float> metric_opt(dst.points, dst.normals, src.points);
//
//    // ICP object
//    cilantro::IterativeClosestPoint<decltype(metric_opt),decltype(corr_engine)> icp(metric_opt, corr_engine);


    // Common case examples
    // Point-to-point
//    cilantro::PointToPointMetricRigidProjectiveICP3f icp(dst.points, src.points);
//    cilantro::PointToPointMetricRigidICP3f icp(dst.points, src.points);

    // Weighted combination of point-to-point and point-to-plane
//    cilantro::CombinedMetricRigidProjectiveICP3f icp(dst.points, dst.normals, src.points);
    cilantro::CombinedMetricRigidICP3f icp(dst.points, dst.normals, src.points);

    // Set parameters
    icp.setConvergenceTolerance(1e-4f).setMaxNumberOfIterations(30);

    icp.metricOptimizer().setMaxNumberOfOptimizationIterations(1).setPointToPointMetricWeight(0.0).setPointToPlaneMetricWeight(1.0);
    icp.correspondenceSearchEngine().setMaxDistance(0.1f*0.1f);

    cilantro::RigidTransformation3f tf_est = icp.estimateTransformation().getTransformation();

    timer.stop();

    std::cout << "Registration time: " << timer.getElapsedTime() << "ms" << std::endl;
    std::cout << "Iterations performed: " << icp.getNumberOfPerformedIterations() << std::endl;
    std::cout << "Has converged: " << icp.hasConverged() << std::endl;
    std::cout << "TRUE transformation:" << std::endl << tf_ref.inverse().matrix() << std::endl;
    std::cout << "ESTIMATED transformation R:" << std::endl << tf_est.matrix() << std::endl;

    // Visualize registration results
    cilantro::PointCloud3f src_trans(src);

    src_trans.transform(tf_est);

    viz.addObject<cilantro::PointCloudRenderable>("dst", dst, cilantro::RenderingProperties().setPointColor(0,0,1));
    viz.addObject<cilantro::PointCloudRenderable>("src", src_trans, cilantro::RenderingProperties().setPointColor(1,0,0));

    std::cout << "Press 'a' to compute residuals" << std::endl;
    while (!proceed && !viz.wasStopped()) {
        if (viz.wasStopped()) return 0;
        viz.spinOnce();
    }
    proceed = false;

    timer.start();
    auto residuals = icp.getResiduals();
    timer.stop();
    std::cout << "Residual computation time: " << timer.getElapsedTime() << "ms" << std::endl;

    viz.clear();
    viz.addObject<cilantro::PointCloudRenderable>("src", src_trans, cilantro::RenderingProperties().setUseLighting(false))
            ->setPointValues(residuals);

    while (!proceed && !viz.wasStopped()) {
        if (viz.wasStopped()) return 0;
        viz.spinOnce();
    }

    return 0;
}
