#include <cilantro/registration/icp_common_instances.hpp>
#include <cilantro/utilities/point_cloud.hpp>
#include <cilantro/visualization.hpp>
#include <cilantro/utilities/timer.hpp>

void color_toggle(cilantro::Visualizer &viz) {
    cilantro::RenderingProperties rp = viz.getRenderingProperties("dst");
    if (rp.pointColor == cilantro::RenderingProperties::noColor) {
        rp.setPointColor(0.0f, 0.0f, 1.0f);
    } else {
        rp.setPointColor(cilantro::RenderingProperties::noColor);
    }
    viz.setRenderingProperties("dst", rp);
    rp = viz.getRenderingProperties("src");
    if (rp.pointColor == cilantro::RenderingProperties::noColor) {
        rp.setPointColor(1.0f, 0.0f, 0.0f);
    } else {
        rp.setPointColor(cilantro::RenderingProperties::noColor);
    }
    viz.setRenderingProperties("src", rp);
}

void generate_input_data(cilantro::PointCloud3f &dst,
                         cilantro::PointCloud3f &src,
                         cilantro::RigidTransform3f &tf_ref)
{
    dst.gridDownsample(0.005f);

    // Create a distorted and transformed version of the dst point cloud
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
    cilantro::RigidTransform3f tf_ref;

    if (!dst.hasNormals()) {
        std::cout << "Input cloud is empty or does not have normals!" << std::endl;
        return 0;
    }

    generate_input_data(dst, src, tf_ref);

    // Perform ICP registration
    cilantro::Timer timer;
    timer.start();

//    // Custom examples
//    // Point features adaptors for correspondence search
////    cilantro::PointNormalColorFeaturesAdaptor3f dst_feat(dst.points, dst.normals, dst.colors, 0.5, 5.0);
////    cilantro::PointNormalColorFeaturesAdaptor3f src_feat(src.points, src.normals, src.colors, 0.5, 5.0);
//    cilantro::PointFeaturesAdaptor3f dst_feat(dst.points);
//    cilantro::PointFeaturesAdaptor3f src_feat(src.points);
//
//    cilantro::DistanceEvaluator<float> dist_eval;
//    cilantro::CorrespondenceSearchKDTree<decltype(dst_feat)> corr_engine(dst_feat, src_feat, dist_eval);
//
//    cilantro::UnityWeightEvaluator<float> corr_weight_eval;
//
//    // Point-to-point
////    cilantro::PointToPointMetricRigidTransformICP3f<decltype(corr_engine)> icp(dst.points, src.points, corr_engine);
//    // Weighted combination of point-to-point and point-to-plane
//    cilantro::CombinedMetricRigidTransformICP3f<decltype(corr_engine)> icp(dst.points, dst.normals, src.points, corr_engine, corr_weight_eval, corr_weight_eval);

    // Common instances
    // Point-to-point
//    cilantro::SimplePointToPointMetricRigidProjectiveICP3f icp(dst.points, src.points);
//    cilantro::SimplePointToPointMetricRigidICP3f icp(dst.points, src.points);

    // Weighted combination of point-to-point and point-to-plane
//    cilantro::SimpleCombinedMetricRigidProjectiveICP3f icp(dst.points, dst.normals, src.points);
    cilantro::SimpleCombinedMetricRigidICP3f icp(dst.points, dst.normals, src.points);

    // Parameter setting
    icp.setMaxNumberOfOptimizationStepIterations(1).setPointToPointMetricWeight(0.0f).setPointToPlaneMetricWeight(1.0f);
    icp.correspondenceSearchEngine().setMaxDistance(0.1f*0.1f);
    icp.setConvergenceTolerance(1e-4f).setMaxNumberOfIterations(30);

    cilantro::RigidTransform3f tf_est = icp.estimate().getTransform();

    timer.stop();

    std::cout << "Registration time: " << timer.getElapsedTime() << "ms" << std::endl;
    std::cout << "Iterations performed: " << icp.getNumberOfPerformedIterations() << std::endl;
    std::cout << "Has converged: " << icp.hasConverged() << std::endl;
    std::cout << "TRUE transformation:" << std::endl << tf_ref.inverse().matrix() << std::endl;
    std::cout << "ESTIMATED transformation:" << std::endl << tf_est.matrix() << std::endl;

    timer.start();
    auto residuals = icp.getResiduals();
    timer.stop();
    std::cout << "Residual computation time: " << timer.getElapsedTime() << "ms" << std::endl;

    // Visualization
    const std::string window_name = "Rigid ICP example";
    pangolin::CreateWindowAndBind(window_name, 1920, 480);
    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual)
            .AddDisplay(pangolin::Display("initial"))
            .AddDisplay(pangolin::Display("registration"))
            .AddDisplay(pangolin::Display("residuals"));

    cilantro::Visualizer initial_viz(window_name, "initial");
    cilantro::Visualizer registration_viz(window_name, "registration");
    cilantro::Visualizer residuals_viz(window_name, "residuals");

    // Keep viewpoints in sync
    registration_viz.setRenderState(initial_viz.getRenderState());
    residuals_viz.setRenderState(initial_viz.getRenderState());

    // Initial state
    initial_viz.registerKeyboardCallback('c', std::bind(color_toggle, std::ref(initial_viz)));
    initial_viz.addObject<cilantro::PointCloudRenderable>("dst", dst, cilantro::RenderingProperties().setPointColor(0,0,1));
    initial_viz.addObject<cilantro::PointCloudRenderable>("src", src, cilantro::RenderingProperties().setPointColor(1,0,0));

    // Registration result
    auto src_t = src.transformed(tf_est);
    registration_viz.registerKeyboardCallback('c', std::bind(color_toggle, std::ref(registration_viz)));
    registration_viz.addObject<cilantro::PointCloudRenderable>("dst", dst, cilantro::RenderingProperties().setPointColor(0,0,1));
    registration_viz.addObject<cilantro::PointCloudRenderable>("src", src_t, cilantro::RenderingProperties().setPointColor(1,0,0));

    // Residuals
    residuals_viz.addObject<cilantro::PointCloudRenderable>("src", src_t, cilantro::RenderingProperties().setUseLighting(false))
            ->setPointValues(residuals);

    std::cout << "Press 'c' to toggle point cloud colors" << std::endl;
    while (!initial_viz.wasStopped() && !registration_viz.wasStopped() && !residuals_viz.wasStopped()) {
        initial_viz.clearRenderArea();
        initial_viz.render();
        registration_viz.render();
        residuals_viz.render();
        pangolin::FinishFrame();
    }

    return 0;
}
