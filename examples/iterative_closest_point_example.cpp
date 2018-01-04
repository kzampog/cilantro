#include <cilantro/iterative_closest_point.hpp>
#include <cilantro/io.hpp>
#include <cilantro/voxel_grid.hpp>
#include <cilantro/visualizer.hpp>

void callback(bool &proceed) {
    proceed = true;
}

int main(int argc, char ** argv) {
    cilantro::PointCloud dst, src;
    cilantro::readPointCloudFromPLYFile(argv[1], dst);

    dst = cilantro::VoxelGrid(dst, 0.005).getDownsampledCloud();

    src = dst;
    for (size_t i = 0; i < src.size(); i++) {
        src.points[i] += 0.005f*Eigen::Vector3f::Random();
        src.normals[i] += 0.005f*Eigen::Vector3f::Random();
        src.normals[i].normalize();
        src.colors[i] += 0.010f*Eigen::Vector3f::Random();
    }

    cilantro::PointCloud dst2;
    for (size_t i = 0; i < dst.size(); i++) {
        if (dst.points[i][0] > -0.4) {
            dst2.points.push_back(dst.points[i]);
            dst2.normals.push_back(dst.normals[i]);
            dst2.colors.push_back(dst.colors[i]);
        }
    }
    dst = dst2;

    Eigen::Matrix3f R_ref;
    R_ref = Eigen::AngleAxisf(-0.10 ,Eigen::Vector3f::UnitZ()) *
            Eigen::AngleAxisf(0.01, Eigen::Vector3f::UnitY()) *
            Eigen::AngleAxisf(-0.07, Eigen::Vector3f::UnitX());
    Eigen::Vector3f t_ref(-0.20, -0.05, 0.09);

    src.pointsMatrixMap() = (R_ref*src.pointsMatrixMap()).colwise() + t_ref;
    src.normalsMatrixMap() = R_ref*src.normalsMatrixMap();

//    std::vector<size_t> ind(dst.size());
//    for (size_t i = 0; i < ind.size(); i++) {
//        ind[i] = i;
//    }
//    std::random_shuffle(ind.begin(), ind.end());

    cilantro::Visualizer viz("IterativeClosestPoint example", "disp");
    bool proceed = false;
    viz.registerKeyboardCallback('a', std::bind(callback, std::ref(proceed)));

    viz.addPointCloud("dst", dst.points, cilantro::RenderingProperties().setPointColor(0,0,1));
    viz.addPointCloudNormals("dst", dst.normals);
    viz.addPointCloud("src", src.points, cilantro::RenderingProperties().setPointColor(1,0,0));
    viz.addPointCloudNormals("src", src.normals);

    std::cout << "Press 'a' to compute transformation" << std::endl;
    while (!proceed && !viz.wasStopped()) {
        if (viz.wasStopped()) return 0;
        viz.spinOnce();
    }
    proceed = false;

    auto start = std::chrono::high_resolution_clock::now();

    Eigen::Matrix3f R_est;
    Eigen::Vector3f t_est;

//    cilantro::IterativeClosestPoint icp(dst, src, cilantro::IterativeClosestPoint::Metric::POINT_TO_POINT);
    cilantro::IterativeClosestPoint icp(dst, src, cilantro::IterativeClosestPoint::Metric::POINT_TO_PLANE);
//    cilantro::IterativeClosestPoint icp(dst, src, cilantro::IterativeClosestPoint::Metric::COMBINED);
//    icp.setPointToPointMetricWeight(0.01f);
//    icp.setPointToPlaneMetricWeight(1.0f);

    icp.setCorrespondencesType(cilantro::IterativeClosestPoint::CorrespondencesType::POINTS_NORMALS_COLORS);
    icp.setCorrespondencePointWeight(1.0);
    icp.setCorrespondenceNormalWeight(50.0);
    icp.setCorrespondenceColorWeight(50.0);

    icp.setMaxCorrespondenceDistance(0.5f).setConvergenceTolerance(1e-3f).setMaxNumberOfIterations(200).setMaxNumberOfOptimizationStepIterations(1);

//    icp.setInitialTransformation(R_ref.transpose(), (-R_ref.transpose()*t_ref));
    icp.getTransformation(R_est, t_est);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Registration time: " << elapsed.count() << "ms" << std::endl;

    std::cout << "Iterations performed: " << icp.getPerformedIterationsCount() << std::endl;
    std::cout << "Has converged: " << icp.hasConverged() << std::endl;

    std::cout << "TRUE transformation R:" << std::endl << R_ref.transpose() << std::endl;
    std::cout << "TRUE transformation t:" << std::endl << (-R_ref.transpose()*t_ref).transpose() << std::endl;
    std::cout << "ESTIMATED transformation R:" << std::endl << R_est << std::endl;
    std::cout << "ESTIMATED transformation t:" << std::endl << t_est.transpose() << std::endl;

    cilantro::PointCloud src_trans(src);

    src_trans.pointsMatrixMap() = (R_est*src_trans.pointsMatrixMap()).colwise() + t_est;
    src_trans.normalsMatrixMap() = R_est*src_trans.normalsMatrixMap();

    viz.addPointCloud("dst", dst.points, cilantro::RenderingProperties().setPointColor(0,0,1));
    viz.addPointCloudNormals("dst", dst.normals);
    viz.addPointCloud("src", src_trans.points, cilantro::RenderingProperties().setPointColor(1,0,0));
    viz.addPointCloudNormals("src", src_trans.normals);

    std::cout << "Press 'a' to compute residuals" << std::endl;
    while (!proceed && !viz.wasStopped()) {
        if (viz.wasStopped()) return 0;
        viz.spinOnce();
    }
    proceed = false;

    start = std::chrono::high_resolution_clock::now();
    auto residuals = icp.getResiduals(cilantro::IterativeClosestPoint::CorrespondencesType::POINTS, cilantro::IterativeClosestPoint::Metric::POINT_TO_POINT);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Residual computation time: " << elapsed.count() << "ms" << std::endl;

    viz.clear();
    viz.addPointCloud("src", src_trans.points, cilantro::RenderingProperties().setUseLighting(false)).addPointCloudValues("src", residuals);
    viz.addPointCloudNormals("src", src_trans.normals);

    while (!proceed && !viz.wasStopped()) {
        if (viz.wasStopped()) return 0;
        viz.spinOnce();
    }

    return 0;
}
