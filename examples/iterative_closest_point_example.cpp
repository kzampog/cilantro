#include <cilantro/iterative_closest_point.hpp>
#include <cilantro/ply_io.hpp>
#include <cilantro/visualizer.hpp>

#include <ctime>

bool proceed = false;

void callback(Visualizer &viz, int key, void *cookie) {
    if (key == 'a') proceed = true;
}


int main(int argc, char ** argv) {

    Visualizer viz("win", "disp");
    viz.registerKeyboardCallback(std::vector<int>(1,'a'), callback, NULL);

    PointCloud dst, src;
    readPointCloudFromPLYFile(argv[1], dst);

    src = dst;
    for (size_t i = 0; i < src.size(); i++) {
        src.points[i] += 0.01f*Eigen::Vector3f::Random();
    }

    Eigen::Matrix3f R_ref(Eigen::Matrix3f::Random());
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(R_ref, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U(svd.matrixU());
    Eigen::Matrix3f Vt(svd.matrixV().transpose());
    Eigen::Matrix3f tmp(U * Vt);
    if (tmp.determinant() < 0) {
        Eigen::Matrix3f S(Eigen::Matrix3f::Identity());
        S(2, 2) = -1;
        R_ref = U * S * Vt;
    } else {
        R_ref = tmp;
    }
    Eigen::Vector3f t_ref(Eigen::Vector3f::Random());

    src.pointsMatrixMap() = (R_ref*src.pointsMatrixMap()).colwise() + t_ref;
    src.normalsMatrixMap() = R_ref*src.normalsMatrixMap();

    std::vector<size_t> ind(dst.size());
    for (size_t i = 0; i < ind.size(); i++) {
        ind[i] = i;
    }
    std::random_shuffle(ind.begin(), ind.end());

    viz.addPointCloud("dst", dst, Visualizer::RenderingProperties().setDrawingColor(0,0,1));
    viz.addPointCloud("src", src, Visualizer::RenderingProperties().setDrawingColor(1,0,0));

    while (!proceed) {
        viz.spinOnce();
    }
    proceed = false;

    clock_t begin, end;
    double build_time;
    begin = clock();

    Eigen::Matrix3f R_est;
    Eigen::Vector3f t_est;
//    estimateRigidTransformPointToPoint(dst.points, src.points, ind, ind, R_est, t_est);
    estimateRigidTransformPointToPlane(dst.points, dst.normals, src.points, ind, ind, R_est, t_est);

    src.pointsMatrixMap() = (R_est*src.pointsMatrixMap()).colwise() + t_est;
    src.normalsMatrixMap() = R_est*src.normalsMatrixMap();
    estimateRigidTransformPointToPlane(dst.points, dst.normals, src.points, ind, ind, R_est, t_est);
    src.pointsMatrixMap() = (R_est*src.pointsMatrixMap()).colwise() + t_est;
    src.normalsMatrixMap() = R_est*src.normalsMatrixMap();
    estimateRigidTransformPointToPlane(dst.points, dst.normals, src.points, ind, ind, R_est, t_est);
    src.pointsMatrixMap() = (R_est*src.pointsMatrixMap()).colwise() + t_est;
    src.normalsMatrixMap() = R_est*src.normalsMatrixMap();
    estimateRigidTransformPointToPlane(dst.points, dst.normals, src.points, ind, ind, R_est, t_est);
    src.pointsMatrixMap() = (R_est*src.pointsMatrixMap()).colwise() + t_est;
    src.normalsMatrixMap() = R_est*src.normalsMatrixMap();
    estimateRigidTransformPointToPlane(dst.points, dst.normals, src.points, ind, ind, R_est, t_est);
    src.pointsMatrixMap() = (R_est*src.pointsMatrixMap()).colwise() + t_est;
    src.normalsMatrixMap() = R_est*src.normalsMatrixMap();
    estimateRigidTransformPointToPlane(dst.points, dst.normals, src.points, ind, ind, R_est, t_est);

    end = clock();

    build_time = 1000.0*double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "Elapsed time: " << build_time << std::endl;


    std::cout << "TRUE:" << std::endl << R_ref.transpose() << std::endl << t_ref.transpose() << std::endl;
    std::cout << "ESTIMATED:" << std::endl << R_est << std::endl << t_est.transpose() << std::endl;

    src.pointsMatrixMap() = (R_est*src.pointsMatrixMap()).colwise() + t_est;
    src.normalsMatrixMap() = R_est*src.normalsMatrixMap();

    viz.addPointCloud("dst", dst, Visualizer::RenderingProperties().setDrawingColor(0,0,1));
    viz.addPointCloud("src", src, Visualizer::RenderingProperties().setDrawingColor(1,0,0));

    while (!proceed) {
        viz.spinOnce();
    }
    proceed = false;

    return 0;
}
