#include <cilantro/point_cloud.hpp>
#include <cilantro/rigid_transform_estimator.hpp>
#include <cilantro/io.hpp>
#include <cilantro/visualizer.hpp>

cilantro::VectorSet<float,3> selectByIndices(const cilantro::VectorSet<float,3> &elements, const std::vector<size_t> &indices) {
    cilantro::VectorSet<float,3> res(3,indices.size());
    for (size_t i = 0; i < indices.size(); i++) {
        res.col(i) = elements.col(indices[i]);
    }
    return res;
}

void callback(unsigned char key, bool &re_estimate, bool &randomize) {
    if (key == 'a') {
        re_estimate = true;
    }
    if (key == 'd') {
        randomize = true;
    }
}

int main(int argc, char **argv) {

    cilantro::PointCloud3f dst;
    readPointCloudFromPLYFile(argv[1], dst);

    cilantro::PointCloud3f src(dst);

    cilantro::Visualizer viz("RigidTransformationEstimator example", "disp");
    bool re_estimate = false;
    bool randomize = true;
    viz.registerKeyboardCallback('a', std::bind(callback, 'a', std::ref(re_estimate), std::ref(randomize)));
    viz.registerKeyboardCallback('d', std::bind(callback, 'd', std::ref(re_estimate), std::ref(randomize)));

    std::vector<size_t> dst_ind(100);
    std::vector<size_t> src_ind(100);

    viz.addPointCloud("dst", dst, cilantro::RenderingProperties().setPointColor(0,0,1));
    while (!viz.wasStopped()) {
        if (randomize) {
            randomize = false;

            // Randomly transform src
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

            // Build noisy correspondences
            for (size_t i = 0; i < dst_ind.size(); i++) {
                float p = 0 + static_cast<float>(rand())/(static_cast<float>(RAND_MAX/(1-0)));
                src_ind[i] = std::rand() % src.size();
                if (p > 0.4f) {
                    dst_ind[i] = src_ind[i];
                } else {
                    dst_ind[i] = std::rand() % dst.size();
                }
            }

            src.transform(R_ref, t_ref);
            viz.addPointCloud("src", src, cilantro::RenderingProperties().setPointColor(1,0,0));
            viz.addPointCorrespondences("corr", selectByIndices(src.points, src_ind), selectByIndices(dst.points, dst_ind), cilantro::RenderingProperties().setOpacity(0.5f));

            std::cout << "Press 'a' for a transform estimate" << std::endl;
        }

        if (re_estimate) {
            re_estimate = false;

            viz.remove("corr");

            cilantro::RigidTransformationEstimator3f te(dst.points, src.points, dst_ind, src_ind);
            te.setMaxInlierResidual(0.01).setTargetInlierCount((size_t)(0.50*dst_ind.size())).setMaxNumberOfIterations(250).setReEstimationStep(true);
            cilantro::RigidTransformation<float,3> tform = te.getModelParameters();
            std::vector<size_t> inliers = te.getModelInliers();

            std::cout << "RANSAC iterations: " << te.getPerformedIterationsCount() << ", inlier count: " << te.getNumberOfInliers() << std::endl;

            src.points = tform*src.points;

            viz.addPointCloud("src", src, cilantro::RenderingProperties().setPointColor(1,0,0));

            std::cout << "Press 'd' for a new random pose" << std::endl;
        }

        viz.spinOnce();
    }

    return 0;
}
