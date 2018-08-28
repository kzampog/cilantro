#include <cilantro/point_cloud.hpp>
#include <cilantro/ransac_rigid_transformation_estimator.hpp>
#include <cilantro/visualizer.hpp>
#include <cilantro/common_renderables.hpp>

//cilantro::VectorSet<float,3> selectByIndices(const cilantro::VectorSet<float,3> &elements, const std::vector<size_t> &indices) {
//    cilantro::VectorSet<float,3> res(3,indices.size());
//    for (size_t i = 0; i < indices.size(); i++) {
//        res.col(i) = elements.col(indices[i]);
//    }
//    return res;
//}

void callback(unsigned char key, bool &re_estimate, bool &randomize) {
    if (key == 'a') {
        re_estimate = true;
    }
    if (key == 'd') {
        randomize = true;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Please provide path to PLY file." << std::endl;
        return 0;
    }

    cilantro::PointCloud3f dst(argv[1]);
    cilantro::PointCloud3f src(dst);

    if (dst.isEmpty()) {
        std::cout << "Input cloud is empty!" << std::endl;
        return 0;
    }

    cilantro::Visualizer viz("RigidTransformationRANSACEstimator example", "disp");
    bool re_estimate = false;
    bool randomize = true;
    viz.registerKeyboardCallback('a', std::bind(callback, 'a', std::ref(re_estimate), std::ref(randomize)));
    viz.registerKeyboardCallback('d', std::bind(callback, 'd', std::ref(re_estimate), std::ref(randomize)));

    cilantro::CorrespondenceSet<float> corr(1000);

    viz.addObject<cilantro::PointCloudRenderable>("dst", dst, cilantro::RenderingProperties().setPointColor(0,0,1));
    while (!viz.wasStopped()) {
        if (randomize) {
            randomize = false;

            // Randomly transform src
            cilantro::RigidTransformation3f tform;
            tform.linear().setRandom();
            tform.linear() = tform.rotation();
            tform.translation().setRandom();

            // Build noisy correspondences
            for (size_t i = 0; i < corr.size(); i++) {
                float p = 0 + static_cast<float>(rand())/(static_cast<float>(RAND_MAX/(1-0)));
                corr[i].indexInSecond = std::rand() % src.size();
                if (p > 0.75f) {
                    corr[i].indexInFirst = corr[i].indexInSecond;
                } else {
                    corr[i].indexInFirst = std::rand() % corr.size();
                }
            }

            src.transform(tform);
            viz.addObject<cilantro::PointCloudRenderable>("src", src, cilantro::RenderingProperties().setPointColor(1,0,0));
            viz.addObject<cilantro::PointCorrespondencesRenderable>("corr", dst, src, corr, cilantro::RenderingProperties().setOpacity(0.5f));

            std::cout << "Press 'a' for a transform estimate" << std::endl;
        }

        if (re_estimate) {
            re_estimate = false;

            viz.remove("corr");

            cilantro::RigidTransformationRANSACEstimator3f te(dst.points, src.points, corr);
            te.setMaxInlierResidual(0.01f).setTargetInlierCount((size_t)(0.50*corr.size()))
                .setMaxNumberOfIterations(250).setReEstimationStep(true);

            cilantro::RigidTransformation3f tform = te.estimateModelParameters().getModelParameters();
            const auto& inliers = te.getModelInliers();

            std::cout << "RANSAC iterations: " << te.getNumberOfPerformedIterations() << ", inlier count: " << te.getNumberOfInliers() << std::endl;

            src.transform(tform);

            viz.addObject<cilantro::PointCloudRenderable>("src", src, cilantro::RenderingProperties().setPointColor(1,0,0));

            std::cout << "Press 'd' for a new random pose" << std::endl;
        }

        viz.spinOnce();
    }

    return 0;
}
