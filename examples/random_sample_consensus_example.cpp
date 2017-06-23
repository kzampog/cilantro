#include <cilantro/point_cloud.hpp>
#include <cilantro/random_sample_consensus.hpp>

#include <cilantro/principal_component_analysis.hpp>

#include <cilantro/ply_io.hpp>
#include <cilantro/visualizer.hpp>

#include <iostream>

typedef Eigen::Vector4f PlaneParameters;

class PlaneEstimator : public RandomSampleConsensus<PlaneEstimator,PlaneParameters> {
public:
    PlaneEstimator(const std::vector<Eigen::Vector3f> &points)
            : RandomSampleConsensus(3, 50, 100, 0.1, true),
              points_(&points)
    {}

    PlaneEstimator& estimateModelParameters(PlaneParameters &model_params) {
        estimate_params_(*points_, model_params);
        return *this;
    }

    PlaneParameters estimateModelParameters() {
        PlaneParameters model_params;
        estimateModelParameters(model_params);
        return model_params;
    }

    PlaneEstimator& estimateModelParameters(const std::vector<size_t> &sample_ind, PlaneParameters &model_params) {
        std::vector<Eigen::Vector3f> points(sample_ind.size());
        for (size_t i = 0; i < sample_ind.size(); i++) {
            points[i] = (*points_)[sample_ind[i]];
        }
        estimate_params_(points, model_params);
        return *this;
    }

    PlaneParameters estimateModelParameters(const std::vector<size_t> &sample_ind) {
        PlaneParameters model_params;
        estimateModelParameters(sample_ind, model_params);
        return model_params;
    }

    PlaneEstimator& computeResiduals(std::vector<float> &residuals) {
        return computeResiduals(getModelParameters(), residuals);
    }

    std::vector<float> computeResiduals() {
        std::vector<float> residuals;
        computeResiduals(residuals);
        return residuals;
    }

    PlaneEstimator& computeResiduals(const PlaneParameters &model_params, std::vector<float> &residuals) {
        residuals.resize(points_->size());
        float norm = model_params.head(3).norm();
        for (size_t i = 0; i < points_->size(); i++) {
            residuals[i] = std::abs((*points_)[i].dot(model_params.head(3)) + model_params[3])/norm;
        }
        return *this;
    }

    std::vector<float> computeResiduals(const PlaneParameters &model_params) {
        std::vector<float> residuals;
        computeResiduals(model_params, residuals);
        return residuals;
    }

    size_t getDataPointsCount() {
        return points_->size();
    }

private:
    const std::vector<Eigen::Vector3f> *points_;

    void estimate_params_(const std::vector<Eigen::Vector3f> &points, PlaneParameters &model_params) {
        PrincipalComponentAnalysis3D pca(points);
        Eigen::Vector3f normal = pca.getEigenVectorsMatrix().col(2);
        model_params.head(3) = normal;
        model_params[3] = -normal.dot(pca.getDataMean());
    }
};

bool re_estimate = false;

void callback(Visualizer &viz, int key, void *cookie) {
    if (key == 'a') {
        re_estimate = true;
    }
}

int main(int argc, char **argv) {

    PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

    Visualizer viz("win", "disp");
    viz.registerKeyboardCallback(std::vector<int>(1,'a'), callback, NULL);

    PlaneParameters plane;
    std::vector<size_t> inliers;

    viz.addPointCloud("cloud", cloud);
    while (!viz.wasStopped()) {
        if (re_estimate) {
            re_estimate = false;

            PlaneEstimator pe(cloud.points);
            pe.setMaxInlierResidual(0.01).setTargetInlierCount((size_t)(0.20*cloud.points.size())).setMaxNumberOfIterations(250).setReEstimationStep(true);
            pe.getModel(plane, inliers);
            std::cout << "RANSAC iterations: " << pe.getPerformedIterationsCount() << std::endl;

            PointCloud planar_cloud(cloud, inliers);
            viz.addPointCloud("plane", planar_cloud, Visualizer::RenderingProperties().setDrawingColor(1,0,0).setPointSize(5.0));
        }
        viz.spinOnce();
    }

    return 0;
}
