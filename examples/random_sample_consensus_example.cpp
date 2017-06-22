#include <cilantro/point_cloud.hpp>
#include <cilantro/random_sample_consensus.hpp>
#include <cilantro/principal_component_analysis.hpp>

#include <Eigen/Dense>

typedef Eigen::Vector4f PlaneParameters;

class PlaneEstimator : public RandomSampleConsensus<PlaneEstimator,PlaneParameters> {
public:
    PlaneEstimator(const std::vector<Eigen::Vector3f> &points)
            : RandomSampleConsensus(3, 50, 100, 0.1, true),
              points_(&points)
    {}

    PlaneEstimator& estimateModelParameters(PlaneParameters &model_params) {
        compute_(*points_, model_params);
        return *this;
    }
    PlaneEstimator& estimateModelParameters(PlaneParameters &model_params, const std::vector<size_t> &sample_ind) {
        std::vector<Eigen::Vector3f> points(sample_ind.size());
        for (size_t i = 0; i < sample_ind.size(); i++) {
            points[i] = (*points_)[sample_ind[i]];
        }
        compute_(points, model_params);
        return *this;
    }

    PlaneEstimator& computeResiduals(const PlaneParameters &model_params, std::vector<float> &residuals) {
        residuals.resize(points_->size());
        float norm = model_params.head(3).norm();
        for (size_t i = 0; i < points_->size(); i++) {
            residuals[i] = std::abs((*points_)[i].dot(model_params.head(3)) + model_params[3])/norm;
        }

        return *this;
    }

private:
    const std::vector<Eigen::Vector3f> * points_;

    void compute_(const std::vector<Eigen::Vector3f> &points, PlaneParameters &model_params) {
        PrincipalComponentAnalysis3D pca(points);
        Eigen::Vector3f normal = pca.getEigenVectorsMatrix().col(2);
        model_params.head(3) = normal;
        model_params[3] = -normal.dot(pca.getDataMean());
    }
};

int main(int argc, char **argv) {

    PointCloud cloud;
    cloud.points.push_back(Eigen::Vector3f(0, 0, 0));
    cloud.points.push_back(Eigen::Vector3f(1, 0, 0));
    cloud.points.push_back(Eigen::Vector3f(0, 1, 0));
//    cloud.points.push_back(Eigen::Vector3f(0, 0, 1));
//    cloud.points.push_back(Eigen::Vector3f(0, 1, 1));
//    cloud.points.push_back(Eigen::Vector3f(1, 0, 1));
//    cloud.points.push_back(Eigen::Vector3f(1, 1, 0));
//    cloud.points.push_back(Eigen::Vector3f(1, 1, 1));

    PlaneParameters plane;
    std::vector<float> residuals;

    PlaneEstimator pe(cloud.points);
    pe.estimateModelParameters(plane).computeResiduals(plane, residuals);

    std::cout << plane.transpose() << std::endl;
    for (size_t i = 0; i < residuals.size(); i++) {
        std::cout << residuals[i] << "  ";
    }
    std::cout << std::endl;


    return 0;
}
