#pragma once

#include <cilantro/random_sample_consensus.hpp>
#include <cilantro/data_containers.hpp>

namespace cilantro {
    class PlaneEstimator : public RandomSampleConsensus<PlaneEstimator,HomogeneousVector<float,3>,float> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PlaneEstimator(const ConstVectorSetMatrixMap<float,3> &points);

        PlaneEstimator& estimateModelParameters(HomogeneousVector<float,3> &model_params);

        HomogeneousVector<float,3> estimateModelParameters();

        PlaneEstimator& estimateModelParameters(const std::vector<size_t> &sample_ind, HomogeneousVector<float,3> &model_params);

        HomogeneousVector<float,3> estimateModelParameters(const std::vector<size_t> &sample_ind);

        PlaneEstimator& computeResiduals(const HomogeneousVector<float,3> &model_params, std::vector<float> &residuals);

        std::vector<float> computeResiduals(const HomogeneousVector<float,3> &model_params);

        inline size_t getDataPointsCount() const { return points_.cols(); }

    private:
        ConstVectorSetMatrixMap<float,3> points_;

        void estimate_params_(const ConstVectorSetMatrixMap<float,3> &points, HomogeneousVector<float,3> &model_params);
    };
}
