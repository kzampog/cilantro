#include <cilantro/rigid_transform_estimator.hpp>
#include <cilantro/registration.hpp>

namespace cilantro {
    RigidTransformEstimator::RigidTransformEstimator(const ConstVectorSetMatrixMap<float,3> &dst_points,
                                                     const ConstVectorSetMatrixMap<float,3> &src_points)
            : RandomSampleConsensus(3, dst_points.cols()/2 + dst_points.cols()%2, 100, 0.01, true),
              dst_points_(dst_points),
              src_points_(src_points)
    {}

    RigidTransformEstimator::RigidTransformEstimator(const ConstVectorSetMatrixMap<float,3> &dst_points,
                                                     const ConstVectorSetMatrixMap<float,3> &src_points,
                                                     const std::vector<size_t> &dst_ind,
                                                     const std::vector<size_t> &src_ind)
            : RandomSampleConsensus(3, dst_ind.size()/2 + dst_ind.size()%2, 100, 0.01, true),
              dst_points_tmp_(3, dst_ind.size()),
              src_points_tmp_(3, src_ind.size()),
              dst_points_(dst_points_tmp_),
              src_points_(src_points_tmp_)
    {
        for (size_t i = 0; i < dst_ind.size(); i++) {
            dst_points_tmp_.col(i) = dst_points.col(dst_ind[i]);
            src_points_tmp_.col(i) = src_points.col(src_ind[i]);
        }
    }

    RigidTransformEstimator& RigidTransformEstimator::estimateModelParameters(RigidTransformation<float,3> &model_params) {
        estimateRigidTransformPointToPointClosedForm<float,3>(dst_points_, src_points_, model_params);
        return *this;
    }

    RigidTransformation<float,3> RigidTransformEstimator::estimateModelParameters() {
        RigidTransformation<float,3> model_params;
        estimateModelParameters(model_params);
        return model_params;
    }

    RigidTransformEstimator& RigidTransformEstimator::estimateModelParameters(const std::vector<size_t> &sample_ind, RigidTransformation<float,3> &model_params) {
        VectorSet<float,3> dst_p(3,sample_ind.size());
        VectorSet<float,3> src_p(3,sample_ind.size());
        for (size_t i = 0; i < sample_ind.size(); i++) {
            dst_p.col(i) = dst_points_.col(sample_ind[i]);
            src_p.col(i) = src_points_.col(sample_ind[i]);
        }
        estimateRigidTransformPointToPointClosedForm<float,3>(dst_p, src_p, model_params);
        return *this;
    }

    RigidTransformation<float,3> RigidTransformEstimator::estimateModelParameters(const std::vector<size_t> &sample_ind) {
        RigidTransformation<float,3> model_params;
        estimateModelParameters(sample_ind, model_params);
        return model_params;
    }

    RigidTransformEstimator& RigidTransformEstimator::computeResiduals(const RigidTransformation<float,3> &model_params, std::vector<float> &residuals) {
        residuals.resize(dst_points_.cols());
        Eigen::Map<Eigen::Matrix<float,1,Eigen::Dynamic> >(residuals.data(),1,residuals.size()) = (model_params*src_points_ - dst_points_).colwise().norm();
        return *this;
    }

    std::vector<float> RigidTransformEstimator::computeResiduals(const RigidTransformation<float,3> &model_params) {
        std::vector<float> residuals;
        computeResiduals(model_params, residuals);
        return residuals;
    }
}
