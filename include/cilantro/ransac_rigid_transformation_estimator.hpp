#pragma once

#include <cilantro/ransac_base.hpp>
#include <cilantro/rigid_registration_utilities.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class RigidTransformationRANSACEstimator : public RandomSampleConsensusBase<RigidTransformationRANSACEstimator<ScalarT,EigenDim>,RigidTransformation<ScalarT,EigenDim>,ScalarT> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        RigidTransformationRANSACEstimator(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst_points,
                                           const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src_points)
                : RandomSampleConsensusBase<RigidTransformationRANSACEstimator<ScalarT,EigenDim>,RigidTransformation<ScalarT,EigenDim>,ScalarT>(dst_points.rows(), dst_points.cols()/2 + dst_points.cols()%2, 100, 0.01, true),
                  dst_points_(dst_points),
                  src_points_(src_points)
        {}

        RigidTransformationRANSACEstimator(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst_points,
                                           const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src_points,
                                           const CorrespondenceSet<ScalarT> &corr)
                : RandomSampleConsensusBase<RigidTransformationRANSACEstimator<ScalarT,EigenDim>,RigidTransformation<ScalarT,EigenDim>,ScalarT>(dst_points.rows(), corr.size()/2 + corr.size()%2, 100, 0.01, true),
                  dst_points_tmp_(dst_points.rows(), corr.size()),
                  src_points_tmp_(src_points.rows(), corr.size()),
                  dst_points_(dst_points_tmp_),
                  src_points_(src_points_tmp_)
        {
            for (size_t i = 0; i < corr.size(); i++) {
                dst_points_tmp_.col(i) = dst_points.col(corr[i].indexInFirst);
                src_points_tmp_.col(i) = src_points.col(corr[i].indexInSecond);
            }
        }

        RigidTransformationRANSACEstimator(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst_points,
                                           const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src_points,
                                           const std::vector<size_t> &dst_ind,
                                           const std::vector<size_t> &src_ind)
                : RandomSampleConsensusBase<RigidTransformationRANSACEstimator<ScalarT,EigenDim>,RigidTransformation<ScalarT,EigenDim>,ScalarT>(dst_points.rows(), dst_ind.size()/2 + dst_ind.size()%2, 100, 0.01, true),
                  dst_points_tmp_(dst_points.rows(), dst_ind.size()),
                  src_points_tmp_(src_points.rows(), src_ind.size()),
                  dst_points_(dst_points_tmp_),
                  src_points_(src_points_tmp_)
        {
            for (size_t i = 0; i < dst_ind.size(); i++) {
                dst_points_tmp_.col(i) = dst_points.col(dst_ind[i]);
                src_points_tmp_.col(i) = src_points.col(src_ind[i]);
            }
        }

        inline RigidTransformationRANSACEstimator& fitModelParameters(RigidTransformation<ScalarT,EigenDim> &model_params) {
            estimateRigidTransformPointToPointClosedForm<ScalarT,EigenDim>(dst_points_, src_points_, model_params);
            return *this;
        }

        inline RigidTransformation<ScalarT,EigenDim> fitModelParameters() {
            RigidTransformation<ScalarT,EigenDim> model_params;
            fitModelParameters(model_params);
            return model_params;
        }

        RigidTransformationRANSACEstimator& fitModelParameters(const std::vector<size_t> &sample_ind,
                                                               RigidTransformation<ScalarT,EigenDim> &model_params)
        {
            VectorSet<ScalarT,EigenDim> dst_p(dst_points_.rows(), sample_ind.size());
            VectorSet<ScalarT,EigenDim> src_p(src_points_.rows(), sample_ind.size());
            for (size_t i = 0; i < sample_ind.size(); i++) {
                dst_p.col(i) = dst_points_.col(sample_ind[i]);
                src_p.col(i) = src_points_.col(sample_ind[i]);
            }
            estimateRigidTransformPointToPointClosedForm<ScalarT,EigenDim>(dst_p, src_p, model_params);
            return *this;
        }

        inline RigidTransformation<ScalarT,EigenDim> fitModelParameters(const std::vector<size_t> &sample_ind) {
            RigidTransformation<ScalarT,EigenDim> model_params;
            fitModelParameters(sample_ind, model_params);
            return model_params;
        }

        inline RigidTransformationRANSACEstimator& computeResiduals(const RigidTransformation<ScalarT,EigenDim> &model_params,
                                                                    std::vector<ScalarT> &residuals)
        {
            residuals.resize(dst_points_.cols());
#pragma omp parallel for
            for (size_t i = 0; i < dst_points_.cols(); i++) {
                residuals[i] = (model_params*src_points_.col(i) - dst_points_.col(i)).norm();
            }
            return *this;
        }

        inline std::vector<ScalarT> computeResiduals(const RigidTransformation<ScalarT,EigenDim> &model_params) {
            std::vector<ScalarT> residuals;
            computeResiduals(model_params, residuals);
            return residuals;
        }

        inline size_t getDataPointsCount() const { return dst_points_.cols(); }

    private:
        VectorSet<ScalarT,EigenDim> dst_points_tmp_;
        VectorSet<ScalarT,EigenDim> src_points_tmp_;
        ConstVectorSetMatrixMap<ScalarT,EigenDim> dst_points_;
        ConstVectorSetMatrixMap<ScalarT,EigenDim> src_points_;
    };

    typedef RigidTransformationRANSACEstimator<float,2> RigidTransformationRANSACEstimator2f;
    typedef RigidTransformationRANSACEstimator<double,2> RigidTransformationRANSACEstimator2d;
    typedef RigidTransformationRANSACEstimator<float,3> RigidTransformationRANSACEstimator3f;
    typedef RigidTransformationRANSACEstimator<double,3> RigidTransformationRANSACEstimator3d;
}
