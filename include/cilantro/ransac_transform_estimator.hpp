#pragma once

#include <cilantro/ransac_base.hpp>
#include <cilantro/transform_estimation.hpp>

namespace cilantro {
    template <class TransformT>
    class TransformRANSACEstimator : public RandomSampleConsensusBase<TransformRANSACEstimator<TransformT>,TransformT,typename TransformT::Scalar> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef TransformT Transform;

        typedef typename TransformT::Scalar Scalar;

        enum {
            Dim = TransformT::Dim,
            MinSampleSize = (int(TransformT::Mode) == int(Eigen::Isometry)) ? Dim : Dim + 1
        };

        TransformRANSACEstimator(const ConstVectorSetMatrixMap<Scalar,Dim> &dst_points,
                                 const ConstVectorSetMatrixMap<Scalar,Dim> &src_points)
                : RandomSampleConsensusBase<TransformRANSACEstimator<TransformT>,TransformT,Scalar>(MinSampleSize, dst_points.cols()/2 + dst_points.cols()%2, 100, (Scalar)0.01, true),
                  dst_points_(dst_points),
                  src_points_(src_points)
        {}

        template <class CorrespondencesT>
        TransformRANSACEstimator(const ConstVectorSetMatrixMap<Scalar,Dim> &dst_points,
                                 const ConstVectorSetMatrixMap<Scalar,Dim> &src_points,
                                 const CorrespondencesT &corr)
                : RandomSampleConsensusBase<TransformRANSACEstimator<TransformT>,TransformT,Scalar>(MinSampleSize, corr.size()/2 + corr.size()%2, 100, (Scalar)0.01, true),
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

        TransformRANSACEstimator(const ConstVectorSetMatrixMap<Scalar,Dim> &dst_points,
                                 const ConstVectorSetMatrixMap<Scalar,Dim> &src_points,
                                 const std::vector<size_t> &dst_ind,
                                 const std::vector<size_t> &src_ind)
                : RandomSampleConsensusBase<TransformRANSACEstimator<TransformT>,TransformT,Scalar>(MinSampleSize, dst_ind.size()/2 + dst_ind.size()%2, 100, (Scalar)0.01, true),
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

        inline TransformRANSACEstimator& estimateModel(TransformT &model_params) {
            estimateTransformPointToPointMetric(dst_points_, src_points_, model_params);
            return *this;
        }

        inline TransformT estimateModel() {
            TransformT model_params;
            estimateModel(model_params);
            return model_params;
        }

        TransformRANSACEstimator& estimateModel(const std::vector<size_t> &sample_ind,
                                                TransformT &model_params)
        {
            VectorSet<Scalar,Dim> dst_p(dst_points_.rows(), sample_ind.size());
            VectorSet<Scalar,Dim> src_p(src_points_.rows(), sample_ind.size());
            for (size_t i = 0; i < sample_ind.size(); i++) {
                dst_p.col(i) = dst_points_.col(sample_ind[i]);
                src_p.col(i) = src_points_.col(sample_ind[i]);
            }
            estimateTransformPointToPointMetric(dst_p, src_p, model_params);
            return *this;
        }

        inline TransformT estimateModel(const std::vector<size_t> &sample_ind) {
            TransformT model_params;
            estimateModel(sample_ind, model_params);
            return model_params;
        }

        inline TransformRANSACEstimator& computeResiduals(const TransformT &model_params,
                                                          std::vector<Scalar> &residuals)
        {
            residuals.resize(dst_points_.cols());
#pragma omp parallel for
            for (size_t i = 0; i < dst_points_.cols(); i++) {
                residuals[i] = (model_params*src_points_.col(i) - dst_points_.col(i)).norm();
            }
            return *this;
        }

        inline std::vector<Scalar> computeResiduals(const TransformT &model_params) {
            std::vector<Scalar> residuals;
            computeResiduals(model_params, residuals);
            return residuals;
        }

        inline size_t getDataPointsCount() const { return dst_points_.cols(); }

    private:
        VectorSet<Scalar,Dim> dst_points_tmp_;
        VectorSet<Scalar,Dim> src_points_tmp_;
        ConstVectorSetMatrixMap<Scalar,Dim> dst_points_;
        ConstVectorSetMatrixMap<Scalar,Dim> src_points_;
    };

    typedef TransformRANSACEstimator<RigidTransform<float,2>> RigidTransformRANSACEstimator2f;
    typedef TransformRANSACEstimator<RigidTransform<float,3>> RigidTransformRANSACEstimator3f;
    typedef TransformRANSACEstimator<RigidTransform<double,2>> RigidTransformRANSACEstimator2d;
    typedef TransformRANSACEstimator<RigidTransform<double,3>> RigidTransformRANSACEstimator3d;

    typedef TransformRANSACEstimator<AffineTransform<float,2>> AffineTransformRANSACEstimator2f;
    typedef TransformRANSACEstimator<AffineTransform<float,3>> AffineTransformRANSACEstimator3f;
    typedef TransformRANSACEstimator<AffineTransform<double,2>> AffineTransformRANSACEstimator2d;
    typedef TransformRANSACEstimator<AffineTransform<double,3>> AffineTransformRANSACEstimator3d;
}
