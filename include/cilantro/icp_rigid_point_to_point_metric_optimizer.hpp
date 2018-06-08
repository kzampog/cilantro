#pragma once

#include <cilantro/rigid_registration_utilities.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class RigidICPPointToPointMetricOptimizer {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef RigidTransformation<ScalarT,EigenDim> Transformation;
        typedef VectorSet<ScalarT,1> ResidualVector;

        RigidICPPointToPointMetricOptimizer(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &dst,
                                            const ConstVectorSetMatrixMap<ScalarT,EigenDim> &src)
                : dst_points_(dst), src_points_(src),
                  src_points_trans_(src_points_.rows(), src_points_.cols())
        {}

        template <typename CorrScalarT>
        void initialize(Transformation &transform, CorrespondenceSet<CorrScalarT> &correspondences) {
            transform.setIdentity();
            correspondences.reserve(std::max(dst_points_.cols(), src_points_.cols()));
        }

        template <typename CorrScalarT>
        void refineTransformation(const CorrespondenceSet<CorrScalarT> &correspondences,
                                  Transformation &transform,
                                  ScalarT &delta_norm)
        {
#pragma omp parallel for
            for (size_t i = 0; i < src_points_.cols(); i++) {
                src_points_trans_.col(i) = transform*src_points_.col(i);
            }

            Transformation tform_iter;
            estimateRigidTransformPointToPointClosedForm<ScalarT,EigenDim,CorrScalarT>(dst_points_, src_points_trans_, correspondences, tform_iter);

            transform = tform_iter*transform;
            transform.linear() = transform.rotation();
            delta_norm = std::sqrt((tform_iter.linear() - Eigen::Matrix<ScalarT,EigenDim,EigenDim>::Identity(src_points_.rows(),src_points_.rows())).squaredNorm() + tform_iter.translation().squaredNorm());
        }

        ResidualVector computeResiduals(const Transformation &transform) {
            VectorSet<ScalarT,1> res(1, src_points_.cols());
            KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> dst_tree(dst_points_);
            NearestNeighborSearchResult<ScalarT> nn;
            Vector<ScalarT,3> src_p_trans;
#pragma omp parallel for shared (res) private (nn, src_p_trans)
            for (size_t i = 0; i < src_points_.cols(); i++) {
                src_p_trans = transform*src_points_.col(i);
                dst_tree.nearestNeighborSearch(src_p_trans, nn);
                res[i] = (dst_points_.col(nn.index) - src_p_trans).squaredNorm();
            }
            return res;
        }

    private:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> dst_points_;
        ConstVectorSetMatrixMap<ScalarT,EigenDim> src_points_;
        VectorSet<ScalarT,EigenDim> src_points_trans_;
    };

    typedef RigidICPPointToPointMetricOptimizer<float,2> RigidICPPointToPointMetricOptimizer2f;
    typedef RigidICPPointToPointMetricOptimizer<double,2> RigidICPPointToPointMetricOptimizer2d;
    typedef RigidICPPointToPointMetricOptimizer<float,3> RigidICPPointToPointMetricOptimizer3f;
    typedef RigidICPPointToPointMetricOptimizer<double,3> RigidICPPointToPointMetricOptimizer3d;
}
