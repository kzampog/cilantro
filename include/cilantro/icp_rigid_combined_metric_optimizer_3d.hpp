#pragma once

#include <cilantro/rigid_registration_utilities.hpp>

namespace cilantro {
    template <typename ScalarT>
    class RigidICPCombinedMetricOptimizer3 {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef RigidTransformation<ScalarT,3> Transformation;
        typedef VectorSet<ScalarT,1> ResidualVector;

        RigidICPCombinedMetricOptimizer3(const ConstVectorSetMatrixMap<ScalarT,3> &dst_p,
                                         const ConstVectorSetMatrixMap<ScalarT,3> &dst_n,
                                         const ConstVectorSetMatrixMap<ScalarT,3> &src_p)
                : dst_points_(dst_p), dst_normals_(dst_n), src_points_(src_p),
                  src_points_trans_(src_points_.rows(), src_points_.cols()),
                  max_optimization_iterations_(1),
                  optimization_convergence_tol_((ScalarT)1e-5),
                  point_to_point_weight_((ScalarT)0.1),
                  point_to_plane_weight_((ScalarT)1.0)
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
            estimateRigidTransformCombinedMetric3D<ScalarT,CorrScalarT>(dst_points_, dst_normals_, src_points_trans_, correspondences, point_to_point_weight_, point_to_plane_weight_, tform_iter, max_optimization_iterations_, optimization_convergence_tol_);

            transform = tform_iter*transform;
            transform.linear() = transform.rotation();
            delta_norm = std::sqrt((tform_iter.linear() - Eigen::Matrix<ScalarT,3,3>::Identity()).squaredNorm() + tform_iter.translation().squaredNorm());
        }

        ResidualVector computeResiduals(const Transformation &transform) {
            ResidualVector res(1, src_points_.cols());
            KDTree<ScalarT,3,KDTreeDistanceAdaptors::L2> dst_tree(dst_points_);
            NearestNeighborSearchResult<ScalarT> nn;
            Vector<ScalarT,3> src_p_trans;
#pragma omp parallel for shared (res) private (nn, src_p_trans)
            for (size_t i = 0; i < src_points_.cols(); i++) {
                src_p_trans = transform*src_points_.col(i);
                dst_tree.nearestNeighborSearch(src_p_trans, nn);
                ScalarT point_to_plane_dist = dst_normals_.col(nn.index).dot(dst_points_.col(nn.index) - src_p_trans);
                res[i] = point_to_point_weight_*(dst_points_.col(nn.index) - src_p_trans).squaredNorm() + point_to_plane_weight_*point_to_plane_dist*point_to_plane_dist;
            }
            return res;
        }

        inline ScalarT getPointToPointMetricWeight() const { return point_to_point_weight_; }

        inline RigidICPCombinedMetricOptimizer3& setPointToPointMetricWeight(ScalarT weight) {
            point_to_point_weight_ = weight;
            return *this;
        }

        inline ScalarT getPointToPlaneMetricWeight() const { return point_to_plane_weight_; }

        inline RigidICPCombinedMetricOptimizer3& setPointToPlaneMetricWeight(ScalarT weight) {
            point_to_plane_weight_ = weight;
            return *this;
        }

        inline size_t getMaxNumberOfOptimizationIterations() const { return max_optimization_iterations_; }

        inline RigidICPCombinedMetricOptimizer3& setMaxNumberOfOptimizationIterations(size_t max_iter) {
            max_optimization_iterations_ = max_iter;
            return *this;
        }

        inline ScalarT geOptimizationtConvergenceTolerance() const { return optimization_convergence_tol_; }

        inline RigidICPCombinedMetricOptimizer3& setOptimizationConvergenceTolerance(ScalarT conv_tol) {
            optimization_convergence_tol_ = conv_tol;
            return *this;
        }

    private:
        ConstVectorSetMatrixMap<ScalarT,3> dst_points_;
        ConstVectorSetMatrixMap<ScalarT,3> dst_normals_;
        ConstVectorSetMatrixMap<ScalarT,3> src_points_;
        VectorSet<ScalarT,3> src_points_trans_;

        size_t max_optimization_iterations_;
        ScalarT optimization_convergence_tol_;

        ScalarT point_to_point_weight_;
        ScalarT point_to_plane_weight_;
    };

    typedef RigidICPCombinedMetricOptimizer3<float> RigidICPCombinedMetricOptimizer3f;
    typedef RigidICPCombinedMetricOptimizer3<double> RigidICPCombinedMetricOptimizer3d;
}
