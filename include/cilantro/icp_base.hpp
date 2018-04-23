#pragma once

#include <Eigen/Dense>

namespace cilantro {
    enum struct CorrespondenceSearchDirection {FIRST_TO_SECOND, SECOND_TO_FIRST, BOTH};

    // CRTP base class
    template <class ICPInstanceT, class TransformT, class ResidualT, typename PointScalarT, typename CorrValueT>
    class IterativeClosestPointBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        IterativeClosestPointBase(size_t max_iter = 15, PointScalarT conv_tol = 1e-5)
                : max_iterations_(max_iter), iterations_(0), convergence_tol_(conv_tol), last_delta_norm_(std::numeric_limits<PointScalarT>::infinity()),
                  corr_search_dir_(CorrespondenceSearchDirection::SECOND_TO_FIRST), corr_max_distance_(0.01*0.01), corr_inlier_fraction_(1.0), corr_require_reciprocal_(false)
        {}

        inline size_t getMaxNumberOfIterations() const { return max_iterations_; }

        inline ICPInstanceT& setMaxNumberOfIterations(size_t max_iter) { max_iterations_ = max_iter; return *static_cast<ICPInstanceT*>(this); }

        inline size_t getNumberOfPerformedIterations() const { return iterations_; }

        inline PointScalarT getConvergenceTolerance() const { return convergence_tol_; }

        inline ICPInstanceT& setConvergenceTolerance(PointScalarT conv_tol) { convergence_tol_ = conv_tol; return *static_cast<ICPInstanceT*>(this); }

        inline const TransformT& getInitialTransformation() const { return transform_init_; }

        inline ICPInstanceT& setInitialTransformation(const TransformT& tform_init) { transform_init_ = tform_init; return *static_cast<ICPInstanceT*>(this); }

        inline TransformT& initialTransformation() { return transform_init_; }

        inline const CorrespondenceSearchDirection& getCorrespondenceSearchDirection() const { return corr_search_dir_; }

        inline ICPInstanceT& setCorrespondenceSearchDirection(const CorrespondenceSearchDirection &search_dir) { corr_search_dir_ = search_dir; return *static_cast<ICPInstanceT*>(this); }

        inline CorrValueT getMaxCorrespondenceDistance() const { return corr_max_distance_; }

        inline ICPInstanceT& setMaxCorrespondenceDistance(CorrValueT dist_thresh) { corr_max_distance_ = dist_thresh; return *static_cast<ICPInstanceT*>(this); }

        inline double getCorrespondenceInlierFraction() const { return corr_inlier_fraction_; }

        inline ICPInstanceT& setCorrespondenceInlierFraction(double fraction) { corr_inlier_fraction_ = fraction; return *static_cast<ICPInstanceT*>(this); }

        inline bool getRequireReciprocalCorrespondences() const { return corr_require_reciprocal_; }

        inline ICPInstanceT& setRequireReciprocalCorrespondences(bool require_reciprocal) { corr_require_reciprocal_ = require_reciprocal; return *static_cast<ICPInstanceT*>(this); }

        inline PointScalarT getLastUpdateNorm() const { return last_delta_norm_; }

        ICPInstanceT& estimateTransformation() {
            // Main ICP loop
            ICPInstanceT& icp_instance = *static_cast<ICPInstanceT*>(this);
            icp_instance.initializeComputation();

            transform_ = transform_init_;
            iterations_ = 0;
            last_delta_norm_ = std::numeric_limits<PointScalarT>::infinity();
            while (iterations_ < max_iterations_) {
                if (!icp_instance.updateCorrespondences()) break;
                if (!icp_instance.updateEstimate()) break;
                iterations_++;
                if (last_delta_norm_ < convergence_tol_) break;
            }

            return icp_instance;
        }

        inline ICPInstanceT& estimateTransformation(size_t max_iter, PointScalarT conv_tol) {
            max_iterations_ = max_iter;
            convergence_tol_ = conv_tol;
            return estimateTransformation();
        }

        inline const TransformT& getTransformation() const { return transform_; }

        inline const ICPInstanceT& getTransformation(TransformT& tform) const { tform = transform_; return *static_cast<ICPInstanceT*>(this); }

        inline ResidualT getResiduals() { return static_cast<ICPInstanceT*>(this)->computeResiduals(); }

        inline bool hasConverged() const { return last_delta_norm_ < convergence_tol_; }

    protected:
        size_t max_iterations_;
        size_t iterations_;
        PointScalarT convergence_tol_;
        PointScalarT last_delta_norm_;

        CorrespondenceSearchDirection corr_search_dir_;
        CorrValueT corr_max_distance_;
        double corr_inlier_fraction_;
        bool corr_require_reciprocal_;

        TransformT transform_init_;
        TransformT transform_;
    };
}
