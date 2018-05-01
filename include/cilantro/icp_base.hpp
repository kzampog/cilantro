#pragma once

#include <Eigen/Dense>

namespace cilantro {
    // CRTP base class
    template <class ICPInstanceT, class TransformT, class CorrespondenceSearchEngineT, class ResidualVectorT>
    class IterativeClosestPointBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef typename TransformT::Scalar PointScalar;

        typedef typename CorrespondenceSearchEngineT::FeatureScalar FeatureScalar;

        typedef typename CorrespondenceSearchEngineT::SearchResult CorrespondenceSearchResults;

        IterativeClosestPointBase(CorrespondenceSearchEngineT &corr_engine,
                                  size_t max_iter = 15,
                                  PointScalar conv_tol = (PointScalar)1e-5)
                : max_iterations_(max_iter),
                  iterations_(0),
                  convergence_tol_(conv_tol),
                  last_delta_norm_(std::numeric_limits<PointScalar>::infinity()),
                  correspondence_search_engine_(corr_engine)
        {}

        inline CorrespondenceSearchEngineT& correspondenceSearchEngine() { return correspondence_search_engine_; }

        inline size_t getMaxNumberOfIterations() const { return max_iterations_; }

        inline ICPInstanceT& setMaxNumberOfIterations(size_t max_iter) { max_iterations_ = max_iter; return *static_cast<ICPInstanceT*>(this); }

        inline size_t getNumberOfPerformedIterations() const { return iterations_; }

        inline PointScalar getConvergenceTolerance() const { return convergence_tol_; }

        inline ICPInstanceT& setConvergenceTolerance(PointScalar conv_tol) { convergence_tol_ = conv_tol; return *static_cast<ICPInstanceT*>(this); }

        inline const TransformT& getInitialTransformation() const { return transform_init_; }

        inline ICPInstanceT& setInitialTransformation(const TransformT& tform_init) { transform_init_ = tform_init; return *static_cast<ICPInstanceT*>(this); }

        inline TransformT& initialTransformation() { return transform_init_; }

        inline PointScalar getLastUpdateNorm() const { return last_delta_norm_; }

        ICPInstanceT& estimateTransformation() {
            // Main ICP loop
            ICPInstanceT& icp_instance = *static_cast<ICPInstanceT*>(this);

            transform_ = transform_init_;
            icp_instance.initializeComputation();

            iterations_ = 0;
            last_delta_norm_ = std::numeric_limits<PointScalar>::infinity();
            while (iterations_ < max_iterations_) {
                if (!icp_instance.updateCorrespondences()) break;
                if (!icp_instance.updateEstimate()) break;
                iterations_++;
                if (last_delta_norm_ < convergence_tol_) break;
            }

            return icp_instance;
        }

        inline ICPInstanceT& estimateTransformation(size_t max_iter, PointScalar conv_tol) {
            max_iterations_ = max_iter;
            convergence_tol_ = conv_tol;
            return estimateTransformation();
        }

        inline const CorrespondenceSearchResults& getCorrespondenceSearchResults() const { return correspondences_; }

        inline const TransformT& getTransformation() const { return transform_; }

        inline const ICPInstanceT& getTransformation(TransformT& tform) const { tform = transform_; return *static_cast<const ICPInstanceT*>(this); }

        inline ResidualVectorT getResiduals() { return static_cast<ICPInstanceT*>(this)->computeResiduals(); }

        inline bool hasConverged() const { return last_delta_norm_ < convergence_tol_; }

    protected:
        size_t max_iterations_;
        size_t iterations_;
        PointScalar convergence_tol_;
        PointScalar last_delta_norm_;

        CorrespondenceSearchEngineT& correspondence_search_engine_;
        CorrespondenceSearchResults correspondences_;

        TransformT transform_init_;
        TransformT transform_;
    };
}
