#pragma once

#include <cstddef>
#include <limits>

namespace cilantro {
    // CRTP base class
    template <class ICPInstanceT, class TransformT, class CorrespondenceSearchEngineT, class ResidualVectorT>
    class IterativeClosestPointBase {
    public:
        typedef TransformT Transformation;
        typedef typename TransformT::Scalar PointScalar;
        typedef CorrespondenceSearchEngineT CorrespondenceSearchEngine;
        typedef typename CorrespondenceSearchEngineT::SearchResult CorrespondenceSearchResults;
        typedef ResidualVectorT ResidualVector;

        IterativeClosestPointBase(CorrespondenceSearchEngine &corr_engine,
                                  size_t max_iter = 15,
                                  PointScalar conv_tol = (PointScalar)1e-5)
                : max_iterations_(max_iter),
                  iterations_(0),
                  convergence_tol_(conv_tol),
                  last_delta_norm_(std::numeric_limits<PointScalar>::infinity()),
                  correspondence_search_engine_(corr_engine)
        {}

        inline CorrespondenceSearchEngine& correspondenceSearchEngine() { return correspondence_search_engine_; }

        inline size_t getMaxNumberOfIterations() const { return max_iterations_; }

        inline ICPInstanceT& setMaxNumberOfIterations(size_t max_iter) {
            max_iterations_ = max_iter;
            return *static_cast<ICPInstanceT*>(this);
        }

        inline size_t getNumberOfPerformedIterations() const { return iterations_; }

        inline PointScalar getConvergenceTolerance() const { return convergence_tol_; }

        inline ICPInstanceT& setConvergenceTolerance(PointScalar conv_tol) {
            convergence_tol_ = conv_tol;
            return *static_cast<ICPInstanceT*>(this);
        }

        inline const Transformation& getInitialTransformation() const { return transform_init_; }

        inline ICPInstanceT& setInitialTransformation(const Transformation& tform_init) {
            transform_init_ = tform_init;
            return *static_cast<ICPInstanceT*>(this);
        }

        inline Transformation& initialTransformation() { return transform_init_; }

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

        inline const Transformation& getTransformation() const { return transform_; }

        inline const ICPInstanceT& getTransformation(Transformation& tform) const {
            tform = transform_;
            return *static_cast<const ICPInstanceT*>(this);
        }

        inline ResidualVector getResiduals() { return static_cast<ICPInstanceT*>(this)->computeResiduals(); }

        inline bool hasConverged() const { return last_delta_norm_ < convergence_tol_; }

    protected:
        size_t max_iterations_;
        size_t iterations_;
        PointScalar convergence_tol_;
        PointScalar last_delta_norm_;

        CorrespondenceSearchEngine& correspondence_search_engine_;
        CorrespondenceSearchResults correspondences_;

        Transformation transform_init_;
        Transformation transform_;
    };
}
