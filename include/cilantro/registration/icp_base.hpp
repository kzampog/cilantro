#pragma once

#include <Eigen/Dense>

namespace cilantro {
    // CRTP base class
    template <class ICPInstanceT, class TransformT, class CorrespondenceSearchEngineT, class ResidualVectorT>
    class IterativeClosestPointBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef TransformT Transform;

        typedef typename TransformT::Scalar Scalar;

        typedef typename TransformT::Scalar PointScalar;

        enum { Dim = Transform::Dim };

        typedef CorrespondenceSearchEngineT CorrespondenceSearchEngine;

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

        inline const CorrespondenceSearchEngine& correspondenceSearchEngine() const { return correspondence_search_engine_; }

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

        inline const Transform& getInitialTransform() const { return transform_init_; }

        inline ICPInstanceT& setInitialTransform(const Transform& tform_init) {
            transform_init_ = tform_init;
            return *static_cast<ICPInstanceT*>(this);
        }

        inline Transform& initialTransform() { return transform_init_; }

        inline PointScalar getLastUpdateNorm() const { return last_delta_norm_; }

        // Main ICP loop
        ICPInstanceT& estimate() {
            ICPInstanceT& icp_instance = *static_cast<ICPInstanceT*>(this);

            transform_ = transform_init_;
            iterations_ = 0;
            last_delta_norm_ = std::numeric_limits<PointScalar>::infinity();
            icp_instance.initializeComputation();

            while (iterations_ < max_iterations_) {
                // Update correspondences_
                icp_instance.updateCorrespondences();
                // Update transform_ and last_delta_norm_ based on correspondences_
                icp_instance.updateEstimate();

                iterations_++;
                if (last_delta_norm_ < convergence_tol_) break;
            }

            return icp_instance;
        }

        inline ICPInstanceT& estimate(size_t max_iter, PointScalar conv_tol) {
            max_iterations_ = max_iter;
            convergence_tol_ = conv_tol;
            return estimate();
        }

        inline const Transform& getTransform() const { return transform_; }

        inline const ICPInstanceT& getTransform(Transform& tform) const {
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

        Transform transform_init_;
        Transform transform_;

        // Default implementation
        inline void initializeComputation() {}
    };
}
