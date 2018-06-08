#pragma once

#include <cstddef>
#include <limits>

namespace cilantro {
    template <class MetricOptimizerT, class CorrespondenceSearchEngineT>
    class IterativeClosestPoint {
    public:
        typedef MetricOptimizerT MetricOptimizer;
        typedef typename MetricOptimizer::Transformation Transformation;
        typedef typename MetricOptimizer::ResidualVector ResidualVector;
        typedef CorrespondenceSearchEngineT CorrespondenceSearchEngine;
        typedef typename CorrespondenceSearchEngine::SearchResult CorrespondenceSearchResults;
        typedef typename Transformation::Scalar PointScalar;

        IterativeClosestPoint(MetricOptimizer &metric_optimizer,
                              CorrespondenceSearchEngine &corr_engine,
                              size_t max_iter = 15,
                              PointScalar conv_tol = (PointScalar)1e-5)
                : metric_optimizer_(metric_optimizer),
                  correspondence_search_engine_(corr_engine),
                  max_iterations_(max_iter),
                  iterations_(0),
                  convergence_tol_(conv_tol),
                  last_delta_norm_(std::numeric_limits<PointScalar>::infinity())
        {
            metric_optimizer_.initializeTransformation(transform_init_);
        }

        inline MetricOptimizer& metricOptimizer() { return metric_optimizer_; }

        inline CorrespondenceSearchEngine& correspondenceSearchEngine() { return correspondence_search_engine_; }

        inline size_t getMaxNumberOfIterations() const { return max_iterations_; }

        inline IterativeClosestPoint& setMaxNumberOfIterations(size_t max_iter) {
            max_iterations_ = max_iter;
            return *this;
        }

        inline size_t getNumberOfPerformedIterations() const { return iterations_; }

        inline PointScalar getConvergenceTolerance() const { return convergence_tol_; }

        inline IterativeClosestPoint& setConvergenceTolerance(PointScalar conv_tol) {
            convergence_tol_ = conv_tol;
            return *this;
        }

        inline const Transformation& getInitialTransformation() const { return transform_init_; }

        inline IterativeClosestPoint& setInitialTransformation(const Transformation& tform_init) {
            transform_init_ = tform_init;
            return *this;
        }

        inline Transformation& initialTransformation() { return transform_init_; }

        inline PointScalar getLastUpdateNorm() const { return last_delta_norm_; }

        // Main ICP loop
        IterativeClosestPoint& estimateTransformation() {
            transform_ = transform_init_;
            iterations_ = 0;
            last_delta_norm_ = std::numeric_limits<PointScalar>::infinity();
            while (iterations_ < max_iterations_) {
                correspondence_search_engine_.findCorrespondences(transform_, correspondences_);
                metric_optimizer_.refineTransformation(correspondences_, transform_, last_delta_norm_);
                iterations_++;
                if (last_delta_norm_ < convergence_tol_) break;
            }
            return *this;
        }

        inline IterativeClosestPoint& estimateTransformation(size_t max_iter, PointScalar conv_tol) {
            max_iterations_ = max_iter;
            convergence_tol_ = conv_tol;
            return estimateTransformation();
        }

        inline const CorrespondenceSearchResults& getCorrespondenceSearchResults() const { return correspondences_; }

        inline const Transformation& getTransformation() const { return transform_; }

        inline const IterativeClosestPoint& getTransformation(Transformation& tform) const {
            tform = transform_;
            return *this;
        }

        inline ResidualVector getResiduals() { return metric_optimizer_.computeResiduals(transform_); }

        inline bool hasConverged() const { return last_delta_norm_ < convergence_tol_; }

    protected:
        MetricOptimizerT& metric_optimizer_;
        Transformation transform_init_;
        Transformation transform_;

        CorrespondenceSearchEngine& correspondence_search_engine_;
        CorrespondenceSearchResults correspondences_;

        size_t max_iterations_;
        size_t iterations_;
        PointScalar convergence_tol_;
        PointScalar last_delta_norm_;
    };
}
