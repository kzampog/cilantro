#pragma once

#include <vector>
#include <algorithm>
#include <random>
#include <Eigen/Dense>

namespace cilantro {
    // CRTP base class
    template <class ModelEstimatorT, class ModelT, typename ResidualScalarT, typename IndexT = size_t>
    class RandomSampleConsensusBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ModelT Model;
        typedef ResidualScalarT ResidualScalar;
        typedef std::vector<ResidualScalarT> ResidualVector;
        typedef IndexT Index;
        typedef std::vector<IndexT> IndexVector;

        RandomSampleConsensusBase(size_t sample_size,
                                  size_t inlier_count_thresh,
                                  size_t max_iter,
                                  ResidualScalar inlier_dist_thresh,
                                  bool re_estimate = true)
                : sample_size_(sample_size),
                  inlier_count_thresh_(inlier_count_thresh),
                  max_iter_(max_iter),
                  inlier_dist_thresh_(inlier_dist_thresh),
                  re_estimate_(re_estimate),
                  iteration_count_(0)
        {}

        inline size_t getSampleSize() const { return sample_size_; }

        inline ModelEstimatorT& setSampleSize(size_t sample_size) {
            sample_size_ = sample_size;
            return *static_cast<ModelEstimatorT*>(this);
        }

        inline size_t getTargetInlierCount() const { return inlier_count_thresh_; }

        inline ModelEstimatorT& setTargetInlierCount(size_t inlier_count_thres) {
            inlier_count_thresh_ = inlier_count_thres;
            return *static_cast<ModelEstimatorT*>(this);
        }

        inline size_t getMaxNumberOfIterations() const { return max_iter_; }

        inline ModelEstimatorT& setMaxNumberOfIterations(size_t max_iter) {
            max_iter_ = max_iter;
            return *static_cast<ModelEstimatorT*>(this);
        }

        inline ResidualScalar getMaxInlierResidual() const { return inlier_dist_thresh_; }

        inline ModelEstimatorT& setMaxInlierResidual(ResidualScalar inlier_dist_thresh) {
            inlier_dist_thresh_ = inlier_dist_thresh;
            return *static_cast<ModelEstimatorT*>(this);
        }

        inline bool getReEstimationStep() const { return re_estimate_; }

        inline ModelEstimatorT& setReEstimationStep(bool re_estimate) {
            re_estimate_ = re_estimate;
            return *static_cast<ModelEstimatorT*>(this);
        }

        ModelEstimatorT& estimate() {
            ModelEstimatorT& estimator = *static_cast<ModelEstimatorT*>(this);
            const size_t num_points = estimator.getDataPointsCount();
            if (num_points < sample_size_) sample_size_ = num_points;
            if (inlier_count_thresh_ > num_points) inlier_count_thresh_ = num_points;

            // Initialize index permutation and random engine
            IndexVector perm(num_points);
            for (size_t i = 0; i < num_points; i++) perm[i] = static_cast<IndexT>(i);
            std::mt19937 rng(std::random_device{}());

            // Random sample results
            Model curr_params;
            ResidualVector curr_residuals;
            IndexVector curr_inliers;

            iteration_count_ = 0;
            while (iteration_count_ < max_iter_) {
                // Pick a random sample
                IndexVector sample_ind(sample_size_);
                size_t prev_size = num_points;
                for (size_t i = 0; i < sample_size_; i++) {
                    std::uniform_int_distribution<size_t> dist(0, prev_size - 1);
                    size_t rand_ind = dist(rng);
                    sample_ind[i] = perm[rand_ind];
                    prev_size--;
                    std::swap(perm[rand_ind], perm[prev_size]);
                }

                // Fit model to sample and get its inliers
                estimator.estimateModel(sample_ind, curr_params);
                estimator.computeResiduals(curr_params, curr_residuals);
                curr_inliers.resize(num_points);
                size_t k = 0;
                for (size_t i = 0; i < num_points; i++) {
                    if (curr_residuals[i] <= inlier_dist_thresh_) curr_inliers[k++] = static_cast<IndexT>(i);
                }
                curr_inliers.resize(k);

                iteration_count_++;
                if (curr_inliers.size() < sample_size_) continue;

                // Update best found
                if (curr_inliers.size() > model_inliers_.size()) {
                    model_params_ = curr_params;
                    model_residuals_ = std::move(curr_residuals);
                    model_inliers_ = std::move(curr_inliers);
                }

                // Check if target inlier count was reached
                if (model_inliers_.size() >= inlier_count_thresh_) break;
            }

            // Re-estimate
            if (re_estimate_) {
                estimator.estimateModel(model_inliers_, model_params_);
                estimator.computeResiduals(model_params_, model_residuals_);
                model_inliers_.resize(num_points);
                size_t k = 0;
                for (size_t i = 0; i < num_points; i++){
                    if (model_residuals_[i] <= inlier_dist_thresh_) model_inliers_[k++] = static_cast<IndexT>(i);
                }
                model_inliers_.resize(k);
            }

            return estimator;
        }

        inline ModelEstimatorT& estimate(ResidualScalar max_residual,
                                         size_t target_inlier_count,
                                         size_t max_iter)
        {
            inlier_count_thresh_ = target_inlier_count;
            max_iter_ = max_iter;
            inlier_dist_thresh_ = max_residual;
            return estimate();
        }

        inline const ModelEstimatorT& getEstimationResults(Model &model_params,
                                                           ResidualVector &model_residuals,
                                                           IndexVector &model_inliers) const
        {
            model_params = model_params_;
            model_residuals = model_residuals_;
            model_inliers = model_inliers_;
            return *static_cast<ModelEstimatorT*>(this);
        }

        inline const Model& getModel() const { return model_params_; }

        inline const ModelEstimatorT& getModel(Model &model_params) const {
            model_params = model_params_;
            return *static_cast<ModelEstimatorT*>(this);
        }

        inline const ResidualVector& getModelResiduals() const { return model_residuals_; }

        inline const ModelEstimatorT& getModelResiduals(ResidualVector &model_residuals) const {
            model_residuals = model_residuals_;
            return *static_cast<ModelEstimatorT*>(this);
        }

        inline const IndexVector& getModelInliers() const { return model_inliers_; }

        inline const ModelEstimatorT& getModelInliers(IndexVector &model_inliers) const {
            model_inliers = model_inliers_;
            return *static_cast<ModelEstimatorT*>(this);
        }

        inline bool targetInlierCountAchieved() const { return model_inliers_.size() >= inlier_count_thresh_; }

        inline size_t getNumberOfPerformedIterations() const { return iteration_count_; }

        inline size_t getNumberOfInliers() const { return model_inliers_.size(); }

    protected:
        // Parameters
        size_t sample_size_;
        size_t inlier_count_thresh_;
        size_t max_iter_;
        ResidualScalar inlier_dist_thresh_;
        bool re_estimate_;

        // Object state and results
        size_t iteration_count_;
        Model model_params_;
        ResidualVector model_residuals_;
        IndexVector model_inliers_;
    };
}
