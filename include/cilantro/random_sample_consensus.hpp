#pragma once

#include <vector>
#include <algorithm>
#include <random>

template <class ModelEstimator, class ModelParamsType, class ResidualType>
class RandomSampleConsensus {
public:
    RandomSampleConsensus(size_t sample_size, size_t inlier_count_thresh, size_t max_iter, ResidualType inlier_dist_thresh, bool re_estimate)
            : sample_size_(sample_size),
              inlier_count_thres_(inlier_count_thresh),
              max_iter_(max_iter),
              inlier_dist_thresh_(inlier_dist_thresh),
              re_estimate_(re_estimate),
              iteration_count_(0)
    {}

    inline size_t getSampleSize() const { return sample_size_; }
    inline ModelEstimator& setSampleSize(size_t sample_size) {
        iteration_count_ = 0;
        sample_size_ = sample_size;
        return *static_cast<ModelEstimator*>(this);
    }

    inline size_t getTargetInlierCount() const { return inlier_count_thres_; }
    inline ModelEstimator& setTargetInlierCount(size_t inlier_count_thres) {
        iteration_count_ = 0;
        inlier_count_thres_ = inlier_count_thres;
        return *static_cast<ModelEstimator*>(this);
    }

    inline size_t getMaxNumberOfIterations() const { return max_iter_; }
    inline ModelEstimator& setMaxNumberOfIterations(size_t max_iter) {
        iteration_count_ = 0;
        max_iter_ = max_iter;
        return *static_cast<ModelEstimator*>(this);
    }

    inline ResidualType getMaxInlierResidual() const { return inlier_dist_thresh_; }
    inline ModelEstimator& setMaxInlierResidual(ResidualType inlier_dist_thresh) {
        iteration_count_ = 0;
        inlier_dist_thresh_ = inlier_dist_thresh;
        return *static_cast<ModelEstimator*>(this);
    }

    inline bool getReEstimationStep() const { return re_estimate_; }
    inline ModelEstimator& setReEstimationStep(bool re_estimate) {
        iteration_count_ = 0;
        re_estimate_ = re_estimate;
        return *static_cast<ModelEstimator*>(this);
    }

    inline ModelEstimator& getModel(ModelParamsType &model_params, std::vector<size_t> &model_inliers) {
        if (!targetInlierCountAchieved()) estimate_();
        model_params = model_params_;
        model_inliers = model_inliers_;
        return *static_cast<ModelEstimator*>(this);
    }

    inline const ModelParamsType& getModelParameters() {
        if (!targetInlierCountAchieved()) estimate_();
        return model_params_;
    }

    inline const std::vector<size_t>& getModelInliers() {
        if (!targetInlierCountAchieved()) estimate_();
        return model_inliers_;
    }

    inline bool targetInlierCountAchieved() const { return iteration_count_ > 0 && model_inliers_.size() >= inlier_count_thres_; }
    inline size_t getPerformedIterationsCount() const { return iteration_count_; }
    inline size_t getNumberOfInliers() const { return (iteration_count_ > 0) ? model_inliers_.size() : 0; }
    inline ModelEstimator& reset() { iteration_count_ = 0; model_inliers_.clear(); return *static_cast<ModelEstimator*>(this); }

private:
    // Parameters
    size_t sample_size_;
    size_t inlier_count_thres_;
    size_t max_iter_;
    ResidualType inlier_dist_thresh_;
    bool re_estimate_;

    // Object state and results
    size_t iteration_count_;
    ModelParamsType model_params_;
    std::vector<size_t> model_inliers_;

    void estimate_() {
        ModelEstimator * estimator = static_cast<ModelEstimator*>(this);
        size_t num_points = estimator->getDataPointsCount();
        if (num_points < sample_size_) return;

        std::random_device rd;
        std::mt19937 rng(rd());

        // Initialize random permutation
        std::vector<size_t> perm(num_points);
        for (size_t i = 0; i < num_points; i++) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        auto sample_start_it = perm.begin();

        ModelParamsType model_params_tmp;
        std::vector<size_t> model_inliers_tmp;
        std::vector<ResidualType> residuals_tmp;
        iteration_count_ = 0;
        while (iteration_count_ < max_iter_) {
            // Pick a random sample
            if (std::distance(sample_start_it, perm.end()) < sample_size_) {
                std::shuffle(perm.begin(), perm.end(), rng);
                sample_start_it = perm.begin();
            }
            std::vector<size_t> sample_ind(sample_start_it, sample_start_it + sample_size_);
            sample_start_it += sample_size_;

            // Fit model to sample and get its inliers
            estimator->estimateModelParameters(sample_ind, model_params_tmp);
            estimator->computeResiduals(model_params_tmp, residuals_tmp);
            model_inliers_tmp.resize(num_points);
            size_t k = 0;
            for (size_t i = 0; i < num_points; i++) {
                if (residuals_tmp[i] <= inlier_dist_thresh_) model_inliers_tmp[k++] = i;
            }
            model_inliers_tmp.resize(k);

            iteration_count_++;
            if (model_inliers_tmp.size() < sample_size_) continue;

            // Update best found
            if (model_inliers_tmp.size() > model_inliers_.size()) {
                // Re-estimate
                if (re_estimate_) {
                    estimator->estimateModelParameters(model_inliers_tmp, model_params_tmp);
                    estimator->computeResiduals(model_params_tmp, residuals_tmp);
                    model_inliers_tmp.resize(num_points);
                    k = 0;
                    for (size_t i = 0; i < num_points; i++){
                        if (residuals_tmp[i] <= inlier_dist_thresh_) model_inliers_tmp[k++] = i;
                    }
                    model_inliers_tmp.resize(k);
                }
                model_params_ = model_params_tmp;
                model_inliers_ = model_inliers_tmp;
            }

            // Check if target inlier count was reached
            if (model_inliers_.size() >= inlier_count_thres_) break;
        }
    }
};
