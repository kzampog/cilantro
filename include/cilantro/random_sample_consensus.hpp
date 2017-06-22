#pragma once

#include <vector>

#include <iostream>

template <class ModelEstimator, class ModelParamsType>
class RandomSampleConsensus {
public:
    RandomSampleConsensus(size_t sample_size, size_t inlier_count_thresh, size_t max_iter, float inlier_dist_thresh, bool re_estimate)
            : sample_size_(sample_size),
              inlier_count_thres_(inlier_count_thresh),
              max_iter_(max_iter),
              inlier_dist_thresh_(inlier_dist_thresh),
              re_estimate_(re_estimate)
    {}

    inline size_t getSampleSize() const { return sample_size_; }
    ModelEstimator& setSampleSize(size_t sample_size) {
        iteration_count_ = 0;
        sample_size_ = sample_size;
        return *static_cast<ModelEstimator*>(this);
    }

    inline size_t getTargetInlierCount() const { return inlier_count_thres_; }
    ModelEstimator& setTargetInlierCount(size_t inlier_count_thres) {
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

    inline float getMaxInlierResidual() const { return inlier_dist_thresh_; }
    ModelEstimator& setMaxInlierResidual(float inlier_dist_thresh) {
        iteration_count_ = 0;
        inlier_dist_thresh_ = inlier_dist_thresh;
        return *static_cast<ModelEstimator*>(this);
    }

    inline bool getReEstimationStep() const { return re_estimate_; }
    ModelEstimator& setReEstimationStep(bool re_estimate) {
        iteration_count_ = 0;
        re_estimate_ = re_estimate;
        return *static_cast<ModelEstimator*>(this);
    }

    inline ModelEstimator& getModel(ModelParamsType &model_params, std::vector<size_t> &model_inliers) {
        if (iteration_count_ == 0) compute_();
        model_params = model_params_;
        model_inliers = model_inliers_;
        return *static_cast<ModelEstimator*>(this);
    }

    inline const ModelParamsType& getModelParameters() {
        if (iteration_count_ == 0) compute_();
        return model_params_;
    }

    inline const std::vector<size_t>& getModelInliers() {
        if (iteration_count_ == 0) compute_();
        return model_inliers_;
    }

    inline bool targetInlierCountAchieved() const { return iteration_count_ > 0 && model_inliers_.size() >= inlier_count_thres_; }
    inline size_t getPerformedIterationsCount() const { return iteration_count_; }

private:
    // Parameters
    size_t sample_size_;
    size_t inlier_count_thres_;
    size_t max_iter_;
    float inlier_dist_thresh_;
    bool re_estimate_;

    // Object state and results
    size_t iteration_count_;
    ModelParamsType model_params_;
    std::vector<size_t> model_inliers_;

    void compute_() {
        // TODO

        iteration_count_ = 0;
        while (iteration_count_ < max_iter_) {

            iteration_count_++;
        }


    }
};
