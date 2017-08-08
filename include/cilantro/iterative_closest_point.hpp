#pragma once

#include <cilantro/kd_tree.hpp>

class IterativeClosestPoint {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    enum struct Metric { POINT_TO_POINT, POINT_TO_PLANE };

    IterativeClosestPoint(const std::vector<Eigen::Vector3f> &dst_p, const std::vector<Eigen::Vector3f> &src_p);
    IterativeClosestPoint(const std::vector<Eigen::Vector3f> &dst_p, const std::vector<Eigen::Vector3f> &src_p, const KDTree &kd_tree);
    IterativeClosestPoint(const std::vector<Eigen::Vector3f> &dst_p, const std::vector<Eigen::Vector3f> &dst_n, const std::vector<Eigen::Vector3f> &src_p);
    IterativeClosestPoint(const std::vector<Eigen::Vector3f> &dst_p, const std::vector<Eigen::Vector3f> &dst_n, const std::vector<Eigen::Vector3f> &src_p, const KDTree &kd_tree);
    IterativeClosestPoint(const PointCloud &dst, const PointCloud &src, const Metric &metric = Metric::POINT_TO_POINT);
    IterativeClosestPoint(const PointCloud &dst, const PointCloud &src, const KDTree &kd_tree, const Metric &metric = Metric::POINT_TO_POINT);

    ~IterativeClosestPoint();

    inline Metric getMetric() const { return metric_; }
    inline IterativeClosestPoint& setMetric(const Metric metric) {
        if (dst_normals_ != NULL && metric != metric_) {
            iteration_count_ = 0;
            metric_ = metric;
        }
        return *this;
    }

    inline float getMaxCorrespondenceDistance() const { return corr_dist_thres_; }
    inline IterativeClosestPoint& setMaxCorrespondenceDistance(float max_dist) {
        iteration_count_ = 0;
        corr_dist_thres_ = max_dist;
        return *this;
    }

    inline size_t getMaxNumberOfIterations() const { return max_iter_; }
    inline IterativeClosestPoint& setMaxNumberOfIterations(size_t max_iter) {
        iteration_count_ = 0;
        max_iter_ = max_iter;
        return *this;
    }

    inline size_t getMaxNumberOfPointToPlaneIterations() const { return point_to_plane_max_iter_; }
    inline IterativeClosestPoint& setMaxNumberOfPointToPlaneIterations(size_t max_iter) {
        iteration_count_ = 0;
        point_to_plane_max_iter_ = max_iter;
        return *this;
    }

    inline float getConvergenceTolerance() const { return convergence_tol_; }
    inline IterativeClosestPoint& setConvergenceTolerance(float conv_tol) {
        iteration_count_ = 0;
        convergence_tol_ = conv_tol;
        return *this;
    }

    inline IterativeClosestPoint& setInitialTransformation(const Eigen::Ref<const Eigen::Matrix3f> &rot_mat, const Eigen::Ref<const Eigen::Vector3f> &t_vec) {
        iteration_count_ = 0;
        rot_mat_init_ = rot_mat;
        t_vec_init_ = t_vec;
        return *this;
    }

    inline IterativeClosestPoint& getTransformation(Eigen::Ref<Eigen::Matrix3f> rot_mat, Eigen::Ref<Eigen::Vector3f> t_vec) {
        if (iteration_count_ == 0) estimate_transform_();
        rot_mat = rot_mat_;
        t_vec = t_vec_;
        return *this;
    }

    inline IterativeClosestPoint& getResiduals(std::vector<float> &residuals) {
        compute_residuals_(metric_, residuals);
        return *this;
    }

    inline const std::vector<float> getResiduals() {
        std::vector<float> residuals;
        compute_residuals_(metric_, residuals);
        return residuals;
    }

    inline IterativeClosestPoint& getResiduals(const Metric &metric, std::vector<float> &residuals) {
        compute_residuals_(metric, residuals);
        return *this;
    }

    inline const std::vector<float> getResiduals(const Metric &metric) {
        std::vector<float> residuals;
        compute_residuals_(metric, residuals);
        return residuals;
    }

    inline bool hasConverged() const { return iteration_count_ > 0 && has_converged_; }
    inline size_t getPerformedIterationsCount() const { return iteration_count_; }

private:
    // Data pointers and parameters
    const std::vector<Eigen::Vector3f> *dst_points_;
    const std::vector<Eigen::Vector3f> *dst_normals_;
    const std::vector<Eigen::Vector3f> *src_points_;
    KDTree *kd_tree_ptr_;
    bool kd_tree_owned_;
    Metric metric_;

    float corr_dist_thres_;
    float convergence_tol_;
    size_t max_iter_;
    size_t point_to_plane_max_iter_;

    Eigen::Matrix3f rot_mat_init_;
    Eigen::Vector3f t_vec_init_;

    // Object state
    bool has_converged_;
    size_t iteration_count_;

    Eigen::Matrix3f rot_mat_;
    Eigen::Vector3f t_vec_;
    std::vector<Eigen::Vector3f> src_points_trans_;

    void init_params_();
    void estimate_transform_();
    void compute_residuals_(const Metric &metric, std::vector<float> &residuals);
};
