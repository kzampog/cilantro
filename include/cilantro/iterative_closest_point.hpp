#pragma once

#include <cilantro/kd_tree.hpp>
#include <cilantro/point_cloud.hpp>

namespace cilantro {
    class IterativeClosestPoint {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        enum struct Metric {POINT_TO_POINT, POINT_TO_PLANE};
        enum struct CorrespondencesType {POINTS, NORMALS, COLORS, POINTS_NORMALS, POINTS_COLORS, NORMALS_COLORS, POINTS_NORMALS_COLORS};

        IterativeClosestPoint(const std::vector<Eigen::Vector3f> &dst_p, const std::vector<Eigen::Vector3f> &src_p);
        IterativeClosestPoint(const std::vector<Eigen::Vector3f> &dst_p, const std::vector<Eigen::Vector3f> &dst_n, const std::vector<Eigen::Vector3f> &src_p);
        IterativeClosestPoint(const PointCloud &dst, const PointCloud &src, const Metric &metric = Metric::POINT_TO_PLANE, const CorrespondencesType &corr_type = CorrespondencesType::POINTS);

        ~IterativeClosestPoint();

        inline Metric getMetric() const { return metric_; }
        inline IterativeClosestPoint& setMetric(const Metric &metric) {
            if (dst_normals_ != NULL && metric != metric_) {
                iteration_count_ = 0;
                metric_ = metric;
            }
            return *this;
        }

        inline CorrespondencesType getCorrespondencesType() const { return corr_type_; }
        inline IterativeClosestPoint& setCorrespondencesType(const CorrespondencesType &corr_type) {
            CorrespondencesType correct_corr_type = correct_correspondences_type_(corr_type);
            if (correct_corr_type != corr_type_) {
                delete_kd_trees_();
                iteration_count_ = 0;
                corr_type_ = correct_corr_type;
            }
            return *this;
        }

        inline float getCorrespondencePointWeight() const { return point_dist_weight_; }
        inline IterativeClosestPoint& setCorrespondencePointWeight(float point_dist_weight) {
            if (corr_type_ == CorrespondencesType::POINTS_NORMALS || corr_type_ == CorrespondencesType::POINTS_COLORS || corr_type_ == CorrespondencesType::POINTS_NORMALS_COLORS) {
                delete_kd_trees_();
                iteration_count_ = 0;
            }
            point_dist_weight_ = point_dist_weight;
            return *this;
        }

        inline float getCorrespondenceNormalWeight() const { return normal_dist_weight_; }
        inline IterativeClosestPoint& setCorrespondenceNormalWeight (float normal_dist_weight) {
            if (corr_type_ == CorrespondencesType::POINTS_NORMALS || corr_type_ == CorrespondencesType::NORMALS_COLORS || corr_type_ == CorrespondencesType::POINTS_NORMALS_COLORS) {
                delete_kd_trees_();
                iteration_count_ = 0;
            }
            normal_dist_weight_ = normal_dist_weight;
            return *this;
        }

        inline float getCorrespondenceColorWeight() const { return color_dist_weight_; }
        inline IterativeClosestPoint& setCorrespondenceColorWeight (float color_dist_weight) {
            if (corr_type_ == CorrespondencesType::POINTS_COLORS || corr_type_ == CorrespondencesType::NORMALS_COLORS || corr_type_ == CorrespondencesType::POINTS_NORMALS_COLORS) {
                delete_kd_trees_();
                iteration_count_ = 0;
            }
            color_dist_weight_ = color_dist_weight;
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

        inline void getInitialTransformation(Eigen::Ref<Eigen::Matrix3f> rot_mat_init, Eigen::Ref<Eigen::Vector3f> t_vec_init) const {
            rot_mat_init = rot_mat_init_;
            t_vec_init = t_vec_init_;
        }
        inline IterativeClosestPoint& setInitialTransformation(const Eigen::Ref<const Eigen::Matrix3f> &rot_mat, const Eigen::Ref<const Eigen::Vector3f> &t_vec) {
            iteration_count_ = 0;
            rot_mat_init_ = orthonormalize_rotation_(rot_mat);
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
            compute_residuals_(corr_type_, metric_, residuals);
            return *this;
        }

        inline const std::vector<float> getResiduals() {
            std::vector<float> residuals;
            compute_residuals_(corr_type_, metric_, residuals);
            return residuals;
        }

        inline IterativeClosestPoint& getResiduals(const CorrespondencesType &corr_type, const Metric &metric, std::vector<float> &residuals) {
            compute_residuals_(corr_type, metric, residuals);
            return *this;
        }

        inline const std::vector<float> getResiduals(const CorrespondencesType &corr_type, const Metric &metric) {
            std::vector<float> residuals;
            compute_residuals_(corr_type, metric, residuals);
            return residuals;
        }

        inline bool hasConverged() const { return iteration_count_ > 0 && has_converged_; }
        inline size_t getPerformedIterationsCount() const { return iteration_count_; }

    private:
        // Data pointers and parameters
        const std::vector<Eigen::Vector3f> *dst_points_;
        const std::vector<Eigen::Vector3f> *dst_normals_;
        const std::vector<Eigen::Vector3f> *dst_colors_;
        const std::vector<Eigen::Vector3f> *src_points_;
        const std::vector<Eigen::Vector3f> *src_normals_;
        const std::vector<Eigen::Vector3f> *src_colors_;

        KDTree<float,3,KDTreeDistanceAdaptors::L2> *kd_tree_3d_;
        KDTree<float,6,KDTreeDistanceAdaptors::L2> *kd_tree_6d_;
        KDTree<float,9,KDTreeDistanceAdaptors::L2> *kd_tree_9d_;

        CorrespondencesType corr_type_;
        Metric metric_;

        float point_dist_weight_;
        float normal_dist_weight_;
        float color_dist_weight_;

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

        std::vector<Eigen::Matrix<float,6,1> > dst_data_points_6d_;
        std::vector<Eigen::Matrix<float,9,1> > dst_data_points_9d_;

        void build_kd_trees_();
        void delete_kd_trees_();
        Eigen::Matrix3f orthonormalize_rotation_(const Eigen::Matrix3f &rot_mat) const;
        CorrespondencesType correct_correspondences_type_(const CorrespondencesType &corr_type) const;

        void init_params_();
        void find_correspondences_(std::vector<size_t> &dst_ind, std::vector<size_t> &src_ind);
        void estimate_transform_();
        void compute_residuals_(const CorrespondencesType &corr_type, const Metric &metric, std::vector<float> &residuals);
    };
}
