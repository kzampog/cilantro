#include <cilantro/iterative_closest_point.hpp>
#include <cilantro/transform_estimation_utilities.hpp>

namespace cilantro {
    IterativeClosestPoint::IterativeClosestPoint(const std::vector<Eigen::Vector3f> &dst_p, const std::vector<Eigen::Vector3f> &src_p)
            : dst_points_(&dst_p),
              dst_normals_(NULL),
              dst_colors_(NULL),
              src_points_(&src_p),
              src_normals_(NULL),
              src_colors_(NULL),
              kd_tree_3d_(NULL),
              kd_tree_6d_(NULL),
              kd_tree_9d_(NULL),
              corr_type_(CorrespondencesType::POINTS),
              metric_(Metric::POINT_TO_POINT),
              has_converged_(false),
              iteration_count_(0),
              src_points_trans_(src_p.size())
    {
        init_params_();
    }

    IterativeClosestPoint::IterativeClosestPoint(const std::vector<Eigen::Vector3f> &dst_p, const std::vector<Eigen::Vector3f> &dst_n, const std::vector<Eigen::Vector3f> &src_p)
            : dst_points_(&dst_p),
              dst_normals_((dst_n.size() == dst_p.size()) ? &dst_n : NULL),
              dst_colors_(NULL),
              src_points_(&src_p),
              src_normals_(NULL),
              src_colors_(NULL),
              kd_tree_3d_(NULL),
              kd_tree_6d_(NULL),
              kd_tree_9d_(NULL),
              corr_type_(CorrespondencesType::POINTS),
              metric_((dst_n.size() == dst_p.size()) ? Metric::POINT_TO_PLANE : Metric::POINT_TO_POINT),
              has_converged_(false),
              iteration_count_(0),
              src_points_trans_(src_p.size())
    {
        init_params_();
    }

    IterativeClosestPoint::IterativeClosestPoint(const PointCloud &dst, const PointCloud &src, const Metric &metric, const CorrespondencesType &corr_type)
            : dst_points_(&dst.points),
              dst_normals_((dst.hasNormals()) ? &dst.normals : NULL),
              dst_colors_((dst.hasColors()) ? &dst.colors : NULL),
              src_points_(&src.points),
              src_normals_((src.hasNormals()) ? &src.normals : NULL),
              src_colors_((src.hasColors()) ? &src.colors : NULL),
              kd_tree_3d_(NULL),
              kd_tree_6d_(NULL),
              kd_tree_9d_(NULL),
              corr_type_(correct_correspondences_type_(corr_type)),
              metric_((dst.hasNormals()) ? metric : Metric::POINT_TO_POINT),
              has_converged_(false),
              iteration_count_(0),
              src_points_trans_(src.points.size())
    {
        init_params_();
    }

    IterativeClosestPoint::~IterativeClosestPoint() {
        delete_kd_trees_();
    }

    void IterativeClosestPoint::build_kd_trees_() {
        switch (corr_type_) {
            case CorrespondencesType::POINTS: {
                if (!kd_tree_3d_) kd_tree_3d_ = new KDTree<float,3,KDTreeDistanceAdaptors::L2>(*dst_points_);
                break;
            }
            case CorrespondencesType::NORMALS: {
                if (!kd_tree_3d_) kd_tree_3d_ = new KDTree<float,3,KDTreeDistanceAdaptors::L2>(*dst_normals_);
                break;
            }
            case CorrespondencesType::COLORS: {
                if (!kd_tree_3d_) kd_tree_3d_ = new KDTree<float,3,KDTreeDistanceAdaptors::L2>(*dst_colors_);
                break;
            }
            case CorrespondencesType::POINTS_NORMALS: {
                if (!kd_tree_6d_) {
                    dst_data_points_6d_.resize(dst_points_->size());
                    Eigen::Map<Eigen::Matrix<float,6,Eigen::Dynamic> > data_map((float *)dst_data_points_6d_.data(), 6, dst_data_points_6d_.size());
                    data_map.topRows(3) = point_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_points_->data(), 3, dst_points_->size());
                    data_map.bottomRows(3) = normal_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_normals_->data(), 3, dst_normals_->size());
                    kd_tree_6d_ = new KDTree<float,6,KDTreeDistanceAdaptors::L2>(dst_data_points_6d_);
                }
                break;
            }
            case CorrespondencesType::POINTS_COLORS: {
                if (!kd_tree_6d_) {
                    dst_data_points_6d_.resize(dst_points_->size());
                    Eigen::Map<Eigen::Matrix<float,6,Eigen::Dynamic> > data_map((float *)dst_data_points_6d_.data(), 6, dst_data_points_6d_.size());
                    data_map.topRows(3) = point_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_points_->data(), 3, dst_points_->size());
                    data_map.bottomRows(3) = color_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_colors_->data(), 3, dst_colors_->size());
                    kd_tree_6d_ = new KDTree<float,6,KDTreeDistanceAdaptors::L2>(dst_data_points_6d_);
                }
                break;
            }
            case CorrespondencesType::NORMALS_COLORS: {
                if (!kd_tree_6d_) {
                    dst_data_points_6d_.resize(dst_points_->size());
                    Eigen::Map<Eigen::Matrix<float,6,Eigen::Dynamic> > data_map((float *)dst_data_points_6d_.data(), 6, dst_data_points_6d_.size());
                    data_map.topRows(3) = normal_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_normals_->data(), 3, dst_normals_->size());
                    data_map.bottomRows(3) = color_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_colors_->data(), 3, dst_colors_->size());
                    kd_tree_6d_ = new KDTree<float,6,KDTreeDistanceAdaptors::L2>(dst_data_points_6d_);
                }
                break;
            }
            case CorrespondencesType::POINTS_NORMALS_COLORS: {
                if (!kd_tree_9d_) {
                    dst_data_points_9d_.resize(dst_points_->size());
                    Eigen::Map<Eigen::Matrix<float,9,Eigen::Dynamic> > data_map((float *)dst_data_points_9d_.data(), 9, dst_data_points_9d_.size());
                    data_map.topRows(3) = point_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_points_->data(), 3, dst_points_->size());
                    data_map.block(3,0,3,dst_data_points_9d_.size()) = normal_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_normals_->data(), 3, dst_normals_->size());
                    data_map.bottomRows(3) = color_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_colors_->data(), 3, dst_colors_->size());
                    kd_tree_9d_ = new KDTree<float,9,KDTreeDistanceAdaptors::L2>(dst_data_points_9d_);
                }
                break;
            }
        }
    }

    void IterativeClosestPoint::delete_kd_trees_() {
        delete kd_tree_3d_;
        delete kd_tree_6d_;
        delete kd_tree_9d_;
        kd_tree_3d_ = NULL;
        kd_tree_6d_ = NULL;
        kd_tree_9d_ = NULL;
        dst_data_points_6d_.clear();
        dst_data_points_9d_.clear();
    }

    Eigen::Matrix3f IterativeClosestPoint::orthonormalize_rotation_(const Eigen::Matrix3f &rot_mat) const {
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(rot_mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
        if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0.0f) {
            Eigen::Matrix3f U(svd.matrixU());
            U.col(2) *= -1.0f;
            return U*svd.matrixV().transpose();
        } else {
            return svd.matrixU()*svd.matrixV().transpose();
        }
    }

    IterativeClosestPoint::CorrespondencesType IterativeClosestPoint::correct_correspondences_type_(const CorrespondencesType &corr_type) const {
        switch (corr_type) {
            case CorrespondencesType::POINTS:
                return CorrespondencesType::POINTS;
            case CorrespondencesType::NORMALS:
                if (dst_normals_ && src_normals_) return CorrespondencesType::NORMALS;
                break;
            case CorrespondencesType::COLORS:
                if (dst_colors_ && src_colors_) return CorrespondencesType::COLORS;
                break;
            case CorrespondencesType::POINTS_NORMALS:
                if (dst_normals_ && src_normals_) return CorrespondencesType::POINTS_NORMALS;
                break;
            case CorrespondencesType::POINTS_COLORS:
                if (dst_colors_ && src_colors_) return CorrespondencesType::POINTS_COLORS;
                break;
            case CorrespondencesType::NORMALS_COLORS:
                if (dst_normals_ && src_normals_ && dst_colors_ && src_colors_) return CorrespondencesType::NORMALS_COLORS;
                break;
            case CorrespondencesType::POINTS_NORMALS_COLORS:
                if (dst_normals_ && src_normals_ && dst_colors_ && src_colors_) return CorrespondencesType::POINTS_NORMALS_COLORS;
                break;
        }
        return CorrespondencesType::POINTS;
    }

    void IterativeClosestPoint::init_params_() {
        point_dist_weight_ = 1.0f;
        normal_dist_weight_ = 1.0f;
        color_dist_weight_ = 1.0f;

        corr_dist_thres_ = 0.05f;
        corr_fraction_ = 1.0f;
        convergence_tol_ = 1e-3f;
        max_iter_ = 15;
        point_to_plane_max_iter_ = 1;

        rot_mat_init_.setIdentity();
        t_vec_init_.setZero();

        dst_ind_.reserve(src_points_->size());
        src_ind_.reserve(src_points_->size());
        dst_ind_all_.reserve(src_points_->size());
        src_ind_all_.reserve(src_points_->size());
        distances_all_.reserve(src_points_->size());
        ind_all_.reserve(src_points_->size());
    }

    void IterativeClosestPoint::find_correspondences_(std::vector<size_t>* &dst_ind, std::vector<size_t>* &src_ind) {
        float corr_thresh_squared = corr_dist_thres_*corr_dist_thres_;
        size_t neighbor;
        float distance;

        dst_ind_all_.clear();
        src_ind_all_.clear();
        distances_all_.clear();
        switch (corr_type_) {
            case CorrespondencesType::POINTS: {
#pragma omp parallel for shared (dst_ind, src_ind) private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.size(); i++) {
                    kd_tree_3d_->nearestNeighborSearch(src_points_trans_[i], neighbor, distance);
                    if (distance >= corr_thresh_squared) continue;
#pragma omp critical
                    if (metric_ != Metric::POINT_TO_PLANE || (*dst_normals_)[neighbor].allFinite()) {
                        dst_ind_all_.emplace_back(neighbor);
                        src_ind_all_.emplace_back(i);
                        distances_all_.emplace_back(distance);
                    }
                }
                break;
            }
            case CorrespondencesType::NORMALS: {
#pragma omp parallel for shared (dst_ind, src_ind) private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.size(); i++) {
                    kd_tree_3d_->nearestNeighborSearch(rot_mat_*(*src_normals_)[i], neighbor, distance);
                    if (distance >= corr_thresh_squared) continue;
#pragma omp critical
                    if (metric_ != Metric::POINT_TO_PLANE || (*dst_normals_)[neighbor].allFinite()) {
                        dst_ind_all_.emplace_back(neighbor);
                        src_ind_all_.emplace_back(i);
                        distances_all_.emplace_back(distance);
                    }
                }
                break;
            }
            case CorrespondencesType::COLORS: {
#pragma omp parallel for shared (dst_ind, src_ind) private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.size(); i++) {
                    kd_tree_3d_->nearestNeighborSearch((*src_colors_)[i], neighbor, distance);
                    if (distance >= corr_thresh_squared) continue;
#pragma omp critical
                    if (metric_ != Metric::POINT_TO_PLANE || (*dst_normals_)[neighbor].allFinite()) {
                        dst_ind_all_.emplace_back(neighbor);
                        src_ind_all_.emplace_back(i);
                        distances_all_.emplace_back(distance);
                    }
                }
                break;
            }
            case CorrespondencesType::POINTS_NORMALS: {
#pragma omp parallel for shared (dst_ind, src_ind) private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.size(); i++) {
                    Eigen::Matrix<float,6,1> query_pt;
                    query_pt.head(3) = point_dist_weight_*src_points_trans_[i];
                    query_pt.tail(3) = normal_dist_weight_*rot_mat_*(*src_normals_)[i];
                    kd_tree_6d_->nearestNeighborSearch(query_pt, neighbor, distance);
                    if (distance >= corr_thresh_squared) continue;
#pragma omp critical
                    if (metric_ != Metric::POINT_TO_PLANE || (*dst_normals_)[neighbor].allFinite()) {
                        dst_ind_all_.emplace_back(neighbor);
                        src_ind_all_.emplace_back(i);
                        distances_all_.emplace_back(distance);
                    }
                }
                break;
            }
            case CorrespondencesType::POINTS_COLORS: {
#pragma omp parallel for shared (dst_ind, src_ind) private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.size(); i++) {
                    Eigen::Matrix<float,6,1> query_pt;
                    query_pt.head(3) = point_dist_weight_*src_points_trans_[i];
                    query_pt.tail(3) = color_dist_weight_*(*src_colors_)[i];
                    kd_tree_6d_->nearestNeighborSearch(query_pt, neighbor, distance);
                    if (distance >= corr_thresh_squared) continue;
#pragma omp critical
                    if (metric_ != Metric::POINT_TO_PLANE || (*dst_normals_)[neighbor].allFinite()) {
                        dst_ind_all_.emplace_back(neighbor);
                        src_ind_all_.emplace_back(i);
                        distances_all_.emplace_back(distance);
                    }
                }
                break;
            }
            case CorrespondencesType::NORMALS_COLORS: {
#pragma omp parallel for shared (dst_ind, src_ind) private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.size(); i++) {
                    Eigen::Matrix<float,6,1> query_pt;
                    query_pt.head(3) = normal_dist_weight_*rot_mat_*(*src_normals_)[i];
                    query_pt.tail(3) = color_dist_weight_*(*src_colors_)[i];
                    kd_tree_6d_->nearestNeighborSearch(query_pt, neighbor, distance);
                    if (distance >= corr_thresh_squared) continue;
#pragma omp critical
                    if (metric_ != Metric::POINT_TO_PLANE || (*dst_normals_)[neighbor].allFinite()) {
                        dst_ind_all_.emplace_back(neighbor);
                        src_ind_all_.emplace_back(i);
                        distances_all_.emplace_back(distance);
                    }
                }
                break;
            }
            case CorrespondencesType::POINTS_NORMALS_COLORS: {
#pragma omp parallel for shared (dst_ind, src_ind) private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.size(); i++) {
                    Eigen::Matrix<float,9,1> query_pt;
                    query_pt.head(3) = point_dist_weight_*src_points_trans_[i];
                    query_pt.segment(3,3) = normal_dist_weight_*rot_mat_*(*src_normals_)[i];
                    query_pt.tail(3) = color_dist_weight_*(*src_colors_)[i];
                    kd_tree_9d_->nearestNeighborSearch(query_pt, neighbor, distance);
                    if (distance >= corr_thresh_squared) continue;
#pragma omp critical
                    if (metric_ != Metric::POINT_TO_PLANE || (*dst_normals_)[neighbor].allFinite()) {
                        dst_ind_all_.emplace_back(neighbor);
                        src_ind_all_.emplace_back(i);
                        distances_all_.emplace_back(distance);
                    }
                }
                break;
            }
        }

        if (corr_fraction_ > 0.0f && corr_fraction_ < 1.0f) {
            size_t num_corr = (size_t)std::llround(corr_fraction_*dst_ind_all_.size());
            num_corr = (metric_ == Metric::POINT_TO_PLANE) ? std::max(num_corr, (size_t)6) : std::max(num_corr, (size_t)3);
            num_corr = std::min(num_corr, dst_ind_all_.size());

            ind_all_.resize(dst_ind_all_.size());
            for (size_t i = 0; i < ind_all_.size(); i++) ind_all_[i] = i;
//            std::partial_sort(ind_all_.begin(), ind_all_.begin()+num_corr, ind_all_.end(), CorrespondenceComparator_(distances_all_));
            std::sort(ind_all_.begin(), ind_all_.end(), CorrespondenceComparator_(distances_all_));

            dst_ind_.resize(num_corr);
            src_ind_.resize(num_corr);
            for (size_t i = 0; i < num_corr; i++) {
                dst_ind_[i] = dst_ind_all_[ind_all_[i]];
                src_ind_[i] = src_ind_all_[ind_all_[i]];
            }

            dst_ind = &dst_ind_;
            src_ind = &src_ind_;

        } else {
            // Use all correspondences
            dst_ind = &dst_ind_all_;
            src_ind = &src_ind_all_;
        }
    }

    void IterativeClosestPoint::estimate_transform_() {
        build_kd_trees_();

        has_converged_ = false;

        rot_mat_ = rot_mat_init_;
        t_vec_ = t_vec_init_;

        Eigen::Matrix3f rot_mat_iter;
        Eigen::Vector3f t_vec_iter;
        Eigen::Matrix<float,6,1> delta;

        Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> > src_p((float *)(src_points_->data()),3,src_points_->size());
        Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> > src_t((float *)(src_points_trans_.data()),3,src_points_trans_.size());

        Eigen::Vector3f v_pi(M_PI, M_PI, M_PI);

        std::vector<size_t>* dst_ind;
        std::vector<size_t>* src_ind;

        iteration_count_ = 0;
        while (iteration_count_ < max_iter_) {
            // Transform src using current estimate
            src_t = (rot_mat_*src_p).colwise() + t_vec_;

            // Compute correspondences
            find_correspondences_(dst_ind, src_ind);

            iteration_count_++;

            if ((metric_ == Metric::POINT_TO_PLANE && dst_ind->size() < 6) || (metric_ == Metric::POINT_TO_POINT && dst_ind->size() < 3)) {
                has_converged_ = false;
                break;
            }

            // Update estimated transformation
            if (metric_ == Metric::POINT_TO_PLANE) {
                estimateRigidTransformPointToPlane(*dst_points_, *dst_normals_, src_points_trans_, *dst_ind, *src_ind, rot_mat_iter, t_vec_iter, point_to_plane_max_iter_, convergence_tol_);
            } else {
                estimateRigidTransformPointToPoint(*dst_points_, src_points_trans_, *dst_ind, *src_ind, rot_mat_iter, t_vec_iter);
            }

            rot_mat_ = rot_mat_iter*rot_mat_;
            t_vec_ = rot_mat_iter*t_vec_ + t_vec_iter;

            // Orthonormalize rotation
            rot_mat_ = orthonormalize_rotation_(rot_mat_);

            // Check for convergence
            Eigen::Vector3f tmp = rot_mat_iter.eulerAngles(2,1,0).cwiseAbs();
            tmp = tmp.cwiseMin((v_pi-tmp).cwiseAbs());
            delta.head(3) = tmp;
            delta.tail(3) = t_vec_iter;

            if (delta.norm() < convergence_tol_) {
                has_converged_ = true;
                break;
            }
        }
    }

    void IterativeClosestPoint::compute_residuals_(const CorrespondencesType &corr_type, const Metric &metric, std::vector<float> &residuals) {
        if (iteration_count_ == 0) estimate_transform_();

        CorrespondencesType req_corr_type = correct_correspondences_type_(corr_type);

        size_t neighbor;
        float distance;

        residuals.resize(src_points_->size());
        switch (req_corr_type) {
            case CorrespondencesType::POINTS: {
                KDTree<float,3,KDTreeDistanceAdaptors::L2> *kd_tree;
                if (req_corr_type == corr_type_) {
                    kd_tree = kd_tree_3d_;
                } else {
                    kd_tree = new KDTree<float,3,KDTreeDistanceAdaptors::L2>(*dst_points_);
                }
#pragma omp parallel for shared (residuals) private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.size(); i++) {
                    Eigen::Vector3f pt_trans = rot_mat_*(*src_points_)[i] + t_vec_;
                    kd_tree->nearestNeighborSearch(pt_trans, neighbor, distance);
                    if (metric == Metric::POINT_TO_PLANE) {
                        const Eigen::Vector3f &dp = (*dst_points_)[neighbor];
                        const Eigen::Vector3f &dn = (*dst_normals_)[neighbor];
                        residuals[i] = std::abs(dn.dot(pt_trans - dp));
                    } else {
                        residuals[i] = std::sqrt(distance);
                    }
                }
                if (req_corr_type != corr_type_) {
                    delete kd_tree;
                }
                break;
            }
            case CorrespondencesType::NORMALS: {
                KDTree<float,3,KDTreeDistanceAdaptors::L2> *kd_tree;
                if (req_corr_type == corr_type_) {
                    kd_tree = kd_tree_3d_;
                } else {
                    kd_tree = new KDTree<float,3,KDTreeDistanceAdaptors::L2>(*dst_normals_);
                }
#pragma omp parallel for shared (residuals) private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.size(); i++) {
                    Eigen::Vector3f pt_trans = rot_mat_*(*src_points_)[i] + t_vec_;
                    kd_tree->nearestNeighborSearch(rot_mat_*(*src_normals_)[i], neighbor, distance);
                    if (metric == Metric::POINT_TO_PLANE) {
                        const Eigen::Vector3f &dp = (*dst_points_)[neighbor];
                        const Eigen::Vector3f &dn = (*dst_normals_)[neighbor];
                        residuals[i] = std::abs(dn.dot(pt_trans - dp));
                    } else {
                        residuals[i] = std::sqrt(distance);
                    }
                }
                if (req_corr_type != corr_type_) {
                    delete kd_tree;
                }
                break;
            }
            case CorrespondencesType::COLORS: {
                KDTree<float,3,KDTreeDistanceAdaptors::L2> *kd_tree;
                if (req_corr_type == corr_type_) {
                    kd_tree = kd_tree_3d_;
                } else {
                    kd_tree = new KDTree<float,3,KDTreeDistanceAdaptors::L2>(*dst_colors_);
                }
#pragma omp parallel for shared (residuals) private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.size(); i++) {
                    Eigen::Vector3f pt_trans = rot_mat_*(*src_points_)[i] + t_vec_;
                    kd_tree->nearestNeighborSearch((*src_colors_)[i], neighbor, distance);
                    if (metric == Metric::POINT_TO_PLANE) {
                        const Eigen::Vector3f &dp = (*dst_points_)[neighbor];
                        const Eigen::Vector3f &dn = (*dst_normals_)[neighbor];
                        residuals[i] = std::abs(dn.dot(pt_trans - dp));
                    } else {
                        residuals[i] = std::sqrt(distance);
                    }
                }
                if (req_corr_type != corr_type_) {
                    delete kd_tree;
                }
                break;
            }
            case CorrespondencesType::POINTS_NORMALS: {
                KDTree<float,6,KDTreeDistanceAdaptors::L2> *kd_tree;
                std::vector<Eigen::Matrix<float,6,1> > data_holder;
                if (req_corr_type == corr_type_) {
                    kd_tree = kd_tree_6d_;
                } else {
                    data_holder.resize(dst_points_->size());
                    Eigen::Map<Eigen::Matrix<float,6,Eigen::Dynamic> > data_map((float *)data_holder.data(), 6, data_holder.size());
                    data_map.topRows(3) = point_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_points_->data(), 3, dst_points_->size());
                    data_map.bottomRows(3) = normal_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_normals_->data(), 3, dst_normals_->size());
                    kd_tree = new KDTree<float,6,KDTreeDistanceAdaptors::L2>(data_holder);
                }
#pragma omp parallel for shared (residuals) private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.size(); i++) {
                    Eigen::Vector3f pt_trans = rot_mat_*(*src_points_)[i] + t_vec_;
                    Eigen::Matrix<float,6,1> query_pt;
                    query_pt.head(3) = point_dist_weight_*pt_trans;
                    query_pt.tail(3) = normal_dist_weight_*rot_mat_*(*src_normals_)[i];
                    kd_tree->nearestNeighborSearch(query_pt, neighbor, distance);
                    if (metric == Metric::POINT_TO_PLANE) {
                        const Eigen::Vector3f& dp = (*dst_points_)[neighbor];
                        const Eigen::Vector3f& dn = (*dst_normals_)[neighbor];
                        residuals[i] = std::abs(dn.dot(pt_trans - dp));
                    } else {
                        residuals[i] = std::sqrt(distance);
                    }
                }
                if (req_corr_type != corr_type_) {
                    delete kd_tree;
                }
                break;
            }
            case CorrespondencesType::POINTS_COLORS: {
                KDTree<float,6,KDTreeDistanceAdaptors::L2> *kd_tree;
                std::vector<Eigen::Matrix<float,6,1> > data_holder;
                if (req_corr_type == corr_type_) {
                    kd_tree = kd_tree_6d_;
                } else {
                    data_holder.resize(dst_points_->size());
                    Eigen::Map<Eigen::Matrix<float,6,Eigen::Dynamic> > data_map((float *)data_holder.data(), 6, data_holder.size());
                    data_map.topRows(3) = point_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_points_->data(), 3, dst_points_->size());
                    data_map.bottomRows(3) = color_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_colors_->data(), 3, dst_colors_->size());
                    kd_tree = new KDTree<float,6,KDTreeDistanceAdaptors::L2>(data_holder);
                }
#pragma omp parallel for shared (residuals) private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.size(); i++) {
                    Eigen::Vector3f pt_trans = rot_mat_*(*src_points_)[i] + t_vec_;
                    Eigen::Matrix<float,6,1> query_pt;
                    query_pt.head(3) = point_dist_weight_*pt_trans;
                    query_pt.tail(3) = color_dist_weight_*(*src_colors_)[i];
                    kd_tree->nearestNeighborSearch(query_pt, neighbor, distance);
                    if (metric == Metric::POINT_TO_PLANE) {
                        const Eigen::Vector3f& dp = (*dst_points_)[neighbor];
                        const Eigen::Vector3f& dn = (*dst_normals_)[neighbor];
                        residuals[i] = std::abs(dn.dot(pt_trans - dp));
                    } else {
                        residuals[i] = std::sqrt(distance);
                    }
                }
                if (req_corr_type != corr_type_) {
                    delete kd_tree;
                }
                break;
            }
            case CorrespondencesType::NORMALS_COLORS: {
                KDTree<float,6,KDTreeDistanceAdaptors::L2> *kd_tree;
                std::vector<Eigen::Matrix<float,6,1> > data_holder;
                if (req_corr_type == corr_type_) {
                    kd_tree = kd_tree_6d_;
                } else {
                    data_holder.resize(dst_points_->size());
                    Eigen::Map<Eigen::Matrix<float,6,Eigen::Dynamic> > data_map((float *)data_holder.data(), 6, data_holder.size());
                    data_map.topRows(3) = normal_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_normals_->data(), 3, dst_normals_->size());
                    data_map.bottomRows(3) = color_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_colors_->data(), 3, dst_colors_->size());
                    kd_tree = new KDTree<float,6,KDTreeDistanceAdaptors::L2>(data_holder);
                }
#pragma omp parallel for shared (residuals) private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.size(); i++) {
                    Eigen::Vector3f pt_trans = rot_mat_*(*src_points_)[i] + t_vec_;
                    Eigen::Matrix<float,6,1> query_pt;
                    query_pt.head(3) = normal_dist_weight_*rot_mat_*(*src_normals_)[i];
                    query_pt.tail(3) = color_dist_weight_*(*src_colors_)[i];
                    kd_tree->nearestNeighborSearch(query_pt, neighbor, distance);
                    if (metric == Metric::POINT_TO_PLANE) {
                        const Eigen::Vector3f& dp = (*dst_points_)[neighbor];
                        const Eigen::Vector3f& dn = (*dst_normals_)[neighbor];
                        residuals[i] = std::abs(dn.dot(pt_trans - dp));
                    } else {
                        residuals[i] = std::sqrt(distance);
                    }
                }
                if (req_corr_type != corr_type_) {
                    delete kd_tree;
                }
                break;
            }
            case CorrespondencesType::POINTS_NORMALS_COLORS: {
                KDTree<float,9,KDTreeDistanceAdaptors::L2> *kd_tree;
                std::vector<Eigen::Matrix<float,9,1> > data_holder;
                if (req_corr_type == corr_type_) {
                    kd_tree = kd_tree_9d_;
                } else {
                    data_holder.resize(dst_points_->size());
                    Eigen::Map<Eigen::Matrix<float,9,Eigen::Dynamic> > data_map((float *)data_holder.data(), 9, data_holder.size());
                    data_map.topRows(3) = point_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_points_->data(), 3, dst_points_->size());
                    data_map.block(3,0,3,dst_data_points_9d_.size()) = normal_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_normals_->data(), 3, dst_normals_->size());
                    data_map.bottomRows(3) = color_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_colors_->data(), 3, dst_colors_->size());
                    kd_tree = new KDTree<float,9,KDTreeDistanceAdaptors::L2>(data_holder);
                }
#pragma omp parallel for shared (residuals) private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.size(); i++) {
                    Eigen::Vector3f pt_trans = rot_mat_*(*src_points_)[i] + t_vec_;
                    Eigen::Matrix<float,9,1> query_pt;
                    query_pt.head(3) = point_dist_weight_*pt_trans;
                    query_pt.segment(3,3) = normal_dist_weight_*rot_mat_*(*src_normals_)[i];
                    query_pt.tail(3) = color_dist_weight_*(*src_colors_)[i];
                    kd_tree->nearestNeighborSearch(query_pt, neighbor, distance);
                    if (metric == Metric::POINT_TO_PLANE) {
                        const Eigen::Vector3f& dp = (*dst_points_)[neighbor];
                        const Eigen::Vector3f& dn = (*dst_normals_)[neighbor];
                        residuals[i] = std::abs(dn.dot(pt_trans - dp));
                    } else {
                        residuals[i] = std::sqrt(distance);
                    }
                }
                if (req_corr_type != corr_type_) {
                    delete kd_tree;
                }
                break;
            }
        }
    }
}
