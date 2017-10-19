#include <cilantro/iterative_closest_point.hpp>
#include <cilantro/transform_estimation.hpp>

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
    if (corr_type_ == CorrespondencesType::POINTS && !kd_tree_3d_) {
        kd_tree_3d_ = new KDTree<float,3,KDTreeDistanceAdaptors::L2>(*dst_points_);
    } else if (corr_type_ == CorrespondencesType::NORMALS && !kd_tree_3d_) {
        kd_tree_3d_ = new KDTree<float,3,KDTreeDistanceAdaptors::L2>(*dst_normals_);
    } else if (corr_type_ == CorrespondencesType::COLORS && !kd_tree_3d_) {
        kd_tree_3d_ = new KDTree<float,3,KDTreeDistanceAdaptors::L2>(*dst_colors_);
    } else if (corr_type_ == CorrespondencesType::POINTS_NORMALS && !kd_tree_6d_) {
        dst_data_points_6d_.resize(dst_points_->size());
        Eigen::Map<Eigen::Matrix<float,6,Eigen::Dynamic> > data_map((float *)dst_data_points_6d_.data(), 6, dst_data_points_6d_.size());
        data_map.topRows(3) = point_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_points_->data(), 3, dst_points_->size());
        data_map.bottomRows(3) = normal_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_normals_->data(), 3, dst_normals_->size());
        kd_tree_6d_ = new KDTree<float,6,KDTreeDistanceAdaptors::L2>(dst_data_points_6d_);
    } else if (corr_type_ == CorrespondencesType::POINTS_COLORS && !kd_tree_6d_) {
        dst_data_points_6d_.resize(dst_points_->size());
        Eigen::Map<Eigen::Matrix<float,6,Eigen::Dynamic> > data_map((float *)dst_data_points_6d_.data(), 6, dst_data_points_6d_.size());
        data_map.topRows(3) = point_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_points_->data(), 3, dst_points_->size());
        data_map.bottomRows(3) = color_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_colors_->data(), 3, dst_colors_->size());
        kd_tree_6d_ = new KDTree<float,6,KDTreeDistanceAdaptors::L2>(dst_data_points_6d_);
    } else if (corr_type_ == CorrespondencesType::NORMALS_COLORS && !kd_tree_6d_) {
        dst_data_points_6d_.resize(dst_points_->size());
        Eigen::Map<Eigen::Matrix<float,6,Eigen::Dynamic> > data_map((float *)dst_data_points_6d_.data(), 6, dst_data_points_6d_.size());
        data_map.topRows(3) = normal_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_normals_->data(), 3, dst_normals_->size());
        data_map.bottomRows(3) = color_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_colors_->data(), 3, dst_colors_->size());
        kd_tree_6d_ = new KDTree<float,6,KDTreeDistanceAdaptors::L2>(dst_data_points_6d_);
    } else if (corr_type_ == CorrespondencesType::POINTS_NORMALS_COLORS && !kd_tree_9d_) {
        dst_data_points_9d_.resize(dst_points_->size());
        Eigen::Map<Eigen::Matrix<float,9,Eigen::Dynamic> > data_map((float *)dst_data_points_9d_.data(), 9, dst_data_points_9d_.size());
        data_map.topRows(3) = point_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_points_->data(), 3, dst_points_->size());
        data_map.block(3,0,3,dst_data_points_9d_.size()) = normal_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_normals_->data(), 3, dst_normals_->size());
        data_map.bottomRows(3) = color_dist_weight_*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_colors_->data(), 3, dst_colors_->size());
        kd_tree_9d_ = new KDTree<float,9,KDTreeDistanceAdaptors::L2>(dst_data_points_9d_);
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
    if (corr_type == CorrespondencesType::POINTS) {
        return CorrespondencesType::POINTS;
    } else if (corr_type == CorrespondencesType::NORMALS && dst_normals_ && src_normals_) {
        return CorrespondencesType::NORMALS;
    } else if (corr_type == CorrespondencesType::COLORS && dst_colors_ && src_colors_) {
        return CorrespondencesType::COLORS;
    } else if (corr_type == CorrespondencesType::POINTS_NORMALS && dst_normals_ && src_normals_) {
        return CorrespondencesType::POINTS_NORMALS;
    } else if (corr_type == CorrespondencesType::POINTS_COLORS && dst_colors_ && src_colors_) {
        return CorrespondencesType::POINTS_COLORS;
    } else if (corr_type == CorrespondencesType::NORMALS_COLORS && dst_normals_ && src_normals_ && dst_colors_ && src_colors_) {
        return CorrespondencesType::NORMALS_COLORS;
    } else if (corr_type == CorrespondencesType::POINTS_NORMALS_COLORS && dst_normals_ && src_normals_ && dst_colors_ && src_colors_) {
        return CorrespondencesType::POINTS_NORMALS_COLORS;
    }
    return CorrespondencesType::POINTS;
}

void IterativeClosestPoint::init_params_() {
    point_dist_weight_ = 1.0f;
    normal_dist_weight_ = 1.0f;
    color_dist_weight_ = 1.0f;

    corr_dist_thres_ = 0.05f;
    convergence_tol_ = 1e-3f;
    max_iter_ = 15;
    point_to_plane_max_iter_ = 1;

    rot_mat_init_.setIdentity();
    t_vec_init_.setZero();
}

void IterativeClosestPoint::find_correspondences_(std::vector<size_t> &dst_ind, std::vector<size_t> &src_ind) {
    dst_ind.clear();
    src_ind.clear();

    size_t neighbor;
    float distance;

    float corr_thresh_squared = corr_dist_thres_*corr_dist_thres_;

    if (corr_type_ == CorrespondencesType::POINTS) {
#pragma omp parallel for shared (dst_ind, src_ind) private (neighbor, distance)
        for (size_t i = 0; i < src_points_trans_.size(); i++) {
            kd_tree_3d_->nearestNeighborSearch(src_points_trans_[i], neighbor, distance);
            if (distance >= corr_thresh_squared) continue;
#pragma omp critical
            if (metric_ != Metric::POINT_TO_PLANE || (*dst_normals_)[neighbor].allFinite()) {
                dst_ind.emplace_back(neighbor);
                src_ind.emplace_back(i);
            }
        }
    } else if (corr_type_ == CorrespondencesType::NORMALS) {
#pragma omp parallel for shared (dst_ind, src_ind) private (neighbor, distance)
        for (size_t i = 0; i < src_points_trans_.size(); i++) {
            kd_tree_3d_->nearestNeighborSearch(rot_mat_*(*src_normals_)[i], neighbor, distance);
            if (distance >= corr_thresh_squared) continue;
#pragma omp critical
            if (metric_ != Metric::POINT_TO_PLANE || (*dst_normals_)[neighbor].allFinite()) {
                dst_ind.emplace_back(neighbor);
                src_ind.emplace_back(i);
            }
        }
    } else if (corr_type_ == CorrespondencesType::COLORS) {
#pragma omp parallel for shared (dst_ind, src_ind) private (neighbor, distance)
        for (size_t i = 0; i < src_points_trans_.size(); i++) {
            kd_tree_3d_->nearestNeighborSearch((*src_colors_)[i], neighbor, distance);
            if (distance >= corr_thresh_squared) continue;
#pragma omp critical
            if (metric_ != Metric::POINT_TO_PLANE || (*dst_normals_)[neighbor].allFinite()) {
                dst_ind.emplace_back(neighbor);
                src_ind.emplace_back(i);
            }
        }
    } else if (corr_type_ == CorrespondencesType::POINTS_NORMALS) {
#pragma omp parallel for shared (dst_ind, src_ind) private (neighbor, distance)
        for (size_t i = 0; i < src_points_trans_.size(); i++) {
            Eigen::Matrix<float,6,1> query_pt;
            query_pt.head(3) = point_dist_weight_*src_points_trans_[i];
            query_pt.tail(3) = normal_dist_weight_*rot_mat_*(*src_normals_)[i];
            kd_tree_6d_->nearestNeighborSearch(query_pt, neighbor, distance);
            if (distance >= corr_thresh_squared) continue;
#pragma omp critical
            if (metric_ != Metric::POINT_TO_PLANE || (*dst_normals_)[neighbor].allFinite()) {
                dst_ind.emplace_back(neighbor);
                src_ind.emplace_back(i);
            }
        }
    } else if (corr_type_ == CorrespondencesType::POINTS_COLORS) {
#pragma omp parallel for shared (dst_ind, src_ind) private (neighbor, distance)
        for (size_t i = 0; i < src_points_trans_.size(); i++) {
            Eigen::Matrix<float,6,1> query_pt;
            query_pt.head(3) = point_dist_weight_*src_points_trans_[i];
            query_pt.tail(3) = color_dist_weight_*(*src_colors_)[i];
            kd_tree_6d_->nearestNeighborSearch(query_pt, neighbor, distance);
            if (distance >= corr_thresh_squared) continue;
#pragma omp critical
            if (metric_ != Metric::POINT_TO_PLANE || (*dst_normals_)[neighbor].allFinite()) {
                dst_ind.emplace_back(neighbor);
                src_ind.emplace_back(i);
            }
        }
    } else if (corr_type_ == CorrespondencesType::NORMALS_COLORS) {
#pragma omp parallel for shared (dst_ind, src_ind) private (neighbor, distance)
        for (size_t i = 0; i < src_points_trans_.size(); i++) {
            Eigen::Matrix<float,6,1> query_pt;
            query_pt.head(3) = normal_dist_weight_*rot_mat_*(*src_normals_)[i];
            query_pt.tail(3) = color_dist_weight_*(*src_colors_)[i];
            kd_tree_6d_->nearestNeighborSearch(query_pt, neighbor, distance);
            if (distance >= corr_thresh_squared) continue;
#pragma omp critical
            if (metric_ != Metric::POINT_TO_PLANE || (*dst_normals_)[neighbor].allFinite()) {
                dst_ind.emplace_back(neighbor);
                src_ind.emplace_back(i);
            }
        }
    } else if (corr_type_ == CorrespondencesType::POINTS_NORMALS_COLORS) {
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
                dst_ind.emplace_back(neighbor);
                src_ind.emplace_back(i);
            }
        }
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

    std::vector<size_t> dst_ind, src_ind;
    dst_ind.reserve(src_points_trans_.size());
    src_ind.reserve(src_points_trans_.size());

    iteration_count_ = 0;
    while (iteration_count_ < max_iter_) {
        // Transform src using current estimate
        src_t = (rot_mat_*src_p).colwise() + t_vec_;

        // Compute correspondences
        find_correspondences_(dst_ind, src_ind);

        iteration_count_++;

        if (dst_ind.empty()) {
            has_converged_ = false;
            break;
        }

        // Update estimated transformation
        if (metric_ == Metric::POINT_TO_PLANE) {
            estimateRigidTransformPointToPlane(*dst_points_, *dst_normals_, src_points_trans_, dst_ind, src_ind, rot_mat_iter, t_vec_iter, point_to_plane_max_iter_, convergence_tol_);
        } else if (metric_ == Metric::POINT_TO_POINT) {
            estimateRigidTransformPointToPoint(*dst_points_, src_points_trans_, dst_ind, src_ind, rot_mat_iter, t_vec_iter);
        } else {
            break;
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

void IterativeClosestPoint::compute_residuals_(const Metric &metric, std::vector<float> &residuals) {
    if (iteration_count_ == 0) estimate_transform_();

    size_t neighbor;
    float distance;

    residuals.resize(src_points_->size());
    if (corr_type_ == CorrespondencesType::POINTS) {
#pragma omp parallel for shared (residuals) private (neighbor, distance)
        for (size_t i = 0; i < src_points_trans_.size(); i++) {
            Eigen::Vector3f pt_trans = rot_mat_*(*src_points_)[i] + t_vec_;
            kd_tree_3d_->nearestNeighborSearch(pt_trans, neighbor, distance);
            if (metric == Metric::POINT_TO_PLANE) {
                const Eigen::Vector3f &dp = (*dst_points_)[neighbor];
                const Eigen::Vector3f &dn = (*dst_normals_)[neighbor];
                residuals[i] = std::abs(dn.dot(pt_trans - dp));
            } else if (metric == Metric::POINT_TO_POINT) {
                residuals[i] = std::sqrt(distance);
            }
        }
    } else if (corr_type_ == CorrespondencesType::NORMALS) {
#pragma omp parallel for shared (residuals) private (neighbor, distance)
        for (size_t i = 0; i < src_points_trans_.size(); i++) {
            Eigen::Vector3f pt_trans = rot_mat_*(*src_points_)[i] + t_vec_;
            kd_tree_3d_->nearestNeighborSearch(rot_mat_*(*src_normals_)[i], neighbor, distance);
            if (metric == Metric::POINT_TO_PLANE) {
                const Eigen::Vector3f &dp = (*dst_points_)[neighbor];
                const Eigen::Vector3f &dn = (*dst_normals_)[neighbor];
                residuals[i] = std::abs(dn.dot(pt_trans - dp));
            } else if (metric == Metric::POINT_TO_POINT) {
                residuals[i] = std::sqrt(distance);
            }
        }
    } else if (corr_type_ == CorrespondencesType::COLORS) {
#pragma omp parallel for shared (residuals) private (neighbor, distance)
        for (size_t i = 0; i < src_points_trans_.size(); i++) {
            Eigen::Vector3f pt_trans = rot_mat_*(*src_points_)[i] + t_vec_;
            kd_tree_3d_->nearestNeighborSearch((*src_colors_)[i], neighbor, distance);
            if (metric == Metric::POINT_TO_PLANE) {
                const Eigen::Vector3f &dp = (*dst_points_)[neighbor];
                const Eigen::Vector3f &dn = (*dst_normals_)[neighbor];
                residuals[i] = std::abs(dn.dot(pt_trans - dp));
            } else if (metric == Metric::POINT_TO_POINT) {
                residuals[i] = std::sqrt(distance);
            }
        }
    } else if (corr_type_ == CorrespondencesType::POINTS_NORMALS) {
#pragma omp parallel for shared (residuals) private (neighbor, distance)
        for (size_t i = 0; i < src_points_trans_.size(); i++) {
            Eigen::Vector3f pt_trans = rot_mat_*(*src_points_)[i] + t_vec_;
            Eigen::Matrix<float,6,1> query_pt;
            query_pt.head(3) = point_dist_weight_*pt_trans;
            query_pt.tail(3) = normal_dist_weight_*rot_mat_*(*src_normals_)[i];
            kd_tree_6d_->nearestNeighborSearch(query_pt, neighbor, distance);
            if (metric == Metric::POINT_TO_PLANE) {
                const Eigen::Vector3f& dp = (*dst_points_)[neighbor];
                const Eigen::Vector3f& dn = (*dst_normals_)[neighbor];
                residuals[i] = std::abs(dn.dot(pt_trans - dp));
            } else if (metric == Metric::POINT_TO_POINT) {
                residuals[i] = std::sqrt(distance);
            }
        }
    } else if (corr_type_ == CorrespondencesType::POINTS_COLORS) {
#pragma omp parallel for shared (residuals) private (neighbor, distance)
        for (size_t i = 0; i < src_points_trans_.size(); i++) {
            Eigen::Vector3f pt_trans = rot_mat_*(*src_points_)[i] + t_vec_;
            Eigen::Matrix<float,6,1> query_pt;
            query_pt.head(3) = point_dist_weight_*pt_trans;
            query_pt.tail(3) = color_dist_weight_*(*src_colors_)[i];
            kd_tree_6d_->nearestNeighborSearch(query_pt, neighbor, distance);
            if (metric == Metric::POINT_TO_PLANE) {
                const Eigen::Vector3f& dp = (*dst_points_)[neighbor];
                const Eigen::Vector3f& dn = (*dst_normals_)[neighbor];
                residuals[i] = std::abs(dn.dot(pt_trans - dp));
            } else if (metric == Metric::POINT_TO_POINT) {
                residuals[i] = std::sqrt(distance);
            }
        }
    } else if (corr_type_ == CorrespondencesType::NORMALS_COLORS) {
#pragma omp parallel for shared (residuals) private (neighbor, distance)
        for (size_t i = 0; i < src_points_trans_.size(); i++) {
            Eigen::Vector3f pt_trans = rot_mat_*(*src_points_)[i] + t_vec_;
            Eigen::Matrix<float,6,1> query_pt;
            query_pt.head(3) = normal_dist_weight_*rot_mat_*(*src_normals_)[i];
            query_pt.tail(3) = color_dist_weight_*(*src_colors_)[i];
            kd_tree_6d_->nearestNeighborSearch(query_pt, neighbor, distance);
            if (metric == Metric::POINT_TO_PLANE) {
                const Eigen::Vector3f& dp = (*dst_points_)[neighbor];
                const Eigen::Vector3f& dn = (*dst_normals_)[neighbor];
                residuals[i] = std::abs(dn.dot(pt_trans - dp));
            } else if (metric == Metric::POINT_TO_POINT) {
                residuals[i] = std::sqrt(distance);
            }
        }
    } else if (corr_type_ == CorrespondencesType::POINTS_NORMALS_COLORS) {
#pragma omp parallel for shared (residuals) private (neighbor, distance)
        for (size_t i = 0; i < src_points_trans_.size(); i++) {
            Eigen::Vector3f pt_trans = rot_mat_*(*src_points_)[i] + t_vec_;
            Eigen::Matrix<float,9,1> query_pt;
            query_pt.head(3) = point_dist_weight_*pt_trans;
            query_pt.segment(3,3) = normal_dist_weight_*rot_mat_*(*src_normals_)[i];
            query_pt.tail(3) = color_dist_weight_*(*src_colors_)[i];
            kd_tree_9d_->nearestNeighborSearch(query_pt, neighbor, distance);
            if (metric == Metric::POINT_TO_PLANE) {
                const Eigen::Vector3f& dp = (*dst_points_)[neighbor];
                const Eigen::Vector3f& dn = (*dst_normals_)[neighbor];
                residuals[i] = std::abs(dn.dot(pt_trans - dp));
            } else if (metric == Metric::POINT_TO_POINT) {
                residuals[i] = std::sqrt(distance);
            }
        }
    }
}
