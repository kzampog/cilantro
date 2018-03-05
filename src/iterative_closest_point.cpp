#include <cilantro/iterative_closest_point.hpp>
#include <cilantro/registration.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cilantro {
    IterativeClosestPoint::IterativeClosestPoint(const ConstVectorSetMatrixMap<float,3> &dst_p, const ConstVectorSetMatrixMap<float,3> &src_p)
            : dst_points_(dst_p),
              dst_normals_(NULL),
              dst_colors_(NULL),
              src_points_(src_p),
              src_normals_(NULL),
              src_colors_(NULL),
              kd_tree_3d_(NULL),
              kd_tree_6d_(NULL),
              kd_tree_9d_(NULL),
              corr_type_(CorrespondencesType::POINTS),
              metric_(Metric::POINT_TO_POINT),
              has_converged_(false),
              iteration_count_(0)
    {
        init_params_();
    }

    IterativeClosestPoint::IterativeClosestPoint(const ConstVectorSetMatrixMap<float,3> &dst_p, const ConstVectorSetMatrixMap<float,3> &dst_n, const ConstVectorSetMatrixMap<float,3> &src_p)
            : dst_points_(dst_p),
              dst_normals_((dst_n.cols() == dst_p.cols()) ? dst_n : ConstVectorSetMatrixMap<float,3>(NULL,0)),
              dst_colors_(NULL),
              src_points_(src_p),
              src_normals_(NULL),
              src_colors_(NULL),
              kd_tree_3d_(NULL),
              kd_tree_6d_(NULL),
              kd_tree_9d_(NULL),
              corr_type_(CorrespondencesType::POINTS),
              metric_((dst_n.cols() == dst_p.cols()) ? Metric::POINT_TO_PLANE : Metric::POINT_TO_POINT),
              has_converged_(false),
              iteration_count_(0)
    {
        init_params_();
    }

    IterativeClosestPoint::IterativeClosestPoint(const PointCloud<float,3> &dst, const PointCloud<float,3> &src, const Metric &metric, const CorrespondencesType &corr_type)
            : dst_points_(dst.points),
              dst_normals_((dst.hasNormals()) ? ConstVectorSetMatrixMap<float,3>(dst.normals) : ConstVectorSetMatrixMap<float,3>(NULL,0)),
              dst_colors_((dst.hasColors()) ? ConstVectorSetMatrixMap<float,3>(dst.colors) : ConstVectorSetMatrixMap<float,3>(NULL,0)),
              src_points_(src.points),
              src_normals_((src.hasNormals()) ? ConstVectorSetMatrixMap<float,3>(src.normals) : ConstVectorSetMatrixMap<float,3>(NULL,0)),
              src_colors_((src.hasColors()) ? ConstVectorSetMatrixMap<float,3>(src.colors) : ConstVectorSetMatrixMap<float,3>(NULL,0)),
              kd_tree_3d_(NULL),
              kd_tree_6d_(NULL),
              kd_tree_9d_(NULL),
              corr_type_(correct_correspondences_type_(corr_type)),
              metric_((dst.hasNormals()) ? metric : Metric::POINT_TO_POINT),
              has_converged_(false),
              iteration_count_(0)
    {
        init_params_();
    }

    IterativeClosestPoint::~IterativeClosestPoint() {
        delete_kd_trees_();
    }

    void IterativeClosestPoint::build_kd_trees_() {
        switch (corr_type_) {
            case CorrespondencesType::POINTS: {
                if (!kd_tree_3d_) kd_tree_3d_ = new KDTree<float,3,KDTreeDistanceAdaptors::L2>(dst_points_);
                break;
            }
            case CorrespondencesType::NORMALS: {
                if (!kd_tree_3d_) kd_tree_3d_ = new KDTree<float,3,KDTreeDistanceAdaptors::L2>(dst_normals_);
                break;
            }
            case CorrespondencesType::COLORS: {
                if (!kd_tree_3d_) kd_tree_3d_ = new KDTree<float,3,KDTreeDistanceAdaptors::L2>(dst_colors_);
                break;
            }
            case CorrespondencesType::POINTS_NORMALS: {
                if (!kd_tree_6d_) {
                    dst_data_points_6d_.resize(Eigen::NoChange, dst_points_.cols());
                    dst_data_points_6d_.topRows(3) = point_dist_weight_*dst_points_;
                    dst_data_points_6d_.bottomRows(3) = normal_dist_weight_*dst_normals_;
                    kd_tree_6d_ = new KDTree<float,6,KDTreeDistanceAdaptors::L2>(dst_data_points_6d_);
                }
                break;
            }
            case CorrespondencesType::POINTS_COLORS: {
                if (!kd_tree_6d_) {
                    dst_data_points_6d_.resize(Eigen::NoChange, dst_points_.cols());
                    dst_data_points_6d_.topRows(3) = point_dist_weight_*dst_points_;
                    dst_data_points_6d_.bottomRows(3) = color_dist_weight_*dst_colors_;
                    kd_tree_6d_ = new KDTree<float,6,KDTreeDistanceAdaptors::L2>(dst_data_points_6d_);
                }
                break;
            }
            case CorrespondencesType::NORMALS_COLORS: {
                if (!kd_tree_6d_) {
                    dst_data_points_6d_.resize(Eigen::NoChange, dst_points_.cols());
                    dst_data_points_6d_.topRows(3) = normal_dist_weight_*dst_normals_;
                    dst_data_points_6d_.bottomRows(3) = color_dist_weight_*dst_colors_;
                    kd_tree_6d_ = new KDTree<float,6,KDTreeDistanceAdaptors::L2>(dst_data_points_6d_);
                }
                break;
            }
            case CorrespondencesType::POINTS_NORMALS_COLORS: {
                if (!kd_tree_9d_) {
                    dst_data_points_9d_.resize(Eigen::NoChange, dst_points_.cols());
                    dst_data_points_9d_.topRows(3) = point_dist_weight_*dst_points_;
                    dst_data_points_9d_.block(3,0,3,dst_data_points_9d_.cols()) = normal_dist_weight_*dst_normals_;
                    dst_data_points_9d_.bottomRows(3) = color_dist_weight_*dst_colors_;
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
        dst_data_points_6d_.resize(Eigen::NoChange, 0);
        dst_data_points_9d_.resize(Eigen::NoChange, 0);
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
                if (dst_normals_.cols() > 0 && src_normals_.cols() > 0) return CorrespondencesType::NORMALS;
                break;
            case CorrespondencesType::COLORS:
                if (dst_colors_.cols() > 0 && src_colors_.cols() > 0) return CorrespondencesType::COLORS;
                break;
            case CorrespondencesType::POINTS_NORMALS:
                if (dst_normals_.cols() > 0 && src_normals_.cols() > 0) return CorrespondencesType::POINTS_NORMALS;
                break;
            case CorrespondencesType::POINTS_COLORS:
                if (dst_colors_.cols() > 0 && src_colors_.cols() > 0) return CorrespondencesType::POINTS_COLORS;
                break;
            case CorrespondencesType::NORMALS_COLORS:
                if (dst_normals_.cols() > 0 && src_normals_.cols() > 0 && dst_colors_.cols() > 0 && src_colors_.cols() > 0) return CorrespondencesType::NORMALS_COLORS;
                break;
            case CorrespondencesType::POINTS_NORMALS_COLORS:
                if (dst_normals_.cols() > 0 && src_normals_.cols() > 0 && dst_colors_.cols() > 0 && src_colors_.cols() > 0) return CorrespondencesType::POINTS_NORMALS_COLORS;
                break;
        }
        return CorrespondencesType::POINTS;
    }

    void IterativeClosestPoint::init_params_() {
        point_dist_weight_ = 1.0f;
        normal_dist_weight_ = 1.0f;
        color_dist_weight_ = 1.0f;

        point_to_point_weight_ = 0.01f;
        point_to_plane_weight_ = 1.0f;

        corr_dist_thres_ = 0.05f;
        corr_fraction_ = 1.0f;
        convergence_tol_ = 1e-3f;
        max_iter_ = 15;
        max_estimation_iter_ = 1;

        rot_mat_init_.setIdentity();
        t_vec_init_.setZero();

        correspondences_.reserve(src_points_.cols());
    }

    void IterativeClosestPoint::find_correspondences_() {
        float corr_dist_thres_squared = corr_dist_thres_*corr_dist_thres_;
        size_t neighbor;
        float distance;

        correspondences_.clear();

        switch (corr_type_) {
            case CorrespondencesType::POINTS: {
#pragma omp parallel for private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.cols(); i++) {
                    kd_tree_3d_->nearestNeighborSearch(src_points_trans_.col(i), neighbor, distance);
                    if (distance < corr_dist_thres_squared) {
#pragma omp critical
                        {
                            correspondences_.emplace_back(neighbor, i, distance);
                        }
                    }
                }
                break;
            }
            case CorrespondencesType::NORMALS: {
#pragma omp parallel for private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.cols(); i++) {
                    kd_tree_3d_->nearestNeighborSearch(rot_mat_*src_normals_.col(i), neighbor, distance);
                    if (distance < corr_dist_thres_squared) {
#pragma omp critical
                        {
                            correspondences_.emplace_back(neighbor, i, distance);
                        }
                    }
                }
                break;
            }
            case CorrespondencesType::COLORS: {
#pragma omp parallel for private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.cols(); i++) {
                    kd_tree_3d_->nearestNeighborSearch(src_colors_.col(i), neighbor, distance);
                    if (distance < corr_dist_thres_squared) {
#pragma omp critical
                        {
                            correspondences_.emplace_back(neighbor, i, distance);
                        }
                    }
                }
                break;
            }
            case CorrespondencesType::POINTS_NORMALS: {
#pragma omp parallel for private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.cols(); i++) {
                    Eigen::Matrix<float,6,1> query_pt;
                    query_pt.head(3) = point_dist_weight_*src_points_trans_.col(i);
                    query_pt.tail(3) = normal_dist_weight_*rot_mat_*src_normals_.col(i);
                    kd_tree_6d_->nearestNeighborSearch(query_pt, neighbor, distance);
                    if (distance < corr_dist_thres_squared) {
#pragma omp critical
                        {
                            correspondences_.emplace_back(neighbor, i, distance);
                        }
                    }
                }
                break;
            }
            case CorrespondencesType::POINTS_COLORS: {
#pragma omp parallel for private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.cols(); i++) {
                    Eigen::Matrix<float,6,1> query_pt;
                    query_pt.head(3) = point_dist_weight_*src_points_trans_.col(i);
                    query_pt.tail(3) = color_dist_weight_*src_colors_.col(i);
                    kd_tree_6d_->nearestNeighborSearch(query_pt, neighbor, distance);
                    if (distance < corr_dist_thres_squared) {
#pragma omp critical
                        {
                            correspondences_.emplace_back(neighbor, i, distance);
                        }
                    }
                }
                break;
            }
            case CorrespondencesType::NORMALS_COLORS: {
#pragma omp parallel for private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.cols(); i++) {
                    Eigen::Matrix<float,6,1> query_pt;
                    query_pt.head(3) = normal_dist_weight_*rot_mat_*src_normals_.col(i);
                    query_pt.tail(3) = color_dist_weight_*src_colors_.col(i);
                    kd_tree_6d_->nearestNeighborSearch(query_pt, neighbor, distance);
                    if (distance < corr_dist_thres_squared) {
#pragma omp critical
                        {
                            correspondences_.emplace_back(neighbor, i, distance);
                        }
                    }
                }
                break;
            }
            case CorrespondencesType::POINTS_NORMALS_COLORS: {
#pragma omp parallel for private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.cols(); i++) {
                    Eigen::Matrix<float,9,1> query_pt;
                    query_pt.head(3) = point_dist_weight_*src_points_trans_.col(i);
                    query_pt.segment(3,3) = normal_dist_weight_*rot_mat_*src_normals_.col(i);
                    query_pt.tail(3) = color_dist_weight_*src_colors_.col(i);
                    kd_tree_9d_->nearestNeighborSearch(query_pt, neighbor, distance);
                    if (distance < corr_dist_thres_squared) {
#pragma omp critical
                        {
                            correspondences_.emplace_back(neighbor, i, distance);
                        }
                    }
                }
                break;
            }
        }

        filterCorrespondencesFraction(correspondences_, corr_fraction_);
    }

    void IterativeClosestPoint::estimate_transform_() {
        build_kd_trees_();

        has_converged_ = false;

        rot_mat_ = rot_mat_init_;
        t_vec_ = t_vec_init_;

        RigidTransformation<float,3> tform_iter;

        Eigen::Matrix<float,6,1> delta;

        Eigen::Vector3f v_pi(M_PI, M_PI, M_PI);

        iteration_count_ = 0;
        while (iteration_count_ < max_iter_) {
            // Transform src using current estimate
            src_points_trans_ = (rot_mat_*src_points_).colwise() + t_vec_;

            // Compute correspondences
            find_correspondences_();

            iteration_count_++;

            if (correspondences_.size() < 3 || (metric_ != Metric::POINT_TO_POINT && correspondences_.size() < 6)) {
                has_converged_ = false;
                break;
            }

            // Update estimated transformation
            switch (metric_) {
                case Metric::POINT_TO_POINT:
                    estimateRigidTransformPointToPointClosedForm<float,3>(dst_points_, src_points_trans_, correspondences_, tform_iter);
//                    estimateRigidTransformPointToPointIterative<float>(*dst_points_, src_points_trans_, correspondences_, tform_iter, max_estimation_iter_, convergence_tol_);
                    break;
                case Metric::POINT_TO_PLANE:
                    estimateRigidTransformPointToPlane3D<float>(dst_points_, dst_normals_, src_points_trans_, correspondences_, tform_iter, max_estimation_iter_, convergence_tol_);
                    break;
                case Metric::COMBINED:
                    estimateRigidTransformCombinedMetric3D<float>(dst_points_, dst_normals_, src_points_trans_, correspondences_, point_to_point_weight_, point_to_plane_weight_, tform_iter, max_estimation_iter_, convergence_tol_);
                    break;
            }

            rot_mat_ = tform_iter.linear()*rot_mat_;
            t_vec_ = tform_iter.linear()*t_vec_ + tform_iter.translation();

            // Orthonormalize rotation
            rot_mat_ = orthonormalize_rotation_(rot_mat_);

            // Check for convergence
            Eigen::Vector3f tmp = tform_iter.linear().eulerAngles(2,1,0).cwiseAbs();
            tmp = tmp.cwiseMin((v_pi-tmp).cwiseAbs());
            delta.head(3) = tmp;
            delta.tail(3) = tform_iter.translation();

            if (delta.norm() < convergence_tol_) {
                has_converged_ = true;
                break;
            }
        }
    }

    void IterativeClosestPoint::compute_residuals_(const CorrespondencesType &corr_type, const Metric &metric, std::vector<float> &residuals) {
        if (iteration_count_ == 0) estimate_transform_();

        CorrespondencesType req_corr_type = correct_correspondences_type_(corr_type);
        Metric req_metric = (dst_normals_.cols() > 0) ? metric : Metric::POINT_TO_POINT;

        size_t neighbor;
        float distance;

        residuals.resize(src_points_.cols());
        switch (req_corr_type) {
            case CorrespondencesType::POINTS: {
                KDTree<float,3,KDTreeDistanceAdaptors::L2> *kd_tree;
                if (req_corr_type == corr_type_) {
                    kd_tree = kd_tree_3d_;
                } else {
                    kd_tree = new KDTree<float,3,KDTreeDistanceAdaptors::L2>(dst_points_);
                }
#pragma omp parallel for shared (residuals) private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.cols(); i++) {
                    Eigen::Vector3f pt_trans = rot_mat_*src_points_.col(i) + t_vec_;
                    kd_tree->nearestNeighborSearch(pt_trans, neighbor, distance);
                    switch (req_metric) {
                        case Metric::POINT_TO_POINT: {
                            residuals[i] = std::sqrt(distance);
                            break;
                        }
                        case Metric::POINT_TO_PLANE: {
                            const Eigen::Vector3f &dp = dst_points_.col(neighbor);
                            const Eigen::Vector3f &dn = dst_normals_.col(neighbor);
                            residuals[i] = std::abs(dn.dot(pt_trans - dp));
                            break;
                        }
                        case Metric::COMBINED: {
                            const Eigen::Vector3f &dp = dst_points_.col(neighbor);
                            const Eigen::Vector3f &dn = dst_normals_.col(neighbor);
                            residuals[i] = point_to_point_weight_*std::sqrt(distance) + point_to_plane_weight_*std::abs(dn.dot(pt_trans - dp));
                            break;
                        }
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
                    kd_tree = new KDTree<float,3,KDTreeDistanceAdaptors::L2>(dst_normals_);
                }
#pragma omp parallel for shared (residuals) private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.cols(); i++) {
                    Eigen::Vector3f pt_trans = rot_mat_*src_points_.col(i) + t_vec_;
                    kd_tree->nearestNeighborSearch(rot_mat_*src_normals_.col(i), neighbor, distance);
                    switch (req_metric) {
                        case Metric::POINT_TO_POINT: {
                            residuals[i] = std::sqrt(distance);
                            break;
                        }
                        case Metric::POINT_TO_PLANE: {
                            const Eigen::Vector3f &dp = dst_points_.col(neighbor);
                            const Eigen::Vector3f &dn = dst_normals_.col(neighbor);
                            residuals[i] = std::abs(dn.dot(pt_trans - dp));
                            break;
                        }
                        case Metric::COMBINED: {
                            const Eigen::Vector3f &dp = dst_points_.col(neighbor);
                            const Eigen::Vector3f &dn = dst_normals_.col(neighbor);
                            residuals[i] = point_to_point_weight_*std::sqrt(distance) + point_to_plane_weight_*std::abs(dn.dot(pt_trans - dp));
                            break;
                        }
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
                    kd_tree = new KDTree<float,3,KDTreeDistanceAdaptors::L2>(dst_colors_);
                }
#pragma omp parallel for shared (residuals) private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.cols(); i++) {
                    Eigen::Vector3f pt_trans = rot_mat_*src_points_.col(i) + t_vec_;
                    kd_tree->nearestNeighborSearch(src_colors_.col(i), neighbor, distance);
                    switch (req_metric) {
                        case Metric::POINT_TO_POINT: {
                            residuals[i] = std::sqrt(distance);
                            break;
                        }
                        case Metric::POINT_TO_PLANE: {
                            const Eigen::Vector3f &dp = dst_points_.col(neighbor);
                            const Eigen::Vector3f &dn = dst_normals_.col(neighbor);
                            residuals[i] = std::abs(dn.dot(pt_trans - dp));
                            break;
                        }
                        case Metric::COMBINED: {
                            const Eigen::Vector3f &dp = dst_points_.col(neighbor);
                            const Eigen::Vector3f &dn = dst_normals_.col(neighbor);
                            residuals[i] = point_to_point_weight_*std::sqrt(distance) + point_to_plane_weight_*std::abs(dn.dot(pt_trans - dp));
                            break;
                        }
                    }
                }
                if (req_corr_type != corr_type_) {
                    delete kd_tree;
                }
                break;
            }
            case CorrespondencesType::POINTS_NORMALS: {
                KDTree<float,6,KDTreeDistanceAdaptors::L2> *kd_tree;
                VectorSet<float,6> data_holder;
                if (req_corr_type == corr_type_) {
                    kd_tree = kd_tree_6d_;
                } else {
                    data_holder.resize(Eigen::NoChange, dst_points_.cols());
                    data_holder.topRows(3) = point_dist_weight_*dst_points_;
                    data_holder.bottomRows(3) = normal_dist_weight_*dst_normals_;
                    kd_tree = new KDTree<float,6,KDTreeDistanceAdaptors::L2>(data_holder);
                }
#pragma omp parallel for shared (residuals) private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.cols(); i++) {
                    Eigen::Vector3f pt_trans = rot_mat_*src_points_.col(i) + t_vec_;
                    Eigen::Matrix<float,6,1> query_pt;
                    query_pt.head(3) = point_dist_weight_*pt_trans;
                    query_pt.tail(3) = normal_dist_weight_*rot_mat_*src_normals_.col(i);
                    kd_tree->nearestNeighborSearch(query_pt, neighbor, distance);
                    switch (req_metric) {
                        case Metric::POINT_TO_POINT: {
                            residuals[i] = std::sqrt(distance);
                            break;
                        }
                        case Metric::POINT_TO_PLANE: {
                            const Eigen::Vector3f &dp = dst_points_.col(neighbor);
                            const Eigen::Vector3f &dn = dst_normals_.col(neighbor);
                            residuals[i] = std::abs(dn.dot(pt_trans - dp));
                            break;
                        }
                        case Metric::COMBINED: {
                            const Eigen::Vector3f &dp = dst_points_.col(neighbor);
                            const Eigen::Vector3f &dn = dst_normals_.col(neighbor);
                            residuals[i] = point_to_point_weight_*std::sqrt(distance) + point_to_plane_weight_*std::abs(dn.dot(pt_trans - dp));
                            break;
                        }
                    }
                }
                if (req_corr_type != corr_type_) {
                    delete kd_tree;
                }
                break;
            }
            case CorrespondencesType::POINTS_COLORS: {
                KDTree<float,6,KDTreeDistanceAdaptors::L2> *kd_tree;
                VectorSet<float,6> data_holder;
                if (req_corr_type == corr_type_) {
                    kd_tree = kd_tree_6d_;
                } else {
                    data_holder.resize(Eigen::NoChange, dst_points_.cols());
                    data_holder.topRows(3) = point_dist_weight_*dst_points_;
                    data_holder.bottomRows(3) = color_dist_weight_*dst_colors_;
                    kd_tree = new KDTree<float,6,KDTreeDistanceAdaptors::L2>(data_holder);
                }
#pragma omp parallel for shared (residuals) private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.cols(); i++) {
                    Eigen::Vector3f pt_trans = rot_mat_*src_points_.col(i) + t_vec_;
                    Eigen::Matrix<float,6,1> query_pt;
                    query_pt.head(3) = point_dist_weight_*pt_trans;
                    query_pt.tail(3) = color_dist_weight_*src_colors_.col(i);
                    kd_tree->nearestNeighborSearch(query_pt, neighbor, distance);
                    switch (req_metric) {
                        case Metric::POINT_TO_POINT: {
                            residuals[i] = std::sqrt(distance);
                            break;
                        }
                        case Metric::POINT_TO_PLANE: {
                            const Eigen::Vector3f &dp = dst_points_.col(neighbor);
                            const Eigen::Vector3f &dn = dst_normals_.col(neighbor);
                            residuals[i] = std::abs(dn.dot(pt_trans - dp));
                            break;
                        }
                        case Metric::COMBINED: {
                            const Eigen::Vector3f &dp = dst_points_.col(neighbor);
                            const Eigen::Vector3f &dn = dst_normals_.col(neighbor);
                            residuals[i] = point_to_point_weight_*std::sqrt(distance) + point_to_plane_weight_*std::abs(dn.dot(pt_trans - dp));
                            break;
                        }
                    }
                }
                if (req_corr_type != corr_type_) {
                    delete kd_tree;
                }
                break;
            }
            case CorrespondencesType::NORMALS_COLORS: {
                KDTree<float,6,KDTreeDistanceAdaptors::L2> *kd_tree;
                VectorSet<float,6> data_holder;
                if (req_corr_type == corr_type_) {
                    kd_tree = kd_tree_6d_;
                } else {
                    data_holder.resize(Eigen::NoChange, dst_points_.cols());
                    data_holder.topRows(3) = normal_dist_weight_*dst_normals_;
                    data_holder.bottomRows(3) = color_dist_weight_*dst_colors_;
                    kd_tree = new KDTree<float,6,KDTreeDistanceAdaptors::L2>(data_holder);
                }
#pragma omp parallel for shared (residuals) private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.cols(); i++) {
                    Eigen::Vector3f pt_trans = rot_mat_*src_points_.col(i) + t_vec_;
                    Eigen::Matrix<float,6,1> query_pt;
                    query_pt.head(3) = normal_dist_weight_*rot_mat_*src_normals_.col(i);
                    query_pt.tail(3) = color_dist_weight_*src_colors_.col(i);
                    kd_tree->nearestNeighborSearch(query_pt, neighbor, distance);
                    switch (req_metric) {
                        case Metric::POINT_TO_POINT: {
                            residuals[i] = std::sqrt(distance);
                            break;
                        }
                        case Metric::POINT_TO_PLANE: {
                            const Eigen::Vector3f &dp = dst_points_.col(neighbor);
                            const Eigen::Vector3f &dn = dst_normals_.col(neighbor);
                            residuals[i] = std::abs(dn.dot(pt_trans - dp));
                            break;
                        }
                        case Metric::COMBINED: {
                            const Eigen::Vector3f &dp = dst_points_.col(neighbor);
                            const Eigen::Vector3f &dn = dst_normals_.col(neighbor);
                            residuals[i] = point_to_point_weight_*std::sqrt(distance) + point_to_plane_weight_*std::abs(dn.dot(pt_trans - dp));
                            break;
                        }
                    }
                }
                if (req_corr_type != corr_type_) {
                    delete kd_tree;
                }
                break;
            }
            case CorrespondencesType::POINTS_NORMALS_COLORS: {
                KDTree<float,9,KDTreeDistanceAdaptors::L2> *kd_tree;
                VectorSet<float,9> data_holder;
                if (req_corr_type == corr_type_) {
                    kd_tree = kd_tree_9d_;
                } else {
                    data_holder.resize(Eigen::NoChange, dst_points_.cols());
                    data_holder.topRows(3) = point_dist_weight_*dst_points_;
                    data_holder.block(3,0,3,dst_data_points_9d_.cols()) = normal_dist_weight_*dst_normals_;
                    data_holder.bottomRows(3) = color_dist_weight_*dst_colors_;
                    kd_tree = new KDTree<float,9,KDTreeDistanceAdaptors::L2>(data_holder);
                }
#pragma omp parallel for shared (residuals) private (neighbor, distance)
                for (size_t i = 0; i < src_points_trans_.cols(); i++) {
                    Eigen::Vector3f pt_trans = rot_mat_*src_points_.col(i) + t_vec_;
                    Eigen::Matrix<float,9,1> query_pt;
                    query_pt.head(3) = point_dist_weight_*pt_trans;
                    query_pt.segment(3,3) = normal_dist_weight_*rot_mat_*src_normals_.col(i);
                    query_pt.tail(3) = color_dist_weight_*src_colors_.col(i);
                    kd_tree->nearestNeighborSearch(query_pt, neighbor, distance);
                    switch (req_metric) {
                        case Metric::POINT_TO_POINT: {
                            residuals[i] = std::sqrt(distance);
                            break;
                        }
                        case Metric::POINT_TO_PLANE: {
                            const Eigen::Vector3f &dp = dst_points_.col(neighbor);
                            const Eigen::Vector3f &dn = dst_normals_.col(neighbor);
                            residuals[i] = std::abs(dn.dot(pt_trans - dp));
                            break;
                        }
                        case Metric::COMBINED: {
                            const Eigen::Vector3f &dp = dst_points_.col(neighbor);
                            const Eigen::Vector3f &dn = dst_normals_.col(neighbor);
                            residuals[i] = point_to_point_weight_*std::sqrt(distance) + point_to_plane_weight_*std::abs(dn.dot(pt_trans - dp));
                            break;
                        }
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
