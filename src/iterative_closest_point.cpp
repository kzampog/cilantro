#include <cilantro/iterative_closest_point.hpp>
#include <cilantro/transform_estimation.hpp>

IterativeClosestPoint::IterativeClosestPoint(const std::vector<Eigen::Vector3f> &dst_p, const std::vector<Eigen::Vector3f> &src_p)
        : dst_points_(&dst_p),
          dst_normals_(NULL),
          src_points_(&src_p),
          kd_tree_ptr_(new KDTree3D(dst_p)),
          kd_tree_owned_(true),
          metric_(Metric::POINT_TO_POINT),
          src_points_trans_(std::vector<Eigen::Vector3f>(src_p.size()))
{
    init_params_();
}

IterativeClosestPoint::IterativeClosestPoint(const std::vector<Eigen::Vector3f> &dst_p, const std::vector<Eigen::Vector3f> &src_p, const KDTree3D &kd_tree)
        : dst_points_(&dst_p),
          dst_normals_(NULL),
          src_points_(&src_p),
          kd_tree_ptr_((KDTree3D*)&kd_tree),
          kd_tree_owned_(false),
          metric_(Metric::POINT_TO_POINT),
          src_points_trans_(std::vector<Eigen::Vector3f>(src_p.size()))
{
    init_params_();
}

IterativeClosestPoint::IterativeClosestPoint(const std::vector<Eigen::Vector3f> &dst_p, const std::vector<Eigen::Vector3f> &dst_n, const std::vector<Eigen::Vector3f> &src_p)
        : dst_points_(&dst_p),
          dst_normals_(&dst_n),
          src_points_(&src_p),
          kd_tree_ptr_(new KDTree3D(dst_p)),
          kd_tree_owned_(true),
          metric_((dst_n.size() == dst_p.size()) ? Metric::POINT_TO_PLANE : Metric::POINT_TO_POINT),
          src_points_trans_(std::vector<Eigen::Vector3f>(src_p.size()))
{
    init_params_();
}

IterativeClosestPoint::IterativeClosestPoint(const std::vector<Eigen::Vector3f> &dst_p, const std::vector<Eigen::Vector3f> &dst_n, const std::vector<Eigen::Vector3f> &src_p, const KDTree3D &kd_tree)
        : dst_points_(&dst_p),
          dst_normals_(&dst_n),
          src_points_(&src_p),
          kd_tree_ptr_((KDTree3D*)&kd_tree),
          kd_tree_owned_(false),
          metric_((dst_n.size() == dst_p.size()) ? Metric::POINT_TO_PLANE : Metric::POINT_TO_POINT),
          src_points_trans_(std::vector<Eigen::Vector3f>(src_p.size()))
{
    init_params_();
}

IterativeClosestPoint::IterativeClosestPoint(const PointCloud &dst, const PointCloud &src, const Metric &metric)
        : dst_points_(&dst.points),
          dst_normals_((dst.hasNormals()) ? &dst.normals : NULL),
          src_points_(&src.points),
          kd_tree_ptr_(new KDTree3D(dst.points)),
          kd_tree_owned_(true),
          metric_((dst.hasNormals()) ? metric : Metric::POINT_TO_POINT),
          src_points_trans_(std::vector<Eigen::Vector3f>(src.points.size()))
{
    init_params_();
}

IterativeClosestPoint::IterativeClosestPoint(const PointCloud &dst, const PointCloud &src, const KDTree3D &kd_tree, const Metric &metric)
        : dst_points_(&dst.points),
          dst_normals_((dst.hasNormals()) ? &dst.normals : NULL),
          src_points_(&src.points),
          kd_tree_ptr_((KDTree3D*)&kd_tree),
          kd_tree_owned_(false),
          metric_((dst.hasNormals()) ? metric : Metric::POINT_TO_POINT),
          src_points_trans_(std::vector<Eigen::Vector3f>(src.points.size()))
{
    init_params_();
}

IterativeClosestPoint::~IterativeClosestPoint() {
    if (kd_tree_owned_) delete kd_tree_ptr_;
}

void IterativeClosestPoint::init_params_() {
    corr_dist_thres_ = 0.05f;
    convergence_tol_ = 1e-3f;
    max_iter_ = 15;
    point_to_plane_max_iter_ = 1;

    has_converged_ = false;
    iteration_count_ = 0;

    rot_mat_init_.setIdentity();
    t_vec_init_.setZero();
}

void IterativeClosestPoint::estimate_transform_() {
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
    std::vector<size_t> neighbors(1);
    std::vector<float> distances(1);

    float corr_thresh_squared = corr_dist_thres_*corr_dist_thres_;

    iteration_count_ = 0;
    while (iteration_count_ < max_iter_) {
        // Transform src using current estimate
        src_t = (rot_mat_*src_p).colwise() + t_vec_;

        // Compute correspondences
        dst_ind.resize(src_points_trans_.size());
        src_ind.resize(src_points_trans_.size());
        size_t k = 0;
#pragma omp parallel for shared (k) private (neighbors, distances)
        for (size_t i = 0; i < src_points_trans_.size(); i++) {
            kd_tree_ptr_->kNNSearch(src_points_trans_[i], 1, neighbors, distances);
#pragma omp critical
            if (distances[0] < corr_thresh_squared) {
                if (metric_ != Metric::POINT_TO_PLANE || (*dst_normals_)[neighbors[0]].allFinite()) {
                    dst_ind[k] = neighbors[0];
                    src_ind[k] = i;
                    k++;
                }
            }
        }
        dst_ind.resize(k);
        src_ind.resize(k);

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
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(rot_mat_, Eigen::ComputeFullU | Eigen::ComputeFullV);
        rot_mat_ = svd.matrixU()*(svd.matrixV().transpose());

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

    Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> > src_p((float *)(src_points_->data()),3,src_points_->size());
    Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> > src_t((float *)(src_points_trans_.data()),3,src_points_trans_.size());

    src_t = (rot_mat_*src_p).colwise() + t_vec_;

    std::vector<size_t> neighbors(1);
    std::vector<float> distances(1);
    residuals.resize(src_points_->size());
    if (metric == Metric::POINT_TO_PLANE) {
#pragma omp parallel for private (neighbors, distances)
        for (size_t i = 0; i < residuals.size(); i++) {
            kd_tree_ptr_->kNNSearch(src_points_trans_[i], 1, neighbors, distances);
            const Eigen::Vector3f& dp = (*dst_points_)[neighbors[0]];
            const Eigen::Vector3f& dn = (*dst_normals_)[neighbors[0]];
            residuals[i] = std::abs(dn.dot(src_points_trans_[i] - dp));
        }
    } else if (metric == Metric::POINT_TO_POINT) {
#pragma omp parallel for private (neighbors, distances)
        for (size_t i = 0; i < residuals.size(); i++) {
            kd_tree_ptr_->kNNSearch(src_points_trans_[i], 1, neighbors, distances);
            residuals[i] = std::sqrt(distances[0]);
        }
    }
}
