#include <cilantro/iterative_closest_point.hpp>

IterativeClosestPoint::IterativeClosestPoint(const std::vector<Eigen::Vector3f> &dst_p, const std::vector<Eigen::Vector3f> &src_p)
        : dst_points_(&dst_p),
          dst_normals_(NULL),
          src_points_(&src_p),
          kd_tree_ptr_(new KDTree(dst_p)),
          kd_tree_owned_(true),
          metric_(Metric::POINT_TO_POINT)
{
    init_params_();
}

IterativeClosestPoint::IterativeClosestPoint(const std::vector<Eigen::Vector3f> &dst_p, const std::vector<Eigen::Vector3f> &src_p, const KDTree &kd_tree)
        : dst_points_(&dst_p),
          dst_normals_(NULL),
          src_points_(&src_p),
          kd_tree_ptr_((KDTree*)&kd_tree),
          kd_tree_owned_(false),
          metric_(Metric::POINT_TO_POINT)
{
    init_params_();
}

IterativeClosestPoint::IterativeClosestPoint(const std::vector<Eigen::Vector3f> &dst_p, const std::vector<Eigen::Vector3f> &dst_n, const std::vector<Eigen::Vector3f> &src_p)
        : dst_points_(&dst_p),
          dst_normals_(&dst_n),
          src_points_(&src_p),
          kd_tree_ptr_(new KDTree(dst_p)),
          kd_tree_owned_(true),
          metric_((dst_n.size() == dst_p.size()) ? Metric::POINT_TO_PLANE : Metric::POINT_TO_POINT)
{
    init_params_();
}

IterativeClosestPoint::IterativeClosestPoint(const std::vector<Eigen::Vector3f> &dst_p, const std::vector<Eigen::Vector3f> &dst_n, const std::vector<Eigen::Vector3f> &src_p, const KDTree &kd_tree)
        : dst_points_(&dst_p),
          dst_normals_(&dst_n),
          src_points_(&src_p),
          kd_tree_ptr_((KDTree*)&kd_tree),
          kd_tree_owned_(false),
          metric_((dst_n.size() == dst_p.size()) ? Metric::POINT_TO_PLANE : Metric::POINT_TO_POINT)
{
    init_params_();
}

IterativeClosestPoint::IterativeClosestPoint(const PointCloud &dst, const PointCloud &src, Metric metric)
        : dst_points_(&dst.points),
          dst_normals_((dst.hasNormals()) ? &dst.normals : NULL),
          src_points_(&src.points),
          kd_tree_ptr_(new KDTree(dst.points)),
          kd_tree_owned_(true),
          metric_((dst.hasNormals()) ? metric : Metric::POINT_TO_POINT)
{
    init_params_();
}

IterativeClosestPoint::IterativeClosestPoint(const PointCloud &dst, const PointCloud &src, const KDTree &kd_tree, Metric metric)
        : dst_points_(&dst.points),
          dst_normals_((dst.hasNormals()) ? &dst.normals : NULL),
          src_points_(&src.points),
          kd_tree_ptr_((KDTree*)&kd_tree),
          kd_tree_owned_(false),
          metric_((dst.hasNormals()) ? metric : Metric::POINT_TO_POINT)
{
    init_params_();
}

IterativeClosestPoint::~IterativeClosestPoint() {
    if (kd_tree_owned_) delete kd_tree_ptr_;
}

void IterativeClosestPoint::init_params_() {
    corr_dist_thres_ = 0.05f;
    convergence_tol_ = 1e-3f;
    max_iter_ = 10;
    point_to_plane_max_iter_ = 5;

    has_converged_ = false;
    iteration_count_ = 0;

    rot_mat_init_.setIdentity();
    t_vec_init_.setZero();
}

void IterativeClosestPoint::compute_() {

    has_converged_ = false;

    rot_mat_ = rot_mat_init_;
    t_vec_ = t_vec_init_;

    Eigen::Matrix3f rot_mat_iter;
    Eigen::Vector3f t_vec_iter;
    Eigen::Matrix<float,6,1> delta;

    Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> > src_p((float *)(src_points_->data()),3,src_points_->size());
    std::vector<Eigen::Vector3f> src_points_t(*src_points_);

    Eigen::Vector3f v_pi(M_PI, M_PI, M_PI);

    std::vector<size_t> dst_ind, src_ind;
    std::vector<size_t> neighbors(1);
    std::vector<float> distances(1);

    float corr_thresh_squared = corr_dist_thres_*corr_dist_thres_;

    iteration_count_ = 0;
    while (iteration_count_ < max_iter_) {
        // Transform src using current estimate
        Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)(src_points_t.data()),3,src_points_t.size()) = (rot_mat_*src_p).colwise() + t_vec_;

        // Compute correspondences
        dst_ind.resize(src_points_->size());
        src_ind.resize(src_points_->size());
        size_t k = 0;
        for (size_t i = 0; i < src_points_->size(); i++) {
            kd_tree_ptr_->kNNSearch(src_points_t[i], 1, neighbors, distances);
            if (distances[0] < corr_thresh_squared) {
                if (metric_ == Metric::POINT_TO_PLANE && (*dst_normals_)[neighbors[0]].array().isNaN().any()) continue;
                dst_ind[k] = neighbors[0];
                src_ind[k] = i;
                k++;
            }
        }
        dst_ind.resize(k);
        src_ind.resize(k);

        if (dst_ind.empty()) {
            has_converged_ = false;
            break;
        }

        // Update estimated transformation
        if (metric_ == Metric::POINT_TO_PLANE) {
            estimateRigidTransformPointToPlane(*dst_points_, *dst_normals_, src_points_t, dst_ind, src_ind, rot_mat_iter, t_vec_iter, point_to_plane_max_iter_, convergence_tol_);
        } else if (metric_ == Metric::POINT_TO_POINT) {
            estimateRigidTransformPointToPoint(*dst_points_, src_points_t, dst_ind, src_ind, rot_mat_iter, t_vec_iter);
        } else {
            return;
        }

        rot_mat_ = rot_mat_iter*rot_mat_;
        t_vec_ = rot_mat_iter*t_vec_ + t_vec_iter;

        iteration_count_++;

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
