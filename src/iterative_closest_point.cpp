#include <cilantro/iterative_closest_point.hpp>

bool estimateRigidTransformPointToPoint(const Eigen::Matrix<float,3,Eigen::Dynamic> &dst,
                                        const Eigen::Matrix<float,3,Eigen::Dynamic> &src,
                                        Eigen::Matrix3f &rot_mat,
                                        Eigen::Vector3f &t_vec)
{
    if (src.cols() < 3 || src.cols() != dst.cols()) {
        return false;
    }

    Eigen::Vector3f mu_dst(dst.rowwise().mean());
    Eigen::Vector3f mu_src(src.rowwise().mean());

    Eigen::Matrix<float,3,Eigen::Dynamic> dst_centered(dst.colwise() - mu_dst);
    Eigen::Matrix<float,3,Eigen::Dynamic> src_centered(src.colwise() - mu_src);

    Eigen::Matrix3f cov = dst_centered*(src_centered.transpose())/src.cols();

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U(svd.matrixU());
    Eigen::Matrix3f Vt(svd.matrixV().transpose());

    if (U.determinant() * Vt.determinant() < 0.0f) {
        U.col(2) *= -1.0f;
    }

    rot_mat = U*Vt;
    t_vec = mu_dst - rot_mat*mu_src;

    return true;
}

bool estimateRigidTransformPointToPoint(const Eigen::Matrix<float,3,Eigen::Dynamic> &dst,
                                        const Eigen::Matrix<float,3,Eigen::Dynamic> &src,
                                        const std::vector<size_t> &dst_ind,
                                        const std::vector<size_t> &src_ind,
                                        Eigen::Matrix3f &rot_mat,
                                        Eigen::Vector3f &t_vec)
{
    if (dst_ind.size() != src_ind.size()) {
        return false;
    }

    Eigen::Matrix<float,3,Eigen::Dynamic> dst_corr(3, dst_ind.size());
    Eigen::Matrix<float,3,Eigen::Dynamic> src_corr(3, src_ind.size());
    for (size_t i = 0; i < dst_ind.size(); i++) {
        dst_corr.col(i) = dst.col(dst_ind[i]);
        src_corr.col(i) = src.col(src_ind[i]);
    }

    return estimateRigidTransformPointToPoint(dst_corr, src_corr, rot_mat, t_vec);
}

bool estimateRigidTransformPointToPlane(const Eigen::Matrix<float,3,Eigen::Dynamic> &dst_p,
                                        const Eigen::Matrix<float,3,Eigen::Dynamic> &dst_n,
                                        const Eigen::Matrix<float,3,Eigen::Dynamic> &src_p,
                                        Eigen::Matrix3f &rot_mat,
                                        Eigen::Vector3f &t_vec,
                                        size_t max_iter,
                                        float convergence_tol)
{
    if (src_p.cols() < 6 || src_p.cols() != dst_p.cols() || dst_p.cols() != dst_n.cols()) {
        return false;
    }

    Eigen::Matrix3f rot_mat_iter;
    Eigen::Matrix<float,6,1> delta;
    rot_mat.setIdentity();
    t_vec.setZero();
    Eigen::Matrix<float,3,Eigen::Dynamic> src_t(src_p);

    Eigen::Matrix<float,Eigen::Dynamic,6> A(dst_p.cols(),6);
    Eigen::Matrix<float,Eigen::Dynamic,1> b(dst_p.cols(),1);

    size_t iter = 0;
    while (iter < max_iter) {
        if (iter > 0) {
            src_t = (rot_mat*src_p).colwise() + t_vec;
        }

        for (size_t i = 0; i < A.rows(); i++) {
            const Eigen::Vector3f& d = dst_p.col(i);
            const Eigen::Vector3f& n = dst_n.col(i);
            const Eigen::Vector3f& s = src_t.col(i);
            A(i,0) = n[2]*s[1] - n[1]*s[2];
            A(i,1) = n[0]*s[2] - n[2]*s[0];
            A(i,2) = n[1]*s[0] - n[0]*s[1];
            A(i,3) = n[0];
            A(i,4) = n[1];
            A(i,5) = n[2];
            b[i] = n[0]*d[0] + n[1]*d[1] + n[2]*d[2] - n[0]*s[0] - n[1]*s[1] - n[2]*s[2];
        }

        delta = (A.transpose()*A).ldlt().solve(A.transpose()*b);

        rot_mat_iter(0, 0) = std::cos(delta[2]) * std::cos(delta[1]);
        rot_mat_iter(0, 1) = -std::sin(delta[2]) * std::cos(delta[0]) + std::cos(delta[2]) * std::sin(delta[1]) * std::sin(delta[0]);
        rot_mat_iter(0, 2) = std::sin(delta[2]) * std::sin(delta[0]) + std::cos(delta[2]) * std::sin(delta[1]) * std::cos(delta[0]);
        rot_mat_iter(1, 0) = std::sin(delta[2]) * std::cos(delta[1]);
        rot_mat_iter(1, 1) = std::cos(delta[2]) * std::cos(delta[0]) + std::sin(delta[2]) * std::sin(delta[1]) * std::sin(delta[0]);
        rot_mat_iter(1, 2) = -std::cos(delta[2]) * std::sin(delta[0]) + std::sin(delta[2]) * std::sin(delta[1]) * std::cos(delta[0]);
        rot_mat_iter(2, 0) = -std::sin(delta[1]);
        rot_mat_iter(2, 1) = std::cos(delta[1]) * std::sin(delta[0]);
        rot_mat_iter(2, 2) = std::cos(delta[1]) * std::cos(delta[0]);

        rot_mat = rot_mat_iter*rot_mat;
        t_vec = rot_mat_iter*t_vec + delta.tail(3);

        if (delta.norm() < convergence_tol) return true;

        iter++;
    }

    return delta.norm() < convergence_tol;
}

bool estimateRigidTransformPointToPlane(const Eigen::Matrix<float,3,Eigen::Dynamic> &dst_p,
                                        const Eigen::Matrix<float,3,Eigen::Dynamic> &dst_n,
                                        const Eigen::Matrix<float,3,Eigen::Dynamic> &src_p,
                                        const std::vector<size_t> &dst_ind,
                                        const std::vector<size_t> &src_ind,
                                        Eigen::Matrix3f &rot_mat,
                                        Eigen::Vector3f &t_vec,
                                        size_t max_iter,
                                        float convergence_tol)
{
    if (dst_ind.size() != src_ind.size()) {
        return false;
    }

    Eigen::Matrix<float,3,Eigen::Dynamic> dst_p_corr(3, dst_ind.size());
    Eigen::Matrix<float,3,Eigen::Dynamic> dst_n_corr(3, dst_ind.size());
    Eigen::Matrix<float,3,Eigen::Dynamic> src_p_corr(3, src_ind.size());
    for (size_t i = 0; i < dst_ind.size(); i++) {
        dst_p_corr.col(i) = dst_p.col(dst_ind[i]);
        dst_n_corr.col(i) = dst_n.col(dst_ind[i]);
        src_p_corr.col(i) = src_p.col(src_ind[i]);
    }

    return estimateRigidTransformPointToPlane(dst_p_corr, dst_n_corr, src_p_corr, rot_mat, t_vec, max_iter, convergence_tol);
}

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

void IterativeClosestPoint::getTransformation(Eigen::Matrix3f &rot_mat, Eigen::Vector3f &t_vec) {
    if (!has_valid_results_) compute_();
    rot_mat = rot_mat_;
    t_vec = t_vec_;
}

void IterativeClosestPoint::init_params_() {
    corr_dist_thres_ = 0.05f;
    convergence_tol_ = 1e-3f;
    max_iter_ = 10;
    point_to_plane_max_iter_ = 5;

    has_valid_results_ = false;
    has_converged_ = false;
    rot_mat_.setIdentity();
    t_vec_.setZero();
}

void IterativeClosestPoint::compute_() {
    // TODO

    has_valid_results_ = true;
}
