#include <cilantro/iterative_closest_point.hpp>

#include <iostream>

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
    Eigen::Matrix<float,6,1> d_theta;
    rot_mat.setIdentity();
    t_vec.setZero();
    Eigen::Matrix<float,3,Eigen::Dynamic> src_t(src_p);

    Eigen::Matrix<float,Eigen::Dynamic,6> A(dst_p.cols(),6);
    Eigen::Matrix<float,Eigen::Dynamic,1> b(dst_p.cols(),1);

    Eigen::Vector3f v_pi(M_PI, M_PI, M_PI);

    size_t iter = 0;
    while (iter < max_iter) {
        if (iter > 0) {
            src_t = (rot_mat*src_p).colwise() + t_vec;
        }

        // Compute differential
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

        d_theta = (A.transpose()*A).ldlt().solve(A.transpose()*b);

        // Update estimate
        rot_mat_iter = Eigen::AngleAxisf(d_theta[2],Eigen::Vector3f::UnitZ()) *
                       Eigen::AngleAxisf(d_theta[1],Eigen::Vector3f::UnitY()) *
                       Eigen::AngleAxisf(d_theta[0],Eigen::Vector3f::UnitX());

        rot_mat = rot_mat_iter*rot_mat;
        t_vec = rot_mat_iter*t_vec + d_theta.tail(3);

        iter++;

        // Check for convergence
        if (d_theta.norm() < convergence_tol) return true;
    }

    return false;
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
