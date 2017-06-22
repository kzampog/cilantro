#pragma once

#include <cilantro/point_cloud.hpp>
#include <cilantro/kd_tree.hpp>

bool estimateRigidTransformPointToPoint(const Eigen::Matrix<float,3,Eigen::Dynamic> &dst,
                                        const Eigen::Matrix<float,3,Eigen::Dynamic> &src,
                                        Eigen::Matrix3f &rot_mat,
                                        Eigen::Vector3f &t_vec);

bool estimateRigidTransformPointToPoint(const Eigen::Matrix<float,3,Eigen::Dynamic> &dst,
                                        const Eigen::Matrix<float,3,Eigen::Dynamic> &src,
                                        const std::vector<size_t> &dst_ind,
                                        const std::vector<size_t> &src_ind,
                                        Eigen::Matrix3f &rot_mat,
                                        Eigen::Vector3f &t_vec);

inline bool estimateRigidTransformPointToPoint(const std::vector<Eigen::Vector3f> &dst,
                                               const std::vector<Eigen::Vector3f> &src,
                                               Eigen::Matrix3f &rot_mat,
                                               Eigen::Vector3f &t_vec)
{
    return estimateRigidTransformPointToPoint(Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst.data(),3,dst.size()), Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)src.data(),3,src.size()), rot_mat, t_vec);
}

inline bool estimateRigidTransformPointToPoint(const std::vector<Eigen::Vector3f> &dst,
                                               const std::vector<Eigen::Vector3f> &src,
                                               const std::vector<size_t> &dst_ind,
                                               const std::vector<size_t> &src_ind,
                                               Eigen::Matrix3f &rot_mat,
                                               Eigen::Vector3f &t_vec)
{
    return estimateRigidTransformPointToPoint(Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst.data(),3,dst.size()), Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)src.data(),3,src.size()), dst_ind, src_ind, rot_mat, t_vec);
}

inline bool estimateRigidTransformPointToPoint(const PointCloud &dst,
                                               const PointCloud &src,
                                               Eigen::Matrix3f &rot_mat,
                                               Eigen::Vector3f &t_vec)
{
    return estimateRigidTransformPointToPoint(dst.points, src.points, rot_mat, t_vec);
}

inline bool estimateRigidTransformPointToPoint(const PointCloud &dst,
                                               const PointCloud &src,
                                               const std::vector<size_t> &dst_ind,
                                               const std::vector<size_t> &src_ind,
                                               Eigen::Matrix3f &rot_mat,
                                               Eigen::Vector3f &t_vec)
{
    return estimateRigidTransformPointToPoint(dst.points, src.points, dst_ind, src_ind, rot_mat, t_vec);
}

bool estimateRigidTransformPointToPlane(const Eigen::Matrix<float,3,Eigen::Dynamic> &dst_p,
                                        const Eigen::Matrix<float,3,Eigen::Dynamic> &dst_n,
                                        const Eigen::Matrix<float,3,Eigen::Dynamic> &src_p,
                                        Eigen::Matrix3f &rot_mat,
                                        Eigen::Vector3f &t_vec,
                                        size_t max_iter = 1,
                                        float convergence_tol = 1e-5f);

bool estimateRigidTransformPointToPlane(const Eigen::Matrix<float,3,Eigen::Dynamic> &dst_p,
                                        const Eigen::Matrix<float,3,Eigen::Dynamic> &dst_n,
                                        const Eigen::Matrix<float,3,Eigen::Dynamic> &src_p,
                                        const std::vector<size_t> &dst_ind,
                                        const std::vector<size_t> &src_ind,
                                        Eigen::Matrix3f &rot_mat,
                                        Eigen::Vector3f &t_vec,
                                        size_t max_iter = 1,
                                        float convergence_tol = 1e-5f);

inline bool estimateRigidTransformPointToPlane(const std::vector<Eigen::Vector3f> &dst_p,
                                               const std::vector<Eigen::Vector3f> &dst_n,
                                               const std::vector<Eigen::Vector3f> &src_p,
                                               Eigen::Matrix3f &rot_mat,
                                               Eigen::Vector3f &t_vec,
                                               size_t max_iter = 1,
                                               float convergence_tol = 1e-5f)
{
    return estimateRigidTransformPointToPlane(Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_p.data(),3,dst_p.size()), Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_n.data(),3,dst_n.size()), Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)src_p.data(),3,src_p.size()), rot_mat, t_vec, max_iter, convergence_tol);
}

inline bool estimateRigidTransformPointToPlane(const std::vector<Eigen::Vector3f> &dst_p,
                                               const std::vector<Eigen::Vector3f> &dst_n,
                                               const std::vector<Eigen::Vector3f> &src_p,
                                               const std::vector<size_t> &dst_ind,
                                               const std::vector<size_t> &src_ind,
                                               Eigen::Matrix3f &rot_mat,
                                               Eigen::Vector3f &t_vec,
                                               size_t max_iter = 1,
                                               float convergence_tol = 1e-5f)
{
    return estimateRigidTransformPointToPlane(Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_p.data(),3,dst_p.size()), Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)dst_n.data(),3,dst_n.size()), Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)src_p.data(),3,src_p.size()), dst_ind, src_ind, rot_mat, t_vec, max_iter, convergence_tol);
}

inline bool estimateRigidTransformPointToPlane(const PointCloud &dst,
                                               const PointCloud &src,
                                               Eigen::Matrix3f &rot_mat,
                                               Eigen::Vector3f &t_vec,
                                               size_t max_iter = 1,
                                               float convergence_tol = 1e-5f)
{
    return estimateRigidTransformPointToPlane(dst.points, dst.normals, src.points, rot_mat, t_vec, max_iter, convergence_tol);
}

inline bool estimateRigidTransformPointToPlane(const PointCloud &dst,
                                               const PointCloud &src,
                                               const std::vector<size_t> &dst_ind,
                                               const std::vector<size_t> &src_ind,
                                               Eigen::Matrix3f &rot_mat,
                                               Eigen::Vector3f &t_vec,
                                               size_t max_iter = 1,
                                               float convergence_tol = 1e-5f)
{
    return estimateRigidTransformPointToPlane(dst.points, dst.normals, src.points, dst_ind, src_ind, rot_mat, t_vec, max_iter, convergence_tol);
}

class IterativeClosestPoint {
public:
    enum struct Metric { POINT_TO_POINT, POINT_TO_PLANE };

    IterativeClosestPoint(const std::vector<Eigen::Vector3f> &dst_p, const std::vector<Eigen::Vector3f> &src_p);
    IterativeClosestPoint(const std::vector<Eigen::Vector3f> &dst_p, const std::vector<Eigen::Vector3f> &src_p, const KDTree &kd_tree);
    IterativeClosestPoint(const std::vector<Eigen::Vector3f> &dst_p, const std::vector<Eigen::Vector3f> &dst_n, const std::vector<Eigen::Vector3f> &src_p);
    IterativeClosestPoint(const std::vector<Eigen::Vector3f> &dst_p, const std::vector<Eigen::Vector3f> &dst_n, const std::vector<Eigen::Vector3f> &src_p, const KDTree &kd_tree);
    IterativeClosestPoint(const PointCloud &dst, const PointCloud &src, Metric metric = Metric::POINT_TO_POINT);
    IterativeClosestPoint(const PointCloud &dst, const PointCloud &src, const KDTree &kd_tree, Metric metric = Metric::POINT_TO_POINT);

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

    inline IterativeClosestPoint& setInitialTransformation(const Eigen::Matrix3f &rot_mat, const Eigen::Vector3f &t_vec) {
        iteration_count_ = 0;
        rot_mat_init_ = rot_mat;
        t_vec_init_ = t_vec;
        return *this;
    }

    inline IterativeClosestPoint& getTransformation(Eigen::Matrix3f &rot_mat, Eigen::Vector3f &t_vec) {
        if (iteration_count_ == 0) compute_();
        rot_mat = rot_mat_;
        t_vec = t_vec_;
        return *this;
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

    // Object state
    bool has_converged_;
    size_t iteration_count_;
    Eigen::Matrix3f rot_mat_init_;
    Eigen::Vector3f t_vec_init_;

    Eigen::Matrix3f rot_mat_;
    Eigen::Vector3f t_vec_;

    void init_params_();
    void compute_();
};
