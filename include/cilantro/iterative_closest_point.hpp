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

private:
    const std::vector<Eigen::Vector3f> *dst_points_;
    const std::vector<Eigen::Vector3f> *dst_normals_;
    const std::vector<Eigen::Vector3f> *src_points_;
    KDTree *kd_tree_ptr_;
    bool kd_tree_owned_;
    Metric metric_;
};
