#include <cilantro/convex_hull.hpp>

namespace cilantro {
    PointCloudHullFlat::PointCloudHullFlat(const ConstPointSetMatrixMap<float,3> &points, bool compute_topology, bool simplicial_facets, double merge_tol)
            : PrincipalComponentAnalysis3D(points),
              ConvexHull2D(project<2>(points), compute_topology, simplicial_facets, merge_tol),
              vertices_3d_(reconstruct<2>(vertices_))
    {}

//    PointCloudHullFlat::PointCloudHullFlat(const PointCloud &cloud, bool compute_topology, bool simplicial_facets, double merge_tol)
//            : PrincipalComponentAnalysis3D(cloud.points),
//              ConvexHull2D(project<2>(cloud.points), compute_topology, simplicial_facets, merge_tol),
//              vertices_3d_(reconstruct<2>(vertices_))
//    {}

    PointCloudHullFlat& PointCloudHullFlat::transform(const Eigen::Ref<const Eigen::Matrix3f> &rotation, const Eigen::Ref<const Eigen::Vector3f> &translation) {
        if (is_empty_) return *this;

        vertices_3d_ = (rotation*vertices_3d_).colwise() + translation;

        Eigen::Matrix<float,4,3> rec_mat;
        rec_mat.topLeftCorner(3,2) = eigenvectors_.leftCols(2);
        rec_mat.topRightCorner(3,1) = mean_;
        rec_mat.row(3).setZero();
        rec_mat(3,2) = 1.0f;

        Eigen::Matrix4f tform;
        tform.topLeftCorner(3,3) = rotation;
        tform.topRightCorner(3,1) = translation;
        tform.row(3).setZero();
        tform(3,3) = 1.0f;

        eigenvectors_ = rotation*eigenvectors_;
        mean_ = rotation*mean_ + translation;

        Eigen::Matrix<float,3,4> proj_mat;
        proj_mat.topLeftCorner(2,3) = eigenvectors_.leftCols(2).transpose();
        proj_mat.topRightCorner(2,1) = -eigenvectors_.leftCols(2).transpose()*mean_;
        proj_mat.row(2).setZero();
        proj_mat(2,3) = 1.0f;

        ConvexHull2D::transform(proj_mat*tform*rec_mat);

        return *this;
    }

    PointCloudHullFlat& PointCloudHullFlat::transform(const Eigen::Ref<const Eigen::Matrix4f> &rigid_transform) {
        return transform(rigid_transform.topLeftCorner(3,3), rigid_transform.topRightCorner(3,1));
    }

//    PointCloudHull::PointCloudHull(const std::vector<Eigen::Vector3f> &points, bool compute_topology, bool simplicial_facets, double merge_tol)
//            : ConvexHull3D(points, compute_topology, simplicial_facets, merge_tol)
//    {}
//
//    PointCloudHull::PointCloudHull(const PointCloud &cloud, bool compute_topology, bool simplicial_facets, double merge_tol)
//            : ConvexHull3D(cloud.points, compute_topology, simplicial_facets, merge_tol)
//    {}
//
//    Eigen::MatrixXf PointCloudHull::getPointSignedDistancesFromFacets(const PointCloud &cloud) const {
//        return ConvexHull3D::getPointSignedDistancesFromFacets(cloud.points);
//    }
//
//    std::vector<size_t> PointCloudHull::getInteriorPointIndices(const PointCloud &cloud, float offset) const {
//        return ConvexHull3D::getInteriorPointIndices(cloud.points, offset);
//    }
}
