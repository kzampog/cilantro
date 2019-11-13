#pragma once

#include <cilantro/spatial/convex_polytope.hpp>

namespace cilantro {
    template <typename ScalarT, typename IndexT = size_t>
    class FlatConvexHull3 : public PrincipalComponentAnalysis<ScalarT,3>, public ConvexHull<ScalarT,2,IndexT> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        FlatConvexHull3(const ConstVectorSetMatrixMap<ScalarT,3> &points,
                        bool compute_topology = false,
                        bool simplicial_facets = false,
                        double merge_tol = 0.0)
                : PrincipalComponentAnalysis<ScalarT,3>(points),
                  ConvexHull<ScalarT,2,IndexT>(this->template project<2>(points), compute_topology, simplicial_facets, merge_tol),
                  vertices_3d_(this->template reconstruct<2>(this->vertices_))
        {}

        inline const VectorSet<ScalarT,3>& getVertices3D() const { return vertices_3d_; }

        FlatConvexHull3& transform(const Eigen::Ref<const Eigen::Matrix<ScalarT,3,3>> &rotation,
                                   const Eigen::Ref<const Eigen::Matrix<ScalarT,3,1>> &translation)
        {
            if (this->is_empty_) return *this;

            vertices_3d_ = (rotation*vertices_3d_).colwise() + translation;

            Eigen::Matrix<ScalarT,4,3> rec_mat;
            rec_mat.topLeftCorner(3,2) = this->eigenvectors_.leftCols(2);
            rec_mat.topRightCorner(3,1) = this->mean_;
            rec_mat.row(3).setZero();
            rec_mat(3,2) = 1.0;

            Eigen::Matrix<ScalarT,4,4> tform;
            tform.topLeftCorner(3,3) = rotation;
            tform.topRightCorner(3,1) = translation;
            tform.row(3).setZero();
            tform(3,3) = 1.0;

            this->eigenvectors_ = rotation*this->eigenvectors_;
            this->mean_ = rotation*this->mean_ + translation;

            Eigen::Matrix<ScalarT,3,4> proj_mat;
            proj_mat.topLeftCorner(2,3) = this->eigenvectors_.leftCols(2).transpose();
            proj_mat.topRightCorner(2,1).noalias() = -this->eigenvectors_.leftCols(2).transpose()*this->mean_;
            proj_mat.row(2).setZero();
            proj_mat(2,3) = 1.0;

            ConvexHull<ScalarT,2,IndexT>::transform(proj_mat*tform*rec_mat);

            return *this;
        }

        inline FlatConvexHull3& transform(const Eigen::Ref<const Eigen::Matrix<ScalarT,4,4>> &rigid_transform) {
            return transform(rigid_transform.topLeftCorner(3,3), rigid_transform.topRightCorner(3,1));
        }

        inline FlatConvexHull3& transform(const RigidTransform<ScalarT,3> &rigid_transform) {
            return transform(rigid_transform.linear(), rigid_transform.translation());
        }

    protected:
        VectorSet<ScalarT,3> vertices_3d_;
    };

    typedef FlatConvexHull3<float> FlatConvexHull3f;
    typedef FlatConvexHull3<double> FlatConvexHull3d;
}
