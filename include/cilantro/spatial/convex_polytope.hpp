#pragma once

#include <cilantro/spatial/convex_hull_utilities.hpp>
#include <cilantro/core/space_transformations.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim, typename IndexT = size_t>
    class ConvexPolytope {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;
        typedef IndexT Index;

        enum { Dimension = EigenDim };

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        ConvexPolytope() : dim_(EigenDim), is_empty_(true), is_bounded_(true), area_(0.0), volume_(0.0) {
            halfspaces_.resize(EigenDim+1, 2);
            halfspaces_.setZero();
            halfspaces_(0,0) = 1.0;
            halfspaces_(EigenDim,0) = 1.0;
            halfspaces_(0,1) = -1.0;
            halfspaces_(EigenDim,1) = 1.0;
            interior_point_.setConstant(EigenDim, 1, std::numeric_limits<ScalarT>::quiet_NaN());
        }

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim == Eigen::Dynamic>::type>
        ConvexPolytope(size_t dim = 2) : dim_(dim), is_empty_(true), is_bounded_(true), area_(0.0), volume_(0.0) {
            halfspaces_.resize(dim+1, 2);
            halfspaces_.setZero();
            halfspaces_(0,0) = 1.0;
            halfspaces_(dim,0) = 1.0;
            halfspaces_(0,1) = -1.0;
            halfspaces_(dim,1) = 1.0;
            interior_point_.setConstant(dim, 1, std::numeric_limits<ScalarT>::quiet_NaN());
        }

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        ConvexPolytope(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                       bool compute_topology = false,
                       bool simplicial_facets = false,
                       double merge_tol = 0.0)
                : dim_(EigenDim)
        {
            init_points_(points, compute_topology, simplicial_facets, merge_tol);
        }

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        ConvexPolytope(const ConstHomogeneousVectorSetMatrixMap<ScalarT,EigenDim> &halfspaces,
                       bool compute_topology = false,
                       bool simplicial_facets = false,
                       double merge_tol = 0.0,
                       double dist_tol = std::numeric_limits<ScalarT>::epsilon())
                : dim_(EigenDim)
        {
            init_halfspaces_(halfspaces, compute_topology, simplicial_facets, merge_tol, dist_tol);
        }

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim == Eigen::Dynamic>::type>
        ConvexPolytope(const ConstDataMatrixMap<ScalarT,EigenDim> &input_data,
                       size_t dim,
                       bool compute_topology = false,
                       bool simplicial_facets = false,
                       double merge_tol = 0.0,
                       double dist_tol = std::numeric_limits<ScalarT>::epsilon())
                : dim_(dim)
        {
            if (dim == input_data.rows()) {
                init_points_(input_data, compute_topology, simplicial_facets, merge_tol);
            } else if (dim == input_data.rows() - 1) {
                init_halfspaces_(input_data, compute_topology, simplicial_facets, merge_tol, dist_tol);
            } else {
                *this = ConvexPolytope<ScalarT,EigenDim,IndexT>(dim);
            }
        }

        ~ConvexPolytope() {}

        template <ptrdiff_t Dim = EigenDim>
        typename std::enable_if<Dim != Eigen::Dynamic, ConvexPolytope>::type intersectionWith(const ConvexPolytope &poly,
                                                                                              bool compute_topology = false,
                                                                                              bool simplicial_facets = false,
                                                                                              double merge_tol = 0.0,
                                                                                              double dist_tol = std::numeric_limits<ScalarT>::epsilon()) const
        {
            HomogeneousVectorSet<ScalarT,EigenDim> hs_intersection(EigenDim+1, halfspaces_.cols() + poly.halfspaces_.cols());
            hs_intersection.leftCols(halfspaces_.cols()) = halfspaces_;
            hs_intersection.rightCols(poly.halfspaces_.cols()) = poly.halfspaces_;
            return ConvexPolytope(hs_intersection, compute_topology, simplicial_facets, merge_tol, dist_tol);
        }

        template <ptrdiff_t Dim = EigenDim>
        typename std::enable_if<Dim == Eigen::Dynamic, ConvexPolytope>::type intersectionWith(const ConvexPolytope &poly,
                                                                                              bool compute_topology = false,
                                                                                              bool simplicial_facets = false,
                                                                                              double merge_tol = 0.0,
                                                                                              double dist_tol = std::numeric_limits<ScalarT>::epsilon()) const
        {
            HomogeneousVectorSet<ScalarT,EigenDim> hs_intersection(dim_+1, halfspaces_.cols() + poly.halfspaces_.cols());
            hs_intersection.leftCols(halfspaces_.cols()) = halfspaces_;
            hs_intersection.rightCols(poly.halfspaces_.cols()) = poly.halfspaces_;
            return ConvexPolytope(hs_intersection, dim_, compute_topology, simplicial_facets, merge_tol, dist_tol);
        }

        inline size_t getSpaceDimension() const { return dim_; }

        inline bool isEmpty() const { return is_empty_; }

        inline bool isBounded() const { return is_bounded_; }

        inline double getArea() const { return area_; }

        inline double getVolume() const { return volume_; }

        inline const VectorSet<ScalarT,EigenDim>& getVertices() const { return vertices_; }

        inline const HomogeneousVectorSet<ScalarT,EigenDim>& getFacetHyperplanes() const { return halfspaces_; }

        inline const Vector<ScalarT,EigenDim>& getInteriorPoint() const { return interior_point_; }

        inline bool containsPoint(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &point, ScalarT offset = 0.0) const {
            for (size_t i = 0; i < halfspaces_.cols(); i++) {
                if (point.dot(halfspaces_.col(i).head(dim_)) + halfspaces_(dim_,i) > -offset) return false;
            }
            return true;
        }

        inline Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> getPointSignedDistancesFromFacets(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points) const {
            return (halfspaces_.topRows(dim_).transpose()*points).colwise() + halfspaces_.row(dim_).transpose();
        }

        Eigen::Matrix<bool,1,Eigen::Dynamic> getInteriorPointsIndexMask(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                                                        ScalarT offset = 0.0) const
        {
            Eigen::Matrix<bool,1,Eigen::Dynamic> mask(1,points.cols());
            for (size_t i = 0; i < points.cols(); i++) {
                mask(i) = containsPoint(points.col(i), offset);
            }
            return mask;
        }

        template <typename IdxT = IndexT>
        std::vector<IdxT> getInteriorPointIndices(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                                  ScalarT offset = 0.0) const
        {
            std::vector<IdxT> indices;
            indices.reserve(points.cols());
            for (size_t i = 0; i < points.cols(); i++) {
                if (containsPoint(points.col(i), offset)) indices.emplace_back(i);
            }
            return indices;
        }

        inline const std::vector<std::vector<IndexT>>& getFacetVertexIndices() const { return faces_; }

        inline const std::vector<std::vector<IndexT>>& getVertexNeighborFacets() const { return vertex_neighbor_faces_; }

        inline const std::vector<std::vector<IndexT>>& getFacetNeighborFacets() const { return face_neighbor_faces_; }

        inline const std::vector<IndexT>& getVertexPointIndices() const { return vertex_point_indices_; }

        ConvexPolytope& transform(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,EigenDim>> &rotation,
                                  const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,1>> &translation)
        {
            if (is_empty_) return *this;

            vertices_ = (rotation*vertices_).colwise() + translation;
            interior_point_ = rotation*interior_point_ + translation;

            typename std::conditional<EigenDim == Eigen::Dynamic, Eigen::Matrix<ScalarT,EigenDim,EigenDim>, Eigen::Matrix<ScalarT,EigenDim+1,EigenDim+1>>::type hs_tform(dim_+1,dim_+1);
            hs_tform.topLeftCorner(dim_,dim_) = rotation;
            hs_tform.block(dim_,0,1,dim_).noalias() = -translation.transpose()*rotation;
            hs_tform.col(dim_).setZero();
            hs_tform(dim_,dim_) = 1.0;

            halfspaces_ = hs_tform*halfspaces_;

            return *this;
        }

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        inline ConvexPolytope& transform(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim+1,EigenDim+1>> &tform) {
            return transform(tform.topLeftCorner(EigenDim,EigenDim), tform.topRightCorner(EigenDim,1));
        }

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim == Eigen::Dynamic>::type>
        inline ConvexPolytope& transform(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,EigenDim>> &tform) {
            return transform(tform.topLeftCorner(dim_,dim_), tform.topRightCorner(dim_,1));
        }

        inline ConvexPolytope& transform(const RigidTransform<ScalarT,EigenDim> &tform) {
            return transform(tform.linear(), tform.translation());
        }

        template <class TransformT>
        inline ConvexPolytope transformed(const TransformT &tform) const {
            ConvexPolytope res = *this;
            res.transform(tform);
            return res;
        }

        template <class RotationT, class TranslationT>
        inline ConvexPolytope transformed(const RotationT &rot, const TranslationT trans) const {
            ConvexPolytope res = *this;
            res.transform(rot, trans);
            return res;
        }

    protected:
        // Polytope properties
        size_t dim_;
        bool is_empty_;
        bool is_bounded_;
        double area_;
        double volume_;

        VectorSet<ScalarT,EigenDim> vertices_;
        HomogeneousVectorSet<ScalarT,EigenDim> halfspaces_;
        Vector<ScalarT,EigenDim> interior_point_;

        // Topological properties: only available for bounded (full-dimensional) polytopes
        std::vector<std::vector<IndexT>> faces_;
        std::vector<std::vector<IndexT>> vertex_neighbor_faces_;
        std::vector<std::vector<IndexT>> face_neighbor_faces_;
        std::vector<IndexT> vertex_point_indices_;

        inline void init_points_(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, bool compute_topology, bool simplicial_facets, double merge_tol) {
            is_empty_ = (compute_topology) ? !convexHullFromPoints<ScalarT,EigenDim,IndexT>(points, vertices_, halfspaces_, faces_, vertex_neighbor_faces_, face_neighbor_faces_, vertex_point_indices_, area_, volume_, simplicial_facets, merge_tol)
                                           : !halfspaceIntersectionFromVertices<ScalarT,EigenDim>(points, vertices_, halfspaces_, area_, volume_, true, merge_tol);
            is_bounded_ = true;
            if (is_empty_) {
                interior_point_.setConstant(dim_, 1, std::numeric_limits<ScalarT>::quiet_NaN());
            } else {
                interior_point_ = vertices_.rowwise().mean();
            }
        }

        inline void init_halfspaces_(const ConstHomogeneousVectorSetMatrixMap<ScalarT,EigenDim> &halfspaces, bool compute_topology, bool simplicial_facets, double merge_tol, double dist_tol) {
            is_empty_ = !evaluateHalfspaceIntersection<ScalarT,EigenDim>(halfspaces, halfspaces_, vertices_, interior_point_, is_bounded_, dist_tol, merge_tol);
            if (is_empty_) {
                area_ = 0.0;
                volume_ = 0.0;
            } else if (is_bounded_) {
                if (compute_topology) {
                    is_empty_ = !convexHullFromPoints<ScalarT,EigenDim,IndexT>(vertices_, vertices_, halfspaces_, faces_, vertex_neighbor_faces_, face_neighbor_faces_, vertex_point_indices_, area_, volume_, simplicial_facets, merge_tol);
                    if (is_empty_) {
                        interior_point_.setConstant(dim_, 1, std::numeric_limits<ScalarT>::quiet_NaN());
                    } else {
                        interior_point_ = vertices_.rowwise().mean();
                    }
                } else {
                    computeConvexHullAreaAndVolume<ScalarT,EigenDim>(vertices_, area_, volume_, merge_tol);
                }
            } else {
                area_ = std::numeric_limits<double>::infinity();
                volume_ = std::numeric_limits<double>::infinity();
            }
        }
    };

    typedef ConvexPolytope<float,2> ConvexPolytope2f;
    typedef ConvexPolytope<double,2> ConvexPolytope2d;
    typedef ConvexPolytope<float,3> ConvexPolytope3f;
    typedef ConvexPolytope<double,3> ConvexPolytope3d;
    typedef ConvexPolytope<float,Eigen::Dynamic> ConvexPolytopeXf;
    typedef ConvexPolytope<double,Eigen::Dynamic> ConvexPolytopeXd;

    template <typename ScalarT, ptrdiff_t EigenDim, typename IndexT = size_t>
    using ConvexHull = ConvexPolytope<ScalarT,EigenDim,IndexT>;

    typedef ConvexPolytope<float,2> ConvexHull2f;
    typedef ConvexPolytope<double,2> ConvexHull2d;
    typedef ConvexPolytope<float,3> ConvexHull3f;
    typedef ConvexPolytope<double,3> ConvexHull3d;
    typedef ConvexPolytope<float,Eigen::Dynamic> ConvexHullXf;
    typedef ConvexPolytope<double,Eigen::Dynamic> ConvexHullXd;
}
