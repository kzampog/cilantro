#pragma once

#include <cilantro/convex_hull_utilities.hpp>

namespace cilantro {
    template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
    class ConvexPolytope {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ConvexPolytope ()
                : is_empty_(true), is_bounded_(true), area_(0.0), volume_(0.0), halfspaces_(2),
                  interior_point_(Eigen::Matrix<OutputScalarT,EigenDim,1>::Constant(std::numeric_limits<OutputScalarT>::quiet_NaN()))
        {
            halfspaces_[0].setZero();
            halfspaces_[0](0) = 1.0;
            halfspaces_[0](EigenDim) = 1.0;
            halfspaces_[1].setZero();
            halfspaces_[1](0) = -1.0;
            halfspaces_[1](EigenDim) = 1.0;
        }

        template <class Derived, class = typename std::enable_if<std::is_same<typename Derived::Scalar,InputScalarT>::value && Derived::RowsAtCompileTime == EigenDim && Derived::IsRowMajor == 0>::type>
        ConvexPolytope(const Eigen::MatrixBase<Derived> &points, bool compute_topology = false, bool simplicial_facets = false, double merge_tol = 0.0) {
            init_points_(points, compute_topology, simplicial_facets, merge_tol);
        }
        ConvexPolytope(const std::vector<Eigen::Matrix<InputScalarT,EigenDim,1> > &points, bool compute_topology = false, bool simplicial_facets = false, double merge_tol = 0.0) {
            init_points_(Eigen::Map<Eigen::Matrix<InputScalarT,EigenDim,Eigen::Dynamic> >((InputScalarT *)points.data(),EigenDim,points.size()), compute_topology, simplicial_facets, merge_tol);
        }
        template <class Derived, class = typename std::enable_if<std::is_same<typename Derived::Scalar,InputScalarT>::value && Derived::RowsAtCompileTime == EigenDim+1 && Derived::IsRowMajor == 0>::type>
        ConvexPolytope(const Eigen::MatrixBase<Derived> &halfspaces, bool compute_topology = false, bool simplicial_facets = false, double dist_tol = std::numeric_limits<InputScalarT>::epsilon(), double merge_tol = 0.0) {
            init_halfspaces_(halfspaces, compute_topology, simplicial_facets, dist_tol, merge_tol);
        }
        ConvexPolytope(const std::vector<Eigen::Matrix<InputScalarT,EigenDim+1,1> > &halfspaces, bool compute_topology = false, bool simplicial_facets = false, double dist_tol = std::numeric_limits<InputScalarT>::epsilon(), double merge_tol = 0.0) {
            init_halfspaces_(Eigen::Map<Eigen::Matrix<InputScalarT,EigenDim+1,Eigen::Dynamic> >((InputScalarT *)halfspaces.data(),EigenDim+1,halfspaces.size()), compute_topology, simplicial_facets, dist_tol, merge_tol);
        }

        ~ConvexPolytope() {}

        ConvexPolytope intersectionWith(const ConvexPolytope &poly, bool compute_topology = false, bool simplicial_facets = false, double dist_tol = std::numeric_limits<InputScalarT>::epsilon(), double merge_tol = 0.0) const {
            std::vector<Eigen::Matrix<OutputScalarT,EigenDim+1,1> > hs_intersection(halfspaces_);
            hs_intersection.insert(hs_intersection.end(), poly.halfspaces_.begin(), poly.halfspaces_.end());
            return ConvexPolytope(std::move(hs_intersection), compute_topology, simplicial_facets, dist_tol, merge_tol);
        }

        inline bool isEmpty() const { return is_empty_; }
        inline bool isBounded() const { return is_bounded_; }
        inline double getArea() const { return area_; }
        inline double getVolume() const { return volume_; }

        inline const std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> >& getVertices() const { return vertices_; }
        inline const std::vector<Eigen::Matrix<OutputScalarT,EigenDim+1,1> >& getFacetHyperplanes() const { return halfspaces_; }

        inline const Eigen::Matrix<OutputScalarT,EigenDim,1>& getInteriorPoint() const { return interior_point_; }

        inline Eigen::Map<const Eigen::Matrix<OutputScalarT,EigenDim,Eigen::Dynamic> > getVerticesMatrixMap() const {
            return Eigen::Map<const Eigen::Matrix<OutputScalarT,EigenDim,Eigen::Dynamic> >((const OutputScalarT *)vertices_.data(), EigenDim, vertices_.size());
        }

        inline Eigen::Map<const Eigen::Matrix<OutputScalarT,EigenDim+1,Eigen::Dynamic> > getFacetHyperplanesMatrixMap() const {
            return Eigen::Map<const Eigen::Matrix<OutputScalarT,EigenDim+1,Eigen::Dynamic> >((const OutputScalarT *)halfspaces_.data(), EigenDim+1, halfspaces_.size());
        }

        inline bool containsPoint(const Eigen::Matrix<OutputScalarT,EigenDim,1> &point, OutputScalarT offset = 0.0) const {
            for (size_t i = 0; i < halfspaces_.size(); i++) {
                if (point.dot(halfspaces_[i].head(EigenDim)) + halfspaces_[i](EigenDim) > -offset) return false;
            }
            return true;
        }

        inline Eigen::Matrix<OutputScalarT,Eigen::Dynamic,Eigen::Dynamic> getPointSignedDistancesFromFacets(const std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &points) const {
            Eigen::Map<Eigen::Matrix<OutputScalarT,EigenDim,Eigen::Dynamic> > pts_map((OutputScalarT *)points.data(), EigenDim, points.size());
            Eigen::Map<const Eigen::Matrix<OutputScalarT,EigenDim+1,Eigen::Dynamic> > hs_map(getFacetHyperplanesMatrixMap());
            return (hs_map.topRows(EigenDim).transpose()*pts_map).colwise() + hs_map.row(EigenDim).transpose();
        }

        Eigen::Matrix<bool,1,Eigen::Dynamic> getInteriorPointsIndexMask(const std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &points, OutputScalarT offset = 0.0) const {
            Eigen::Matrix<bool,1,Eigen::Dynamic> mask(1,points.size());
            for (size_t i = 0; i < points.size(); i++) {
                mask(i) = containsPoint(points[i], offset);
            }
            return mask;
        }

        std::vector<size_t> getInteriorPointIndices(const std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &points, OutputScalarT offset = 0.0) const {
            std::vector<size_t> indices;
            indices.reserve(points.size());
            for (size_t i = 0; i < points.size(); i++) {
                if (containsPoint(points[i], offset)) indices.emplace_back(i);
            }
            return indices;
        }

        inline const std::vector<std::vector<size_t> >& getFacetVertexIndices() const { return faces_; }
        inline const std::vector<std::vector<size_t> >& getVertexNeighborFacets() const { return vertex_neighbor_faces_; }
        inline const std::vector<std::vector<size_t> >& getFacetNeighborFacets() const { return face_neighbor_faces_; }
        inline const std::vector<size_t>& getVertexPointIndices() const { return vertex_point_indices_; }

        ConvexPolytope& transform(const Eigen::Ref<const Eigen::Matrix<OutputScalarT,EigenDim,EigenDim> > &rotation, const Eigen::Ref<const Eigen::Matrix<OutputScalarT,EigenDim,1> > &translation) {
            if (is_empty_) return *this;

            Eigen::Map<Eigen::Matrix<OutputScalarT,EigenDim,Eigen::Dynamic> > v_map((OutputScalarT *)vertices_.data(), EigenDim, vertices_.size());
            v_map = (rotation*v_map).colwise() + translation;

            interior_point_ = rotation*interior_point_ + translation;

            Eigen::Matrix<OutputScalarT,EigenDim+1,EigenDim+1> hs_tform;
            hs_tform.topLeftCorner(EigenDim,EigenDim) = rotation;
            hs_tform.block(EigenDim,0,1,EigenDim) = -translation.transpose()*rotation;
            hs_tform.col(EigenDim).setZero();
            hs_tform(EigenDim,EigenDim) = 1.0;

            Eigen::Map<Eigen::Matrix<OutputScalarT,EigenDim+1,Eigen::Dynamic> > hs_map((OutputScalarT *)halfspaces_.data(), EigenDim+1, halfspaces_.size());
            hs_map = hs_tform*hs_map;

            return *this;
        }
        ConvexPolytope& transform(const Eigen::Ref<const Eigen::Matrix<OutputScalarT,EigenDim+1,EigenDim+1> > &rigid_transform) {
            return transform(rigid_transform.topLeftCorner(EigenDim,EigenDim), rigid_transform.topRightCorner(EigenDim,1));
        }

    protected:
        // Polytope properties
        bool is_empty_;
        bool is_bounded_;
        double area_;
        double volume_;

        std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > vertices_;
        std::vector<Eigen::Matrix<OutputScalarT,EigenDim+1,1> > halfspaces_;
        Eigen::Matrix<OutputScalarT,EigenDim,1> interior_point_;

        // Topological properties: only available for bounded (full-dimensional) polytopes
        std::vector<std::vector<size_t> > faces_;
        std::vector<std::vector<size_t> > vertex_neighbor_faces_;
        std::vector<std::vector<size_t> > face_neighbor_faces_;
        std::vector<size_t> vertex_point_indices_;

        inline void init_points_(const Eigen::Ref<const Eigen::Matrix<InputScalarT,EigenDim,Eigen::Dynamic> > &points, bool compute_topology, bool simplicial_facets, double merge_tol) {
            is_empty_ = (compute_topology) ? !convexHullFromPoints<InputScalarT,OutputScalarT,EigenDim>(points, vertices_, halfspaces_, faces_, vertex_neighbor_faces_, face_neighbor_faces_, vertex_point_indices_, area_, volume_, simplicial_facets, merge_tol)
                                           : !halfspaceIntersectionFromVertices<InputScalarT,OutputScalarT,EigenDim>(points, vertices_, halfspaces_, area_, volume_, true, merge_tol);
            is_bounded_ = true;
            if (is_empty_) {
                interior_point_.setConstant(std::numeric_limits<OutputScalarT>::quiet_NaN());
            } else {
                interior_point_ = getVerticesMatrixMap().rowwise().mean();
            }
        }

        inline void init_halfspaces_(const Eigen::Ref<const Eigen::Matrix<InputScalarT,EigenDim+1,Eigen::Dynamic> > &halfspaces, bool compute_topology, bool simplicial_facets, double dist_tol, double merge_tol) {
            is_empty_ = !evaluateHalfspaceIntersection<InputScalarT,OutputScalarT,EigenDim>(halfspaces, halfspaces_, vertices_, interior_point_, is_bounded_, dist_tol, merge_tol);
            if (is_empty_) {
                area_ = 0.0;
                volume_ = 0.0;
            } else if (is_bounded_) {
                if (compute_topology) {
                    is_empty_ = !convexHullFromPoints<OutputScalarT,OutputScalarT,EigenDim>(vertices_, vertices_, halfspaces_, faces_, vertex_neighbor_faces_, face_neighbor_faces_, vertex_point_indices_, area_, volume_, simplicial_facets, merge_tol);
                    if (is_empty_) {
                        interior_point_.setConstant(std::numeric_limits<OutputScalarT>::quiet_NaN());
                    } else {
                        interior_point_ = getVerticesMatrixMap().rowwise().mean();
                    }
                } else {
                    computeConvexHullAreaAndVolume<OutputScalarT,EigenDim>(vertices_, area_, volume_, merge_tol);
                }
            } else {
                area_ = std::numeric_limits<double>::infinity();
                volume_ = std::numeric_limits<double>::infinity();
            }
        }

    };

    typedef ConvexPolytope<float,float,2> ConvexPolytope2D;
    typedef ConvexPolytope<float,float,3> ConvexPolytope3D;
}
