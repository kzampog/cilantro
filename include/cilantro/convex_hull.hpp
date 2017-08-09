#pragma once

#include <cilantro/convex_polytope.hpp>
#include <cilantro/point_cloud.hpp>

typedef ConvexPolytope<float,float,2> ConvexHull2D;
typedef ConvexPolytope<float,float,3> ConvexHull3D;

class CloudHullFlat : public ConvexHull2D {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CloudHullFlat(const std::vector<Eigen::Vector3f> &points, bool simplicial_facets = true, double merge_tol = 0.0);
    CloudHullFlat(const PointCloud &cloud, bool simplicial_facets = true, double merge_tol = 0.0);

    inline const std::vector<Eigen::Vector3f>& getVertices3D() const { return vertices_3d_; }
private:
    std::vector<Eigen::Vector3f> vertices_3d_;
    void init_(const std::vector<Eigen::Vector3f> &points);
};

class CloudHull : public ConvexHull3D {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using ConvexHull3D::getPointSignedDistancesFromFacets;
    using ConvexHull3D::getInteriorPointIndices;
    CloudHull(const PointCloud &cloud, bool simplicial_facets = true, double merge_tol = 0.0);
    Eigen::MatrixXf getPointSignedDistancesFromFacets(const PointCloud &cloud) const;
    std::vector<size_t> getInteriorPointIndices(const PointCloud &cloud, float offset = 0.0) const;
};
