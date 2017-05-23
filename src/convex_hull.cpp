#include <cilantro/convex_hull.hpp>

ConvexHull2D::ConvexHull2D(float * data, size_t dim, size_t num_points, bool simplicial_facets, double merge_tol)
        : ConvexHull(data, dim, num_points, simplicial_facets, merge_tol)
{}

ConvexHull2D::ConvexHull2D(const std::vector<Eigen::Vector2f> &points, bool simplicial_facets, double merge_tol)
        : ConvexHull(points, simplicial_facets, merge_tol)
{}

ConvexHull2D::ConvexHull2D(const std::vector<Eigen::Vector3f> &halfspaces, bool simplicial_facets, double merge_tol)
        : ConvexHull(halfspaces, simplicial_facets, merge_tol)
{}

ConvexHull3D::ConvexHull3D(float * data, size_t dim, size_t num_points, bool simplicial_facets, double merge_tol)
        : ConvexHull(data, dim, num_points, simplicial_facets, merge_tol)
{}

ConvexHull3D::ConvexHull3D(const std::vector<Eigen::Vector3f> &points, bool simplicial_facets, double merge_tol)
        : ConvexHull(points, simplicial_facets, merge_tol)
{}

ConvexHull3D::ConvexHull3D(const std::vector<Eigen::Vector4f> &halfspaces, bool simplicial_facets, double merge_tol)
        : ConvexHull(halfspaces, simplicial_facets, merge_tol)
{}

ConvexHull3D::ConvexHull3D(const PointCloud &cloud, bool simplicial_facets, double merge_tol)
        : ConvexHull(cloud.points, simplicial_facets, merge_tol)
{}
