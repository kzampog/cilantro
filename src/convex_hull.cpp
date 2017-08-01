#include <cilantro/convex_hull.hpp>
#include <cilantro/principal_component_analysis.hpp>

template class ConvexHull<float,float,2>;
template class ConvexHull<float,float,3>;

CloudHullFlat::CloudHullFlat(const std::vector<Eigen::Vector3f> &points, bool simplicial_facets, realT merge_tol)
        : ConvexHull2D(PrincipalComponentAnalysis3D(points).project<2>(points), simplicial_facets, merge_tol)
{
    init_(points);
}

CloudHullFlat::CloudHullFlat(const PointCloud &cloud, bool simplicial_facets, realT merge_tol)
        : ConvexHull2D(PrincipalComponentAnalysis3D(cloud.points).project<2>(cloud.points), simplicial_facets, merge_tol)
{
    init_(cloud.points);
}

void CloudHullFlat::init_(const std::vector<Eigen::Vector3f> &points) {
    const std::vector<size_t>& vertex_ind = getVertexPointIndices();
    vertices_3d_.resize(vertex_ind.size());
    for (size_t i = 0; i < vertex_ind.size(); i++) {
        vertices_3d_[i] = points[vertex_ind[i]];
    }
}

CloudHull::CloudHull(const PointCloud &cloud, bool simplicial_facets, realT merge_tol)
        : ConvexHull3D(cloud.points, simplicial_facets, merge_tol)
{}

Eigen::MatrixXf CloudHull::getSignedDistancesFromFacets(const PointCloud &cloud) const {
    return getSignedDistancesFromFacets(cloud.points);
}

std::vector<size_t> CloudHull::getInteriorPointIndices(const PointCloud &cloud, float offset) const {
    return getInteriorPointIndices(cloud.points, offset);
}
