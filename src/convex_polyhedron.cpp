#include <cilantro/convex_polyhedron.hpp>

#include <iostream>

void VtoH(const std::vector<Eigen::Vector3f> &points) {
    size_t dim = 3;

    Eigen::MatrixXd data = Eigen::Map<Eigen::MatrixXf>((float *)points.data(), dim, points.size()).cast<realT>();

    orgQhull::Qhull q("", dim, points.size(), data.data(), "");
    orgQhull::QhullFacetList facets = q.facetList();
    std::cout << facets;
}

void HtoV(const std::vector<Eigen::Vector4f> &faces) {

}
