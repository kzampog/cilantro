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
    size_t dim = 4;
    Eigen::MatrixXd data = Eigen::Map<Eigen::MatrixXf>((float *)faces.data(), dim, faces.size()).cast<realT>();


    std::vector<coordT> fp;
    fp.push_back(0.2);
    fp.push_back(0.2);
    fp.push_back(0.2);

    orgQhull::Qhull q;
    q.setFeasiblePoint(orgQhull::Coordinates(fp));
    q.runQhull("", dim, faces.size(), data.data(), "H");
    orgQhull::QhullVertexList vertices = q.vertexList();
    std::cout << vertices;
}
