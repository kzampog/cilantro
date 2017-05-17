#include <cilantro/convex_polyhedron.hpp>

#include <iostream>

void VtoH(const std::vector<Eigen::Vector3f> &points) {
    size_t dim = 3;
    Eigen::Matrix<realT, Eigen::Dynamic, Eigen::Dynamic> data(Eigen::Map<Eigen::MatrixXf>((float *)points.data(), dim, points.size()).cast<realT>());

    orgQhull::Qhull q("", dim, points.size(), data.data(), "");
    orgQhull::QhullFacetList facets = q.facetList();
    std::cout << facets;

    std::cout << q.area() << std::endl;
}

void HtoV(const std::vector<Eigen::Vector4f> &faces, const Eigen::Vector3f &interior_point) {
    size_t dim = interior_point.size();
    Eigen::Matrix<realT, Eigen::Dynamic, Eigen::Dynamic> data(Eigen::Map<Eigen::MatrixXf>((float *)faces.data(), dim+1, faces.size()).cast<realT>());

    std::vector<coordT> fp(dim);
    Eigen::Matrix<coordT, Eigen::Dynamic, 1>::Map(fp.data(), dim) = interior_point.cast<coordT>();

    orgQhull::Qhull q;
    q.setFeasiblePoint(orgQhull::Coordinates(fp));
    q.qh()->HALFspace = True;
    q.runQhull("", dim+1, faces.size(), data.data(), "");
//    q.runQhull("", dim+1, faces.size(), data.data(), "H");

//    for (k=qh->hull_dim; k--; )
//        *(coordp++)= (*(normp++) / - facet->offset) + *(feasiblep++);

//    orgQhull::QhullPoints points = q.points();
//    std::cout << points;

//    orgQhull::QhullVertexList vertices = q.vertexList();
//    for (auto it = vertices.begin(); it != vertices.end(); ++it) {
//        std::cout << *it << std::endl;
//    }

//    std::cout << vertices;

//    orgQhull::QhullFacetList facets = q.facetList();
//    std::cout << facets;
    orgQhull::QhullFacetList facets = q.facetList();
    std::cout << facets;
    for (auto it = facets.begin(); it != facets.end(); ++it) {
        Eigen::Matrix<coordT, Eigen::Dynamic, 1> hp(dim+1);
        size_t i = 0;
        for (auto hpi = it->hyperplane().begin(); hpi != it->hyperplane().end(); ++hpi) {
            hp(i++) = *hpi;
        }
        hp(dim) = it->hyperplane().offset();
        std::cout << hp.transpose() << std::endl;
        std::cout << it->hyperplane() << std::endl;
    }

    std::cout << q.area() << std::endl;
}
