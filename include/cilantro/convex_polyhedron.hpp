#pragma once

//#include <libqhullcpp/RboxPoints.h>
//#include <libqhullcpp/QhullError.h>
#include <libqhullcpp/QhullQh.h>
#include <libqhullcpp/QhullFacetList.h>
//#include <libqhullcpp/QhullLinkedList.h>
#include <libqhullcpp/Qhull.h>
#include <libqhullcpp/QhullVertexSet.h>

#include <cilantro/point_cloud.hpp>

template <class VectorInT, class FaceVectorOutT>
std::vector<FaceVectorOutT> VtoH(const std::vector<VectorInT> &points) {
    if (points.empty()) return std::vector<FaceVectorOutT>(0);

    size_t dim = points[0].size();
    Eigen::Matrix<realT, Eigen::Dynamic, Eigen::Dynamic> data(
            Eigen::Map<Eigen::Matrix<typename VectorInT::Scalar, Eigen::Dynamic, Eigen::Dynamic> >((typename VectorInT::Scalar *)points.data(), dim, points.size()).template cast<realT>()
    );

    orgQhull::Qhull qh;
    qh.qh()->TRIangulate = True;
    qh.runQhull("", dim, points.size(), data.data(), "");
    orgQhull::QhullFacetList facets = qh.facetList();

    size_t k = 0;
    std::vector<FaceVectorOutT> res(facets.size());
    for (auto fi = facets.begin(); fi != facets.end(); ++fi) {
        Eigen::Matrix<coordT, Eigen::Dynamic, 1> hp(dim+1);
        size_t i = 0;
        for (auto hpi = fi->hyperplane().begin(); hpi != fi->hyperplane().end(); ++hpi) {
            hp(i++) = *hpi;
        }
        hp(dim) = fi->hyperplane().offset();
        res[k++] = hp.cast<typename FaceVectorOutT::Scalar>();

        orgQhull::QhullVertexSet vs = fi->vertices();
        for (auto vi = vs.begin(); vi != vs.end(); ++vi) {
            std::cout << (*vi).point().id() << "  ";
        }
        std::cout << std::endl;

//        for (auto vi = fi->vertices().begin(); vi != fi->vertices().end(); ++vi) {
//            std::cout << "asdf" << std::endl;
//        }

//        398       // Needed by FOREACHvertex_i_
//        399       int vertex_n, vertex_i;
//        400       FOREACHvertex_i_ ((*facet).vertices)
//        401       //facet_vertices.vertices.push_back (qhid_to_pcidx[vertex->id]);
//        402       polygons[dd].vertices[vertex_i] = qhid_to_pcidx[vertex->id];
//        403       ++dd;

    }

    return res;
}

template <class FaceVectorInT, class VectorInT, class VectorOutT>
std::vector<VectorOutT> HtoV(const std::vector<FaceVectorInT> &faces, const VectorInT &interior_point) {
    if (faces.empty()) return std::vector<VectorOutT>(0);

    size_t dim = interior_point.size();

    Eigen::Matrix<realT, Eigen::Dynamic, Eigen::Dynamic> data(
            Eigen::Map<Eigen::Matrix<typename FaceVectorInT::Scalar, Eigen::Dynamic, Eigen::Dynamic> >((typename FaceVectorInT::Scalar *)faces.data(), dim+1, faces.size()).template cast<realT>()
    );

    Eigen::Matrix<coordT, Eigen::Dynamic, 1> feasible_point(interior_point.template cast<coordT>());
    std::vector<coordT> fpv(dim);
    Eigen::Matrix<coordT, Eigen::Dynamic, 1>::Map(fpv.data(), dim) = feasible_point;

    orgQhull::Qhull qh;
    qh.qh()->HALFspace = True;
    qh.setFeasiblePoint(orgQhull::Coordinates(fpv));
    qh.runQhull("", dim+1, faces.size(), data.data(), "");
    orgQhull::QhullFacetList facets = qh.facetList();

    size_t k = 0;
    std::vector<VectorOutT> res(facets.size());
    for (auto fi = facets.begin(); fi != facets.end(); ++fi) {
        Eigen::Matrix<coordT, Eigen::Dynamic, 1> normal(dim);
        size_t i = 0;
        for (auto hpi = fi->hyperplane().begin(); hpi != fi->hyperplane().end(); ++hpi) {
            normal(i++) = *hpi;
        }
//        hp(dim) = fi->hyperplane().offset();
        res[k++] = (-normal/fi->hyperplane().offset() + feasible_point).cast<typename VectorOutT::Scalar>();
    }

    return res;

//    for (k=qh->hull_dim; k--; )
//        *(coordp++)= (*(normp++) / - facet->offset) + *(feasiblep++);

}
