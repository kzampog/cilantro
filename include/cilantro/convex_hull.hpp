#pragma once

//#include <libqhullcpp/QhullError.h>
#include <libqhullcpp/QhullQh.h>
#include <libqhullcpp/QhullFacetList.h>
#include <libqhullcpp/QhullPoints.h>
//#include <libqhullcpp/QhullLinkedList.h>
#include <libqhullcpp/Qhull.h>
#include <libqhullcpp/QhullVertexSet.h>

#include <cilantro/point_cloud.hpp>

template <class PointInDataT, class PointOutT, class HalfspaceOutT>
void VtoH(PointInDataT * points, size_t dim, size_t num_points, std::vector<PointOutT> &hull_points, std::vector<HalfspaceOutT> &halfspaces, std::vector<std::vector<size_t> > &faces, std::vector<size_t> &hull_point_indices, bool simplicial_faces = true) {

    Eigen::Matrix<realT, Eigen::Dynamic, Eigen::Dynamic> data(
            Eigen::Map<Eigen::Matrix<PointInDataT, Eigen::Dynamic, Eigen::Dynamic> >((PointInDataT *)points, dim, num_points).template cast<realT>()
    );

    orgQhull::Qhull qh;
    if (simplicial_faces) qh.qh()->TRIangulate = True;
    qh.runQhull("", dim, num_points, data.data(), "");
    orgQhull::QhullFacetList facets = qh.facetList();

    // Establish mapping between hull vertex ids and hull points indices
    size_t max_id = 0;
    for (auto vi = qh.vertexList().begin(); vi != qh.vertexList().end(); ++vi)
        if (max_id < vi->id()) max_id = vi->id();
    std::vector<size_t> vid_to_ptidx(max_id + 1);
    size_t k = 0;
    for (auto vi = qh.vertexList().begin(); vi != qh.vertexList().end(); ++vi) vid_to_ptidx[vi->id()] = k++;

    // Populate hull points and their indices in the input cloud
    k = 0;
    hull_points.resize(qh.vertexCount());
    hull_point_indices.resize(qh.vertexCount());
    for (auto vi = qh.vertexList().begin(); vi != qh.vertexList().end(); ++vi) {
        size_t i = 0;
        Eigen::Matrix<coordT, Eigen::Dynamic, 1> v(dim);
        for (auto ci = vi->point().begin(); ci != vi->point().end(); ++ci) {
            v(i++) = *ci;
        }
        hull_points[k] = v.cast<typename PointOutT::Scalar>();
        hull_point_indices[k] = vi->point().id();
        k++;
    }

    // Populate halfspaces and faces (indices in the hull cloud)
    k = 0;
    halfspaces.resize(facets.size());
    faces.resize(facets.size());
    for (auto fi = facets.begin(); fi != facets.end(); ++fi) {
        size_t i = 0;
        Eigen::Matrix<coordT, Eigen::Dynamic, 1> hp(dim+1);
        for (auto hpi = fi->hyperplane().begin(); hpi != fi->hyperplane().end(); ++hpi) {
            hp(i++) = *hpi;
        }
        hp(dim) = fi->hyperplane().offset();
        halfspaces[k] = hp.cast<typename HalfspaceOutT::Scalar>();

        faces[k].resize(fi->vertices().size());
        if (fi->isTopOrient()) {
            i = faces[k].size() - 1;
            for (auto vi = fi->vertices().begin(); vi != fi->vertices().end(); ++vi) {
                faces[k][i--] = vid_to_ptidx[(*vi).id()];
            }
        } else {
            i = 0;
            for (auto vi = fi->vertices().begin(); vi != fi->vertices().end(); ++vi) {
                faces[k][i++] = vid_to_ptidx[(*vi).id()];
            }
        }

        k++;
    }
}

template <class PointInT, class PointOutT, class HalfspaceOutT>
void VtoH(const std::vector<PointInT> &points, std::vector<PointOutT> &hull_points, std::vector<HalfspaceOutT> &halfspaces, std::vector<std::vector<size_t> > &faces, std::vector<size_t> &hull_point_indices, bool simplicial_faces = true) {
    if (points.empty()) return;
    VtoH<typename PointInT::Scalar,PointOutT,HalfspaceOutT>((typename PointInT::Scalar *)points.data(), points[0].size(), points.size(), hull_points, halfspaces, faces, hull_point_indices, simplicial_faces);
}

template <class HalfspaceInDataT, class PointInT, class PointOutT>
void HtoV(HalfspaceInDataT * halfspaces, size_t dim, size_t num_halfspaces, const PointInT &interior_point, std::vector<PointOutT> &hull_points) {
    Eigen::Matrix<realT, Eigen::Dynamic, Eigen::Dynamic> data(
            Eigen::Map<Eigen::Matrix<HalfspaceInDataT, Eigen::Dynamic, Eigen::Dynamic> >((HalfspaceInDataT *)halfspaces, dim+1, num_halfspaces).template cast<realT>()
    );

    Eigen::Matrix<coordT, Eigen::Dynamic, 1> feasible_point(interior_point.head(dim).template cast<coordT>());
    std::vector<coordT> fpv(dim);
    Eigen::Matrix<coordT, Eigen::Dynamic, 1>::Map(fpv.data(), dim) = feasible_point;

    orgQhull::Qhull qh;
    qh.qh()->HALFspace = True;
    qh.setFeasiblePoint(orgQhull::Coordinates(fpv));
    qh.runQhull("", dim+1, num_halfspaces, data.data(), "");
    orgQhull::QhullFacetList facets = qh.facetList();

    size_t k = 0;
    hull_points.resize(facets.size());
    for (auto fi = facets.begin(); fi != facets.end(); ++fi) {
        Eigen::Matrix<coordT, Eigen::Dynamic, 1> normal(dim);
        size_t i = 0;
        for (auto hpi = fi->hyperplane().begin(); hpi != fi->hyperplane().end(); ++hpi) {
            normal(i++) = *hpi;
        }
        hull_points[k++] = (-normal/fi->hyperplane().offset() + feasible_point).cast<typename PointOutT::Scalar>();
    }
}

template <class HalfspaceInT, class PointInT, class PointOutT>
void HtoV(const std::vector<HalfspaceInT> &halfspaces, const PointInT &interior_point, std::vector<PointOutT> &hull_points) {
    if (halfspaces.empty()) return;
    HtoV<typename HalfspaceInT::Scalar,PointInT,PointOutT>((typename HalfspaceInT::Scalar *)halfspaces.data(), halfspaces[0].size()-1, halfspaces.size(), interior_point, hull_points);
}
