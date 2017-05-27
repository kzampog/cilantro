#pragma once

#include <libqhullcpp/Qhull.h>
#include <libqhullcpp/QhullFacetSet.h>
#include <libqhullcpp/QhullFacetList.h>
#include <libqhullcpp/QhullVertexSet.h>
#include <eigen_quadprog/eiquadprog.hpp>
#include <cilantro/point_cloud.hpp>

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
bool convexHullFromPoints(InputScalarT * points,
                          size_t num_points,
                          std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &hull_points,
                          std::vector<Eigen::Matrix<OutputScalarT,EigenDim+1,1> > &halfspaces,
                          std::vector<std::vector<size_t> > &faces,
                          std::vector<std::vector<size_t> > &point_neighbor_faces,
                          std::vector<std::vector<size_t> > &face_neighbor_faces,
                          std::vector<size_t> &hull_point_indices,
                          double &area, double &volume,
                          bool simplicial_faces = true,
                          realT merge_tol = 0.0)
{
    Eigen::Matrix<realT,Eigen::Dynamic,Eigen::Dynamic> data(Eigen::Map<Eigen::Matrix<InputScalarT,Eigen::Dynamic,Eigen::Dynamic> >(points, EigenDim, num_points).template cast<realT>());

    orgQhull::Qhull qh;
    if (simplicial_faces) qh.qh()->TRIangulate = True;
    qh.qh()->premerge_centrum = merge_tol;
    qh.runQhull("", EigenDim, num_points, data.data(), "");
    orgQhull::QhullFacetList facets = qh.facetList();

    // Establish mapping between hull vertex ids and hull points indices
    size_t max_id = 0;
    for (auto vi = qh.vertexList().begin(); vi != qh.vertexList().end(); ++vi)
        if (max_id < vi->id()) max_id = vi->id();
    std::vector<size_t> vid_to_ptidx(max_id + 1);
    size_t k = 0;
    for (auto vi = qh.vertexList().begin(); vi != qh.vertexList().end(); ++vi)
        vid_to_ptidx[vi->id()] = k++;

    // Establish mapping between face ids and face indices
    max_id = 0;
    for (auto fi = facets.begin(); fi != facets.end(); ++fi)
        if (max_id < fi->id()) max_id = fi->id();
    std::vector<size_t> fid_to_fidx(max_id + 1);
    k = 0;
    for (auto fi = facets.begin(); fi != facets.end(); ++fi)
        fid_to_fidx[fi->id()] = k++;

    // Populate hull points and their indices in the input cloud
    k = 0;
    hull_points.resize(qh.vertexCount());
    point_neighbor_faces.resize(qh.vertexCount());
    hull_point_indices.resize(qh.vertexCount());
    for (auto vi = qh.vertexList().begin(); vi != qh.vertexList().end(); ++vi) {
        size_t i = 0;
        Eigen::Matrix<coordT,EigenDim,1> v;
        for (auto ci = vi->point().begin(); ci != vi->point().end(); ++ci) {
            v(i++) = *ci;
        }
        hull_points[k] = v.template cast<OutputScalarT>();

        i = 0;
        point_neighbor_faces[k].resize(vi->neighborFacets().size());
        for (auto fi = vi->neighborFacets().begin(); fi != vi->neighborFacets().end(); ++fi) {
            point_neighbor_faces[k][i++] = fid_to_fidx[(*fi).id()];
        }

        hull_point_indices[k] = vi->point().id();
        k++;
    }

    // Populate halfspaces and faces (indices in the hull cloud)
    k = 0;
    halfspaces.resize(facets.size());
    faces.resize(facets.size());
    face_neighbor_faces.resize(facets.size());
    for (auto fi = facets.begin(); fi != facets.end(); ++fi) {
        size_t i = 0;
        Eigen::Matrix<coordT,EigenDim+1,1> hp;
        for (auto hpi = fi->hyperplane().begin(); hpi != fi->hyperplane().end(); ++hpi) {
            hp(i++) = *hpi;
        }
        hp(EigenDim) = fi->hyperplane().offset();
        halfspaces[k] = hp.template cast<OutputScalarT>();

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

        i = 0;
        face_neighbor_faces[k].resize(fi->neighborFacets().size());
        for (auto nfi = fi->neighborFacets().begin(); nfi != fi->neighborFacets().end(); ++nfi) {
            face_neighbor_faces[k][i++] = fid_to_fidx[(*nfi).id()];
        }

        k++;
    }

    area = qh.area();
    volume = qh.volume();

    return true;
}

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
inline bool convexHullFromPoints(const std::vector<Eigen::Matrix<InputScalarT,EigenDim,1> > &points,
                                 std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &hull_points,
                                 std::vector<Eigen::Matrix<OutputScalarT,EigenDim+1,1> > &halfspaces,
                                 std::vector<std::vector<size_t> > &faces,
                                 std::vector<std::vector<size_t> > &point_neighbor_faces,
                                 std::vector<std::vector<size_t> > &face_neighbor_faces,
                                 std::vector<size_t> &hull_point_indices,
                                 double &area, double &volume,
                                 bool simplicial_faces = true,
                                 realT merge_tol = 0.0)
{
    return convexHullFromPoints<InputScalarT,OutputScalarT,EigenDim>((InputScalarT *)points.data(), points.size(), hull_points, halfspaces, faces, point_neighbor_faces, face_neighbor_faces, hull_point_indices, area, volume, simplicial_faces, merge_tol);
}

template <typename ScalarT, ptrdiff_t EigenDim>
bool findFeasiblePointInHalfspaceIntersection(ScalarT * halfspaces,
                                              size_t num_halfspaces,
                                              Eigen::Matrix<ScalarT,EigenDim,1> &feasible_point)
{
    // Objective
    //Eigen::MatrixXd G(Eigen::MatrixXd::Zero(EigenDim+2,EigenDim+2));
    Eigen::MatrixXd G(Eigen::MatrixXd::Identity(EigenDim+2,EigenDim+2));
    Eigen::VectorXd g0(Eigen::VectorXd::Zero(EigenDim+2));
    g0(EigenDim+1) = -1.0;

    // Equality constraints
    Eigen::MatrixXd CE(EigenDim+2,0);
    Eigen::VectorXd ce0(0);

    // Inequality constraints
    Eigen::MatrixXd ineq_data(Eigen::Map<Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> >(halfspaces, EigenDim+1, num_halfspaces).template cast<double>());
    Eigen::MatrixXd CI(EigenDim+2,num_halfspaces);
    CI.topRows(EigenDim+1) = -ineq_data;
    CI.row(EigenDim+1) = -Eigen::VectorXd::Ones(num_halfspaces);
    Eigen::VectorXd ci0(Eigen::VectorXd::Zero(num_halfspaces));

    // Optimization
    Eigen::VectorXd x(EigenDim+2);
    double val = solve_quadprog(G, g0, CE, ce0, CI, ci0, x);

    if (std::isinf(val) || std::abs(x(EigenDim)) < 1e-10 || x.array().isNaN().any() || x.array().isInf().any())
        return false;

    Eigen::Matrix<double,EigenDim,1> feasible_point_d(x.head(EigenDim)/x(EigenDim));
    feasible_point = feasible_point_d.template cast<ScalarT>();

    return true;
}

template <typename ScalarT, ptrdiff_t EigenDim>
inline bool findFeasiblePointInHalfspaceIntersection(const std::vector<Eigen::Matrix<ScalarT,EigenDim+1,1> > &halfspaces,
                                                     Eigen::Matrix<ScalarT,EigenDim,1> &feasible_point)
{
    return findFeasiblePointInHalfspaceIntersection<ScalarT,EigenDim>((ScalarT *)halfspaces.data(), halfspaces.size(), feasible_point);
}

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
bool convexHullVerticesFromHalfspaces(InputScalarT * halfspaces,
                                      size_t num_halfspaces,
                                      const Eigen::Matrix<InputScalarT,EigenDim,1> &interior_point,
                                      std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &hull_points,
                                      realT merge_tol = 0.0)
{
    Eigen::Matrix<realT,Eigen::Dynamic,Eigen::Dynamic> data(Eigen::Map<Eigen::Matrix<InputScalarT,Eigen::Dynamic,Eigen::Dynamic> >(halfspaces, EigenDim+1, num_halfspaces).template cast<realT>());

    Eigen::Matrix<coordT,EigenDim,1> feasible_point(interior_point.template cast<coordT>());
    std::vector<coordT> fpv(EigenDim);
    Eigen::Matrix<coordT,EigenDim,1>::Map(fpv.data()) = feasible_point;

    orgQhull::Qhull qh;
    qh.qh()->HALFspace = True;
    qh.qh()->premerge_centrum = merge_tol;
    qh.setFeasiblePoint(orgQhull::Coordinates(fpv));
    qh.runQhull("", EigenDim+1, num_halfspaces, data.data(), "");
    orgQhull::QhullFacetList facets = qh.facetList();

    // Get hull points from dual facets
    size_t k = 0;
    hull_points.resize(facets.size());
    for (auto fi = facets.begin(); fi != facets.end(); ++fi) {
        Eigen::Matrix<coordT,EigenDim,1> normal;
        size_t i = 0;
        for (auto hpi = fi->hyperplane().begin(); hpi != fi->hyperplane().end(); ++hpi) {
            normal(i++) = *hpi;
        }
        hull_points[k++] = (-normal/fi->hyperplane().offset() + feasible_point).template cast<OutputScalarT>();
    }

    return true;
}

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
inline bool convexHullVerticesFromHalfspaces(const std::vector<Eigen::Matrix<InputScalarT,EigenDim+1,1> > &halfspaces,
                                             const Eigen::Matrix<InputScalarT,EigenDim,1> &interior_point,
                                             std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &hull_points,
                                             realT merge_tol = 0.0)
{
    return convexHullVerticesFromHalfspaces<InputScalarT,OutputScalarT,EigenDim>((InputScalarT *)halfspaces.data(), halfspaces.size(), interior_point, hull_points, merge_tol);
}

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
inline bool convexHullVerticesFromHalfspaces(InputScalarT * halfspaces,
                                             size_t num_halfspaces,
                                             std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &hull_points,
                                             realT merge_tol = 0.0)
{

    Eigen::Matrix<InputScalarT,EigenDim,1> interior_point;
    if (findFeasiblePointInHalfspaceIntersection<InputScalarT,EigenDim>(halfspaces, num_halfspaces, interior_point))
        return convexHullVerticesFromHalfspaces<InputScalarT,OutputScalarT,EigenDim>(halfspaces, num_halfspaces, interior_point, hull_points, merge_tol);
    return false;
}

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
inline bool convexHullVerticesFromHalfspaces(const std::vector<Eigen::Matrix<InputScalarT,EigenDim+1,1> > &halfspaces,
                                             std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &hull_points,
                                             realT merge_tol = 0.0)
{
    return convexHullVerticesFromHalfspaces<InputScalarT,OutputScalarT,EigenDim>((InputScalarT *)halfspaces.data(), halfspaces.size(), hull_points, merge_tol);
}

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
inline bool convexHullFromHalfspaces(InputScalarT * halfspaces,
                                     size_t num_halfspaces,
                                     const Eigen::Matrix<InputScalarT,EigenDim,1> &interior_point,
                                     std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &hull_points,
                                     std::vector<Eigen::Matrix<OutputScalarT,EigenDim+1,1> > &halfspaces_out,
                                     std::vector<std::vector<size_t> > &faces,
                                     std::vector<std::vector<size_t> > &point_neighbor_faces,
                                     std::vector<std::vector<size_t> > &face_neighbor_faces,
                                     std::vector<size_t> &hull_point_indices,
                                     double &area, double &volume,
                                     bool simplicial_faces = true,
                                     realT merge_tol = 0.0)
{
    std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > hull_points_tmp;
    if (convexHullVerticesFromHalfspaces<InputScalarT,OutputScalarT,EigenDim>(halfspaces, num_halfspaces, interior_point, hull_points_tmp, merge_tol))
        return convexHullFromPoints<InputScalarT,OutputScalarT,EigenDim>(hull_points_tmp, hull_points, halfspaces_out, faces, point_neighbor_faces, face_neighbor_faces, hull_point_indices, area, volume, simplicial_faces, merge_tol);
    return false;
}

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
inline bool convexHullFromHalfspaces(const std::vector<Eigen::Matrix<InputScalarT,EigenDim+1,1> > &halfspaces,
                                     const Eigen::Matrix<InputScalarT,EigenDim,1> &interior_point,
                                     std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &hull_points,
                                     std::vector<Eigen::Matrix<OutputScalarT,EigenDim+1,1> > &halfspaces_out,
                                     std::vector<std::vector<size_t> > &faces,
                                     std::vector<std::vector<size_t> > &point_neighbor_faces,
                                     std::vector<std::vector<size_t> > &face_neighbor_faces,
                                     std::vector<size_t> &hull_point_indices,
                                     double &area, double &volume,
                                     bool simplicial_faces = true,
                                     realT merge_tol = 0.0)
{
    std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > hull_points_tmp;
    if (convexHullVerticesFromHalfspaces<InputScalarT,OutputScalarT,EigenDim>(halfspaces, interior_point, hull_points_tmp, merge_tol))
        return convexHullFromPoints<InputScalarT,OutputScalarT,EigenDim>(hull_points_tmp, hull_points, halfspaces_out, faces, point_neighbor_faces, face_neighbor_faces, hull_point_indices, area, volume, simplicial_faces, merge_tol);
    return false;
}

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
inline bool convexHullFromHalfspaces(InputScalarT * halfspaces,
                                     size_t num_halfspaces,
                                     std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &hull_points,
                                     std::vector<Eigen::Matrix<OutputScalarT,EigenDim+1,1> > &halfspaces_out,
                                     std::vector<std::vector<size_t> > &faces,
                                     std::vector<std::vector<size_t> > &point_neighbor_faces,
                                     std::vector<std::vector<size_t> > &face_neighbor_faces,
                                     std::vector<size_t> &hull_point_indices,
                                     double &area, double &volume,
                                     bool simplicial_faces = true,
                                     realT merge_tol = 0.0)
{
    std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > hull_points_tmp;
    if (convexHullVerticesFromHalfspaces<InputScalarT,OutputScalarT,EigenDim>(halfspaces, num_halfspaces, hull_points_tmp, merge_tol))
        return convexHullFromPoints<InputScalarT,OutputScalarT,EigenDim>(hull_points_tmp, hull_points, halfspaces_out, faces, point_neighbor_faces, face_neighbor_faces, hull_point_indices, area, volume, simplicial_faces, merge_tol);
    return false;
}

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
inline bool convexHullFromHalfspaces(const std::vector<Eigen::Matrix<InputScalarT,EigenDim+1,1> > &halfspaces,
                                     std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &hull_points,
                                     std::vector<Eigen::Matrix<OutputScalarT,EigenDim+1,1> > &halfspaces_out,
                                     std::vector<std::vector<size_t> > &faces,
                                     std::vector<std::vector<size_t> > &point_neighbor_faces,
                                     std::vector<std::vector<size_t> > &face_neighbor_faces,
                                     std::vector<size_t> &hull_point_indices,
                                     double &area, double &volume,
                                     bool simplicial_faces = true,
                                     realT merge_tol = 0.0)
{
    std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > hull_points_tmp;
    if (convexHullVerticesFromHalfspaces<InputScalarT,OutputScalarT,EigenDim>(halfspaces, hull_points_tmp, merge_tol))
        return convexHullFromPoints<InputScalarT,OutputScalarT,EigenDim>(hull_points_tmp, hull_points, halfspaces_out, point_neighbor_faces, face_neighbor_faces, faces, hull_point_indices, area, volume, simplicial_faces, merge_tol);
    return false;
}

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
class ConvexHull {
public:
    ConvexHull(InputScalarT * data, size_t dim, size_t num_points, bool simplicial_facets = true, double merge_tol = 0.0) {
        if (dim == EigenDim)
            is_empty_ = !convexHullFromPoints<InputScalarT,OutputScalarT,EigenDim>(data, num_points, hull_points_, halfspaces_, faces_, point_neighbor_faces_, face_neighbor_faces_, hull_point_indices_, area_, volume_, simplicial_facets, merge_tol);
        else
            is_empty_ = !convexHullFromHalfspaces<InputScalarT,OutputScalarT,EigenDim>(data, num_points, hull_points_, halfspaces_, faces_, point_neighbor_faces_, face_neighbor_faces_, hull_point_indices_, area_, volume_, simplicial_facets, merge_tol);
        init_();
    }
    ConvexHull(const std::vector<Eigen::Matrix<InputScalarT,EigenDim,1> > &points, bool simplicial_facets = true, double merge_tol = 0.0) {
        is_empty_ = !convexHullFromPoints<InputScalarT,OutputScalarT,EigenDim>(points, hull_points_, halfspaces_, faces_, point_neighbor_faces_, face_neighbor_faces_, hull_point_indices_, area_, volume_, simplicial_facets, merge_tol);
        init_();
    }
    ConvexHull(const std::vector<Eigen::Matrix<InputScalarT,EigenDim+1,1> > &halfspaces, bool simplicial_facets = true, double merge_tol = 0.0) {
        is_empty_ = !convexHullFromHalfspaces<InputScalarT,OutputScalarT,EigenDim>(halfspaces, hull_points_, halfspaces_, faces_, point_neighbor_faces_, face_neighbor_faces_, hull_point_indices_, area_, volume_, simplicial_facets, merge_tol);
        init_();
    }

    ~ConvexHull() {}

    inline const std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> >& getVertices() const { return hull_points_; }
    inline const std::vector<Eigen::Matrix<OutputScalarT,EigenDim+1,1> >& getFacetHyperplanes() const { return halfspaces_; }
    inline const std::vector<std::vector<size_t> >& getFacetVertexIndices() const { return faces_; }
    inline const std::vector<std::vector<size_t> >& getVertexNeighborFacets() const { return point_neighbor_faces_; }
    inline const std::vector<std::vector<size_t> >& getFacetNeighborFacets() const { return face_neighbor_faces_; }
    inline const std::vector<size_t>& getVertexPointIndices() const { return hull_point_indices_; }

    inline double getArea() const { return area_; }
    inline double getVolume() const { return volume_; }

    bool isPointInHull(const Eigen::Matrix<OutputScalarT,EigenDim,1> &point) const {
        for (size_t i = 0; i < halfspaces_.size(); i++) {
            if (point.dot(halfspaces_[i].head(EigenDim)) + halfspaces_[i](EigenDim) > 0.0) return false;
        }
        return true;
    }

    std::vector<OutputScalarT> getSignedDistancesFromFacets(const std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &points) const {
        std::vector<OutputScalarT> distances(points.size());
        Eigen::Map<Eigen::Matrix<OutputScalarT,Eigen::Dynamic,Eigen::Dynamic> > map((OutputScalarT *)points.data(), EigenDim, points.size());
        Eigen::Matrix<OutputScalarT,1,Eigen::Dynamic>::Map(distances.data(), distances.size()) = (halfspace_normals_*map).colwise() + halfspace_offsets_;
        return distances;
    }

    std::vector<size_t> getInteriorPointIndices(const std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &points) const {
        std::vector<OutputScalarT> distances = getSignedDistancesFromFacets(points);
        std::vector<size_t> indices;
        indices.reserve(points.size());
        for (size_t i = 0; i < points.size(); i++) {
            if (distances[i] <= 0.0) indices.push_back(i);
        }
        return indices;
    }

    bool isEmpty() const { return is_empty_; }

private:
    Eigen::Matrix<OutputScalarT,Eigen::Dynamic,EigenDim> halfspace_normals_;
    Eigen::Matrix<OutputScalarT,Eigen::Dynamic,1> halfspace_offsets_;
    bool is_empty_;

    std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > hull_points_;
    std::vector<Eigen::Matrix<OutputScalarT,EigenDim+1,1> > halfspaces_;
    std::vector<std::vector<size_t> > faces_;
    std::vector<std::vector<size_t> > point_neighbor_faces_;
    std::vector<std::vector<size_t> > face_neighbor_faces_;
    std::vector<size_t> hull_point_indices_;
    double area_;
    double volume_;

    void init_() {
        Eigen::Map<Eigen::Matrix<OutputScalarT,Eigen::Dynamic,Eigen::Dynamic> > map((OutputScalarT *)halfspaces_.data(), EigenDim+1, halfspaces_.size());
        halfspace_normals_ = map.topRows(EigenDim).transpose();
        halfspace_offsets_ = map.row(EigenDim);
    }
};

typedef ConvexHull<float,float,2> ConvexHull2D;
typedef ConvexHull<float,float,3> ConvexHull3D;

class CloudHullFlat : public ConvexHull2D {
public:
    CloudHullFlat(const std::vector<Eigen::Vector3f> &points, bool simplicial_facets = true, double merge_tol = 0.0);
    CloudHullFlat(const PointCloud &cloud, bool simplicial_facets = true, double merge_tol = 0.0);

    inline const std::vector<Eigen::Vector3f>& getVertices3D() const { return vertices_3d_; }
private:
    std::vector<Eigen::Vector3f> vertices_3d_;
    void init_(const std::vector<Eigen::Vector3f> &points);
};

class CloudHull : public ConvexHull3D {
public:
    CloudHull(const PointCloud &cloud, bool simplicial_facets = true, double merge_tol = 0.0);
};
