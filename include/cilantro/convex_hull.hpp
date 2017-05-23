#pragma once

#include <libqhullcpp/Qhull.h>
#include <libqhullcpp/QhullFacetList.h>
#include <libqhullcpp/QhullVertexSet.h>
#include <eigen_quadprog/eiquadprog.hpp>
#include <cilantro/point_cloud.hpp>

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
void convexHullFromPoints(InputScalarT * points,
                          size_t num_points,
                          std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &hull_points,
                          std::vector<Eigen::Matrix<OutputScalarT,EigenDim+1,1> > &halfspaces,
                          std::vector<std::vector<size_t> > &faces,
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
    for (auto vi = qh.vertexList().begin(); vi != qh.vertexList().end(); ++vi) vid_to_ptidx[vi->id()] = k++;

    // Populate hull points and their indices in the input cloud
    k = 0;
    hull_points.resize(qh.vertexCount());
    hull_point_indices.resize(qh.vertexCount());
    for (auto vi = qh.vertexList().begin(); vi != qh.vertexList().end(); ++vi) {
        size_t i = 0;
        Eigen::Matrix<coordT,EigenDim,1> v;
        for (auto ci = vi->point().begin(); ci != vi->point().end(); ++ci) {
            v(i++) = *ci;
        }
        hull_points[k] = v.template cast<OutputScalarT>();
        hull_point_indices[k] = vi->point().id();
        k++;
    }

    // Populate halfspaces and faces (indices in the hull cloud)
    k = 0;
    halfspaces.resize(facets.size());
    faces.resize(facets.size());
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

        k++;
    }

    area = qh.area();
    volume = qh.volume();
}

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
inline void convexHullFromPoints(const std::vector<Eigen::Matrix<InputScalarT,EigenDim,1> > &points,
                                 std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &hull_points,
                                 std::vector<Eigen::Matrix<OutputScalarT,EigenDim+1,1> > &halfspaces,
                                 std::vector<std::vector<size_t> > &faces,
                                 std::vector<size_t> &hull_point_indices,
                                 double &area, double &volume,
                                 bool simplicial_faces = true,
                                 realT merge_tol = 0.0)
{
    convexHullFromPoints<InputScalarT,OutputScalarT,EigenDim>((InputScalarT *)points.data(), points.size(), hull_points, halfspaces, faces, hull_point_indices, area, volume, simplicial_faces, merge_tol);
}

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
void convexHullVerticesFromHalfspaces(InputScalarT * halfspaces,
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
}

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
inline void convexHullVerticesFromHalfspaces(const std::vector<Eigen::Matrix<InputScalarT,EigenDim+1,1> > &halfspaces,
                                             const Eigen::Matrix<InputScalarT,EigenDim,1> &interior_point,
                                             std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &hull_points,
                                             realT merge_tol = 0.0)
{
    convexHullVerticesFromHalfspaces<InputScalarT,OutputScalarT,EigenDim>((InputScalarT *)halfspaces.data(), halfspaces.size(), interior_point, hull_points, merge_tol);
}

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
void convexHullVerticesFromHalfspaces(InputScalarT * halfspaces,
                                      size_t num_halfspaces,
                                      std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &hull_points,
                                      realT merge_tol = 0.0)
{
    // Find a feasible point
    // Objective
    //Eigen::MatrixXd G(Eigen::MatrixXd::Zero(EigenDim+2,EigenDim+2));
    Eigen::MatrixXd G(Eigen::MatrixXd::Identity(EigenDim+2,EigenDim+2));
    Eigen::VectorXd g0(Eigen::VectorXd::Zero(EigenDim+2));
    g0(EigenDim+1) = -1.0;

    // Equality constraints
    Eigen::MatrixXd CE(EigenDim+2,0);
    Eigen::VectorXd ce0(0);

    // Inequality constraints
    Eigen::MatrixXd ineq_data(Eigen::Map<Eigen::Matrix<InputScalarT,Eigen::Dynamic,Eigen::Dynamic> >(halfspaces, EigenDim+1, num_halfspaces).template cast<double>());
    Eigen::MatrixXd CI(EigenDim+2,num_halfspaces);
    CI.topRows(EigenDim+1) = -ineq_data;
    CI.row(EigenDim+1) = -Eigen::VectorXd::Ones(num_halfspaces);
    Eigen::VectorXd ci0(Eigen::VectorXd::Zero(num_halfspaces));

    // Optimization
    Eigen::VectorXd x(EigenDim+2);
    solve_quadprog(G, g0, CE, ce0, CI, ci0, x);

    Eigen::Matrix<double,EigenDim,1> feasible_point(x.head(EigenDim)/x(EigenDim));
    Eigen::Matrix<InputScalarT,EigenDim,1> interior_point(feasible_point.template cast<InputScalarT>());

    convexHullVerticesFromHalfspaces<InputScalarT,OutputScalarT,EigenDim>(halfspaces, num_halfspaces, interior_point, hull_points, merge_tol);
}

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
inline void convexHullVerticesFromHalfspaces(const std::vector<Eigen::Matrix<InputScalarT,EigenDim+1,1> > &halfspaces,
                                             std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &hull_points,
                                             realT merge_tol = 0.0)
{
    convexHullVerticesFromHalfspaces<InputScalarT,OutputScalarT,EigenDim>((InputScalarT *)halfspaces.data(), halfspaces.size(), hull_points, merge_tol);
}

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
inline void convexHullFromHalfspaces(InputScalarT * halfspaces,
                                     size_t num_halfspaces,
                                     const Eigen::Matrix<InputScalarT,EigenDim,1> &interior_point,
                                     std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &hull_points,
                                     std::vector<Eigen::Matrix<OutputScalarT,EigenDim+1,1> > &halfspaces_out,
                                     std::vector<std::vector<size_t> > &faces,
                                     std::vector<size_t> &hull_point_indices,
                                     double &area, double &volume,
                                     bool simplicial_faces = true,
                                     realT merge_tol = 0.0)
{
    std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > hull_points_tmp;
    convexHullVerticesFromHalfspaces<InputScalarT,OutputScalarT,EigenDim>(halfspaces, num_halfspaces, interior_point, hull_points_tmp, merge_tol);
    convexHullFromPoints<InputScalarT,OutputScalarT,EigenDim>(hull_points_tmp, hull_points, halfspaces_out, faces, hull_point_indices, area, volume, simplicial_faces, merge_tol);
}

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
inline void convexHullFromHalfspaces(const std::vector<Eigen::Matrix<InputScalarT,EigenDim+1,1> > &halfspaces,
                                     const Eigen::Matrix<InputScalarT,EigenDim,1> &interior_point,
                                     std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &hull_points,
                                     std::vector<Eigen::Matrix<OutputScalarT,EigenDim+1,1> > &halfspaces_out,
                                     std::vector<std::vector<size_t> > &faces,
                                     std::vector<size_t> &hull_point_indices,
                                     double &area, double &volume,
                                     bool simplicial_faces = true,
                                     realT merge_tol = 0.0)
{
    std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > hull_points_tmp;
    convexHullVerticesFromHalfspaces<InputScalarT,OutputScalarT,EigenDim>(halfspaces, interior_point, hull_points_tmp, merge_tol);
    convexHullFromPoints<InputScalarT,OutputScalarT,EigenDim>(hull_points_tmp, hull_points, halfspaces_out, faces, hull_point_indices, area, volume, simplicial_faces, merge_tol);
}

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
inline void convexHullFromHalfspaces(InputScalarT * halfspaces,
                                     size_t num_halfspaces,
                                     std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &hull_points,
                                     std::vector<Eigen::Matrix<OutputScalarT,EigenDim+1,1> > &halfspaces_out,
                                     std::vector<std::vector<size_t> > &faces,
                                     std::vector<size_t> &hull_point_indices,
                                     double &area, double &volume,
                                     bool simplicial_faces = true,
                                     realT merge_tol = 0.0)
{
    std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > hull_points_tmp;
    convexHullVerticesFromHalfspaces<InputScalarT,OutputScalarT,EigenDim>(halfspaces, num_halfspaces, hull_points_tmp, merge_tol);
    convexHullFromPoints<InputScalarT,OutputScalarT,EigenDim>(hull_points_tmp, hull_points, halfspaces_out, faces, hull_point_indices, area, volume, simplicial_faces, merge_tol);
}

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
inline void convexHullFromHalfspaces(const std::vector<Eigen::Matrix<InputScalarT,EigenDim+1,1> > &halfspaces,
                                     std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &hull_points,
                                     std::vector<Eigen::Matrix<OutputScalarT,EigenDim+1,1> > &halfspaces_out,
                                     std::vector<std::vector<size_t> > &faces,
                                     std::vector<size_t> &hull_point_indices,
                                     double &area, double &volume,
                                     bool simplicial_faces = true,
                                     realT merge_tol = 0.0)
{
    std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > hull_points_tmp;
    convexHullVerticesFromHalfspaces<InputScalarT,OutputScalarT,EigenDim>(halfspaces, hull_points_tmp, merge_tol);
    convexHullFromPoints<InputScalarT,OutputScalarT,EigenDim>(hull_points_tmp, hull_points, halfspaces_out, faces, hull_point_indices, area, volume, simplicial_faces, merge_tol);
}

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
class ConvexHull {
public:
    ConvexHull(InputScalarT * data, size_t dim, size_t num_points, bool simplicial_facets = true, double merge_tol = 0.0) {
        if (dim == EigenDim)
            convexHullFromPoints<InputScalarT,OutputScalarT,EigenDim>(data, num_points, hull_points_, halfspaces_, faces_, hull_point_indices_, area_, volume_, simplicial_facets, merge_tol);
        else
            convexHullFromHalfspaces<InputScalarT,OutputScalarT,EigenDim>(data, num_points, hull_points_, halfspaces_, faces_, hull_point_indices_, area_, volume_, simplicial_facets, merge_tol);
    }
    ConvexHull(const std::vector<Eigen::Matrix<InputScalarT,EigenDim,1> > &points, bool simplicial_facets = true, double merge_tol = 0.0) {
        convexHullFromPoints<InputScalarT,OutputScalarT,EigenDim>(points, hull_points_, halfspaces_, faces_, hull_point_indices_, area_, volume_, simplicial_facets, merge_tol);
    }
    ConvexHull(const std::vector<Eigen::Matrix<InputScalarT,EigenDim+1,1> > &halfspaces, bool simplicial_facets = true, double merge_tol = 0.0) {
        convexHullFromHalfspaces<InputScalarT,OutputScalarT,EigenDim>(halfspaces, hull_points_, halfspaces_, faces_, hull_point_indices_, area_, volume_, simplicial_facets, merge_tol);
    }

    ~ConvexHull() {}

    inline const std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> >& getVertices() const { return hull_points_; }
    inline const std::vector<Eigen::Matrix<OutputScalarT,EigenDim+1,1> >& getFacetHyperplanes() const { return halfspaces_; }
    inline const std::vector<std::vector<size_t> >& getFacetVertexIndices() const { return faces_; }
    inline const std::vector<size_t>& getVertexPointIndices() const { return hull_point_indices_; }

    inline double getArea() const { return area_; }
    inline double getVolume() const { return volume_; }

    bool isPointInHull(const Eigen::Matrix<OutputScalarT,EigenDim,1> &point) const {
        for (size_t i = 0; i < halfspaces_.size(); i++) {
            if (point.dot(halfspaces_[i].head(3)) + halfspaces_[i](EigenDim) > 0.0) return false;
        }
        return true;
    }

    std::vector<size_t> getInteriorPointIndices(const std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &points) const {
        std::vector<size_t> indices;
        indices.reserve(points.size());
        // TODO
        return indices;
    }

private:
    std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > hull_points_;
    std::vector<Eigen::Matrix<OutputScalarT,EigenDim+1,1> > halfspaces_;
    std::vector<std::vector<size_t> > faces_;
    std::vector<size_t> hull_point_indices_;
    double area_;
    double volume_;
};

class ConvexHull2D : public ConvexHull<float,float,2> {
public:
    ConvexHull2D(float * data, size_t dim, size_t num_points, bool simplicial_facets = true, double merge_tol = 0.0);
    ConvexHull2D(const std::vector<Eigen::Vector2f> &points, bool simplicial_facets = true, double merge_tol = 0.0);
    ConvexHull2D(const std::vector<Eigen::Vector3f> &halfspaces, bool simplicial_facets = true, double merge_tol = 0.0);
};

class ConvexHull3D : public ConvexHull<float,float,3> {
public:
    ConvexHull3D(float * data, size_t dim, size_t num_points, bool simplicial_facets = true, double merge_tol = 0.0);
    ConvexHull3D(const std::vector<Eigen::Vector3f> &points, bool simplicial_facets = true, double merge_tol = 0.0);
    ConvexHull3D(const std::vector<Eigen::Vector4f> &halfspaces, bool simplicial_facets = true, double merge_tol = 0.0);
    ConvexHull3D(const PointCloud &cloud, bool simplicial_facets = true, double merge_tol = 0.0);
};
