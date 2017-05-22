#pragma once

#include <libqhullcpp/Qhull.h>
#include <libqhullcpp/QhullFacetList.h>
#include <libqhullcpp/QhullVertexSet.h>
#include <eigen_quadprog/eiquadprog.hpp>
#include <cilantro/point_cloud.hpp>

class ConvexHull {
public:
    ConvexHull();
    ~ConvexHull();

private:
    size_t dim_;

    std::vector<Eigen::VectorXf> hull_points_;
    std::vector<Eigen::VectorXf> halfspaces_;
    std::vector<std::vector<size_t> > faces_;
    std::vector<size_t> hull_point_indices_;

};

template <class PointInDataT, class PointOutT, class HalfspaceOutT>
void VtoH(PointInDataT * points,
          size_t dim, size_t num_points,
          std::vector<PointOutT> &hull_points,
          std::vector<HalfspaceOutT> &halfspaces,
          std::vector<std::vector<size_t> > &faces,
          std::vector<size_t> &hull_point_indices,
          double &area, double &volume,
          bool simplicial_faces = true,
          realT merge_tol = 0.0)
{
    Eigen::Matrix<realT, Eigen::Dynamic, Eigen::Dynamic> data(Eigen::Map<Eigen::Matrix<PointInDataT, Eigen::Dynamic, Eigen::Dynamic> >(points, dim, num_points).template cast<realT>());

    orgQhull::Qhull qh;
    if (simplicial_faces) qh.qh()->TRIangulate = True;
    qh.qh()->premerge_centrum = merge_tol;
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

    area = qh.area();
    volume = qh.volume();
}

template <class PointInT, class PointOutT, class HalfspaceOutT>
inline void VtoH(const std::vector<PointInT> &points,
                 std::vector<PointOutT> &hull_points,
                 std::vector<HalfspaceOutT> &halfspaces,
                 std::vector<std::vector<size_t> > &faces,
                 std::vector<size_t> &hull_point_indices,
                 double &area, double &volume,
                 bool simplicial_faces = true,
                 realT merge_tol = 0.0)
{
    if (points.empty()) return;
    VtoH<typename PointInT::Scalar,PointOutT,HalfspaceOutT>((typename PointInT::Scalar *)points.data(), points[0].size(), points.size(), hull_points, halfspaces, faces, hull_point_indices, area, volume, simplicial_faces, merge_tol);
}

template <class HalfspaceInDataT, class PointInT, class PointOutT>
void HtoVaux(HalfspaceInDataT * halfspaces,
             size_t dim, size_t num_halfspaces,
             const PointInT &interior_point,
             std::vector<PointOutT> &hull_points,
             realT merge_tol = 0.0)
{
    Eigen::Matrix<realT, Eigen::Dynamic, Eigen::Dynamic> data(Eigen::Map<Eigen::Matrix<HalfspaceInDataT, Eigen::Dynamic, Eigen::Dynamic> >(halfspaces, dim+1, num_halfspaces).template cast<realT>());

    Eigen::Matrix<coordT, Eigen::Dynamic, 1> feasible_point(interior_point.head(dim).template cast<coordT>());
    std::vector<coordT> fpv(dim);
    Eigen::Matrix<coordT, Eigen::Dynamic, 1>::Map(fpv.data(), dim) = feasible_point;

    orgQhull::Qhull qh;
    qh.qh()->HALFspace = True;
    qh.qh()->premerge_centrum = merge_tol;
    qh.setFeasiblePoint(orgQhull::Coordinates(fpv));
    qh.runQhull("", dim+1, num_halfspaces, data.data(), "");
    orgQhull::QhullFacetList facets = qh.facetList();

    // Get hull points from dual facets
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
inline void HtoVaux(const std::vector<HalfspaceInT> &halfspaces,
                    const PointInT &interior_point,
                    std::vector<PointOutT> &hull_points,
                    realT merge_tol = 0.0)
{
    if (halfspaces.empty()) return;
    HtoVaux<typename HalfspaceInT::Scalar,PointInT,PointOutT>((typename HalfspaceInT::Scalar *)halfspaces.data(), halfspaces[0].size()-1, halfspaces.size(), interior_point, hull_points, merge_tol);
}

template <class HalfspaceInDataT, class PointOutT>
void HtoVaux(HalfspaceInDataT * halfspaces,
             size_t dim, size_t num_halfspaces,
             std::vector<PointOutT> &hull_points,
             realT merge_tol = 0.0)
{
    // Find a feasible point
    // Objective
    //Eigen::MatrixXd G(Eigen::MatrixXd::Zero(dim+2,dim+2));
    Eigen::MatrixXd G(Eigen::MatrixXd::Identity(dim+2,dim+2));
    Eigen::VectorXd g0(Eigen::VectorXd::Zero(dim+2));
    g0(dim+1) = -1.0;

    // Equality constraints
    Eigen::MatrixXd CE(dim+2,0);
    Eigen::VectorXd ce0(0);

    // Inequality constraints
    Eigen::MatrixXd ineq_data(Eigen::Map<Eigen::Matrix<HalfspaceInDataT,Eigen::Dynamic,Eigen::Dynamic> >((HalfspaceInDataT *)halfspaces, dim+1, num_halfspaces).template cast<double>());
    Eigen::MatrixXd CI(dim+2,num_halfspaces);
    CI.topRows(dim+1) = -ineq_data;
    CI.row(dim+1) = -Eigen::VectorXd::Ones(num_halfspaces);
    Eigen::VectorXd ci0(Eigen::VectorXd::Zero(num_halfspaces));

    // Optimization
    Eigen::VectorXd x(dim+2);
    solve_quadprog(G, g0, CE, ce0, CI, ci0, x);

    Eigen::VectorXd feasible_point(x.head(dim)/x(dim));
    Eigen::Matrix<HalfspaceInDataT,Eigen::Dynamic,1> interior_point(feasible_point.cast<HalfspaceInDataT>());

    HtoVaux<HalfspaceInDataT,Eigen::Matrix<HalfspaceInDataT,Eigen::Dynamic,1>,PointOutT>(halfspaces, dim, num_halfspaces, interior_point, hull_points, merge_tol);
}

template <class HalfspaceInT, class PointOutT>
inline void HtoVaux(const std::vector<HalfspaceInT> &halfspaces,
                    std::vector<PointOutT> &hull_points,
                    realT merge_tol = 0.0)
{
    if (halfspaces.empty()) return;
    HtoVaux<typename HalfspaceInT::Scalar,PointOutT>((typename HalfspaceInT::Scalar *)halfspaces.data(), halfspaces[0].size()-1, halfspaces.size(), hull_points, merge_tol);
}

template <class HalfspaceInDataT, class PointInT, class PointOutT, class HalfspaceOutT>
inline void HtoV(HalfspaceInDataT * halfspaces,
          size_t dim, size_t num_halfspaces,
          const PointInT &interior_point,
          std::vector<PointOutT> &hull_points,
          std::vector<HalfspaceOutT> &halfspaces_out,
          std::vector<std::vector<size_t> > &faces,
          std::vector<size_t> &hull_point_indices,
          double &area, double &volume,
          bool simplicial_faces = true,
          realT merge_tol = 0.0)
{
    std::vector<PointOutT> hull_points_tmp;
    HtoVaux<HalfspaceInDataT,PointInT,PointOutT>(halfspaces, dim, num_halfspaces, interior_point, hull_points_tmp, merge_tol);
    VtoH<PointOutT,PointOutT,HalfspaceOutT>(hull_points_tmp, hull_points, halfspaces_out, faces, hull_point_indices, area, volume, simplicial_faces, merge_tol);
}

template <class HalfspaceInT, class PointInT, class PointOutT, class HalfspaceOutT>
inline void HtoV(const std::vector<HalfspaceInT> &halfspaces,
                 const PointInT &interior_point,
                 std::vector<PointOutT> &hull_points,
                 std::vector<HalfspaceOutT> &halfspaces_out,
                 std::vector<std::vector<size_t> > &faces,
                 std::vector<size_t> &hull_point_indices,
                 double &area, double &volume,
                 bool simplicial_faces = true,
                 realT merge_tol = 0.0)
{
    std::vector<PointOutT> hull_points_tmp;
    HtoVaux<HalfspaceInT,PointInT,PointOutT>(halfspaces, interior_point, hull_points_tmp, merge_tol);
    VtoH<PointOutT,PointOutT,HalfspaceOutT>(hull_points_tmp, hull_points, halfspaces_out, faces, hull_point_indices, area, volume, simplicial_faces, merge_tol);
}

template <class HalfspaceInDataT, class PointOutT, class HalfspaceOutT>
inline void HtoV(HalfspaceInDataT * halfspaces,
                 size_t dim, size_t num_halfspaces,
                 std::vector<PointOutT> &hull_points,
                 std::vector<HalfspaceOutT> &halfspaces_out,
                 std::vector<std::vector<size_t> > &faces,
                 std::vector<size_t> &hull_point_indices,
                 double &area, double &volume,
                 bool simplicial_faces = true,
                 realT merge_tol = 0.0)
{
    std::vector<PointOutT> hull_points_tmp;
    HtoVaux<HalfspaceInDataT,PointOutT>(halfspaces, dim, num_halfspaces, hull_points_tmp, merge_tol);
    VtoH<PointOutT,PointOutT,HalfspaceOutT>(hull_points_tmp, hull_points, halfspaces_out, faces, hull_point_indices, area, volume, simplicial_faces, merge_tol);
}

template <class HalfspaceInT, class PointOutT, class HalfspaceOutT>
inline void HtoV(const std::vector<HalfspaceInT> &halfspaces,
                 std::vector<PointOutT> &hull_points,
                 std::vector<HalfspaceOutT> &halfspaces_out,
                 std::vector<std::vector<size_t> > &faces,
                 std::vector<size_t> &hull_point_indices,
                 double &area, double &volume,
                 bool simplicial_faces = true,
                 realT merge_tol = 0.0)
{
    std::vector<PointOutT> hull_points_tmp;
    HtoVaux<HalfspaceInT,PointOutT>(halfspaces, hull_points_tmp, merge_tol);
    VtoH<PointOutT,PointOutT,HalfspaceOutT>(hull_points_tmp, hull_points, halfspaces_out, faces, hull_point_indices, area, volume, simplicial_faces, merge_tol);
}
