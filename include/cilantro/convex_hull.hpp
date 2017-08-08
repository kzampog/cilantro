#pragma once

#include <cilantro/convex_polytope.hpp>
#include <cilantro/point_cloud.hpp>

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
bool convexHullFromPoints(const Eigen::Ref<const Eigen::Matrix<InputScalarT,EigenDim,Eigen::Dynamic> > &points,
                          std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &hull_points,
                          std::vector<Eigen::Matrix<OutputScalarT,EigenDim+1,1> > &halfspaces,
                          std::vector<std::vector<size_t> > &facets,
                          std::vector<std::vector<size_t> > &point_neighbor_facets,
                          std::vector<std::vector<size_t> > &facet_neighbor_facets,
                          std::vector<size_t> &hull_point_indices,
                          double &area, double &volume,
                          bool simplicial_facets = true,
                          double merge_tol = 0.0)
{
    size_t num_points = points.cols();
    Eigen::Matrix<double,EigenDim,Eigen::Dynamic> data(points.template cast<double>());

    if (num_points < EigenDim+1) {
        hull_points.clear();
        halfspaces.resize(2);
        halfspaces[0].setZero();
        halfspaces[0](0) = 1.0;
        halfspaces[0](EigenDim) = 1.0;
        halfspaces[1].setZero();
        halfspaces[1](0) = -1.0;
        halfspaces[1](EigenDim) = 1.0;
        facets.clear();
        point_neighbor_facets.clear();
        facet_neighbor_facets.clear();
        hull_point_indices.clear();
        area = 0.0;
        volume = 0.0;
        return false;
    }

    Eigen::Matrix<double,EigenDim,1> mu(data.rowwise().mean());
    if (Eigen::FullPivLU<Eigen::Matrix<double,EigenDim,Eigen::Dynamic> >(data.colwise() - mu).rank() < EigenDim) {
        hull_points.clear();
        halfspaces.resize(2);
        halfspaces[0].setZero();
        halfspaces[0](0) = 1.0;
        halfspaces[0](EigenDim) = 1.0;
        halfspaces[1].setZero();
        halfspaces[1](0) = -1.0;
        halfspaces[1](EigenDim) = 1.0;
        facets.clear();
        point_neighbor_facets.clear();
        facet_neighbor_facets.clear();
        hull_point_indices.clear();
        area = 0.0;
        volume = 0.0;
        return false;
    }

    orgQhull::Qhull qh;
    if (simplicial_facets) qh.qh()->TRIangulate = True;
    qh.qh()->premerge_centrum = merge_tol;
    qh.runQhull("", EigenDim, num_points, data.data(), "");
    qh.defineVertexNeighborFacets();
    orgQhull::QhullFacetList qh_facets = qh.facetList();

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
    for (auto fi = qh_facets.begin(); fi != qh_facets.end(); ++fi)
        if (max_id < fi->id()) max_id = fi->id();
    std::vector<size_t> fid_to_fidx(max_id + 1);
    k = 0;
    for (auto fi = qh_facets.begin(); fi != qh_facets.end(); ++fi)
        fid_to_fidx[fi->id()] = k++;

    // Populate hull points and their indices in the input cloud
    k = 0;
    hull_points.resize(qh.vertexCount());
    point_neighbor_facets.resize(qh.vertexCount());
    hull_point_indices.resize(qh.vertexCount());
    for (auto vi = qh.vertexList().begin(); vi != qh.vertexList().end(); ++vi) {
        size_t i = 0;
        Eigen::Matrix<double,EigenDim,1> v;
        for (auto ci = vi->point().begin(); ci != vi->point().end(); ++ci) {
            v(i++) = *ci;
        }
        hull_points[k] = v.template cast<OutputScalarT>();

        i = 0;
        point_neighbor_facets[k].resize(vi->neighborFacets().size());
        for (auto fi = vi->neighborFacets().begin(); fi != vi->neighborFacets().end(); ++fi) {
            point_neighbor_facets[k][i++] = fid_to_fidx[(*fi).id()];
        }

        hull_point_indices[k] = vi->point().id();
        k++;
    }

    // Populate halfspaces and faces (indices in the hull cloud)
    k = 0;
    halfspaces.resize(qh.facetCount());
    facets.resize(qh_facets.size());
    facet_neighbor_facets.resize(qh_facets.size());
    for (auto fi = qh_facets.begin(); fi != qh_facets.end(); ++fi) {
        size_t i = 0;
        Eigen::Matrix<double,EigenDim+1,1> hp;
        for (auto hpi = fi->hyperplane().begin(); hpi != fi->hyperplane().end(); ++hpi) {
            hp(i++) = *hpi;
        }
        hp(EigenDim) = fi->hyperplane().offset();
        halfspaces[k] = hp.template cast<OutputScalarT>();

        facets[k].resize(fi->vertices().size());
        if (fi->isTopOrient()) {
            i = facets[k].size() - 1;
            for (auto vi = fi->vertices().begin(); vi != fi->vertices().end(); ++vi) {
                facets[k][i--] = vid_to_ptidx[(*vi).id()];
            }
        } else {
            i = 0;
            for (auto vi = fi->vertices().begin(); vi != fi->vertices().end(); ++vi) {
                facets[k][i++] = vid_to_ptidx[(*vi).id()];
            }
        }

        i = 0;
        facet_neighbor_facets[k].resize(fi->neighborFacets().size());
        for (auto nfi = fi->neighborFacets().begin(); nfi != fi->neighborFacets().end(); ++nfi) {
            facet_neighbor_facets[k][i++] = fid_to_fidx[(*nfi).id()];
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
                                 std::vector<std::vector<size_t> > &facets,
                                 std::vector<std::vector<size_t> > &point_neighbor_facets,
                                 std::vector<std::vector<size_t> > &facet_neighbor_facets,
                                 std::vector<size_t> &hull_point_indices,
                                 double &area, double &volume,
                                 bool simplicial_facets = true,
                                 double merge_tol = 0.0)
{
    return convexHullFromPoints<InputScalarT,OutputScalarT,EigenDim>(Eigen::Map<Eigen::Matrix<InputScalarT,EigenDim,Eigen::Dynamic> >((InputScalarT *)points.data(),EigenDim,points.size()), hull_points, halfspaces, facets, point_neighbor_facets, facet_neighbor_facets, hull_point_indices, area, volume, simplicial_facets, merge_tol);
}

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
class ConvexHull : public ConvexPolytope<InputScalarT,OutputScalarT,EigenDim> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ConvexHull(const Eigen::Ref<const Eigen::Matrix<InputScalarT,EigenDim,Eigen::Dynamic> > &points, bool simplicial_facets = true, double merge_tol = 0.0) {
        this->is_bounded_ = true;
        this->is_empty_ = !convexHullFromPoints<InputScalarT,OutputScalarT,EigenDim>(points, this->vertices_, this->halfspaces_, faces_, vertex_neighbor_faces_, face_neighbor_faces_, vertex_point_indices_, this->area_, this->volume_, simplicial_facets, merge_tol);
        if (this->is_empty_) {
            this->interior_point_.setConstant(std::numeric_limits<OutputScalarT>::quiet_NaN());
        } else {
            this->interior_point_ = this->getVerticesMatrixMap().rowwise().mean();
        }
    }
    ConvexHull(const std::vector<Eigen::Matrix<InputScalarT,EigenDim,1> > &points, bool simplicial_facets = true, double merge_tol = 0.0) {
        this->is_bounded_ = true;
        this->is_empty_ = !convexHullFromPoints<InputScalarT,OutputScalarT,EigenDim>(points, this->vertices_, this->halfspaces_, faces_, vertex_neighbor_faces_, face_neighbor_faces_, vertex_point_indices_, this->area_, this->volume_, simplicial_facets, merge_tol);
        if (this->is_empty_) {
            this->interior_point_.setConstant(std::numeric_limits<OutputScalarT>::quiet_NaN());
        } else {
            this->interior_point_ = this->getVerticesMatrixMap().rowwise().mean();
        }
    }
    ConvexHull(const Eigen::Ref<const Eigen::Matrix<InputScalarT,EigenDim+1,Eigen::Dynamic> > &halfspaces, bool simplicial_facets = true, double dist_tol = std::numeric_limits<InputScalarT>::epsilon(), double merge_tol = 0.0) {
        evaluateHalfspaceIntersection<InputScalarT,OutputScalarT,EigenDim>(halfspaces, this->halfspaces_, this->vertices_, this->interior_point_, this->is_bounded_, dist_tol, merge_tol);
        if (this->is_bounded_) {
            this->is_empty_ = !convexHullFromPoints<OutputScalarT,OutputScalarT,EigenDim>(this->vertices_, this->vertices_, this->halfspaces_, faces_, vertex_neighbor_faces_, face_neighbor_faces_, vertex_point_indices_, this->area_, this->volume_, simplicial_facets, merge_tol);
            if (this->is_empty_) {
                this->interior_point_.setConstant(std::numeric_limits<OutputScalarT>::quiet_NaN());
            } else {
                this->interior_point_ = this->getVerticesMatrixMap().rowwise().mean();
            }
        } else {
            this->set_empty_();
        }
    }
    ConvexHull(const std::vector<Eigen::Matrix<InputScalarT,EigenDim+1,1> > &halfspaces, bool simplicial_facets = true, double dist_tol = std::numeric_limits<InputScalarT>::epsilon(), double merge_tol = 0.0) {
        evaluateHalfspaceIntersection<InputScalarT,OutputScalarT,EigenDim>(halfspaces, this->halfspaces_, this->vertices_, this->interior_point_, this->is_bounded_, dist_tol, merge_tol);
        if (this->is_bounded_) {
            this->is_empty_ = !convexHullFromPoints<OutputScalarT,OutputScalarT,EigenDim>(this->vertices_, this->vertices_, this->halfspaces_, faces_, vertex_neighbor_faces_, face_neighbor_faces_, vertex_point_indices_, this->area_, this->volume_, simplicial_facets, merge_tol);
            if (this->is_empty_) {
                this->interior_point_.setConstant(std::numeric_limits<OutputScalarT>::quiet_NaN());
            } else {
                this->interior_point_ = this->getVerticesMatrixMap().rowwise().mean();
            }
        } else {
            this->set_empty_();
        }
    }

    ~ConvexHull() {}

    inline const std::vector<std::vector<size_t> >& getFacetVertexIndices() const { return faces_; }
    inline const std::vector<std::vector<size_t> >& getVertexNeighborFacets() const { return vertex_neighbor_faces_; }
    inline const std::vector<std::vector<size_t> >& getFacetNeighborFacets() const { return face_neighbor_faces_; }
    inline const std::vector<size_t>& getVertexPointIndices() const { return vertex_point_indices_; }

protected:
    std::vector<std::vector<size_t> > faces_;
    std::vector<std::vector<size_t> > vertex_neighbor_faces_;
    std::vector<std::vector<size_t> > face_neighbor_faces_;
    std::vector<size_t> vertex_point_indices_;

};

typedef ConvexHull<float,float,2> ConvexHull2D;
typedef ConvexHull<float,float,3> ConvexHull3D;

class CloudHullFlat : public ConvexHull2D {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CloudHullFlat(const std::vector<Eigen::Vector3f> &points, bool simplicial_facets = true, double merge_tol = 0.0);
    CloudHullFlat(const PointCloud &cloud, bool simplicial_facets = true, double merge_tol = 0.0);

    inline const std::vector<Eigen::Vector3f>& getVertices3D() const { return vertices_3d_; }
private:
    std::vector<Eigen::Vector3f> vertices_3d_;
    void init_(const std::vector<Eigen::Vector3f> &points);
};

class CloudHull : public ConvexHull3D {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using ConvexHull3D::getPointSignedDistancesFromFacets;
    using ConvexHull3D::getInteriorPointIndices;
    CloudHull(const PointCloud &cloud, bool simplicial_facets = true, double merge_tol = 0.0);
    Eigen::MatrixXf getPointSignedDistancesFromFacets(const PointCloud &cloud) const;
    std::vector<size_t> getInteriorPointIndices(const PointCloud &cloud, float offset = 0.0) const;
};
