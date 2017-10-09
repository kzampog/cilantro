#pragma once

#include <vector>
#include <Eigen/Dense>

struct PointCloud {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PointCloud();
    PointCloud(const std::vector<Eigen::Vector3f> &points);
    PointCloud(const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &normals, const std::vector<Eigen::Vector3f> &colors);
    PointCloud(const PointCloud &cloud, const std::vector<size_t> &indices, bool negate = false);

    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Eigen::Vector3f> colors;

    inline size_t size() const { return points.size(); }
    inline bool hasColors() const { return size() > 0 && colors.size() == size(); }
    inline bool hasNormals() const { return size() > 0 && normals.size() == size(); }
    inline bool empty() const { return points.empty(); }
    PointCloud& clear();

    PointCloud& append(const PointCloud &cloud);
    PointCloud& remove(const std::vector<size_t> &indices);

    PointCloud& removeInvalidPoints();
    PointCloud& removeInvalidNormals();
    PointCloud& removeInvalidColors();
    PointCloud& removeInvalidData();

    PointCloud& transform(const Eigen::Ref<const Eigen::Matrix3f> &rotation, const Eigen::Ref<const Eigen::Vector3f> &translation);
    PointCloud& transform(const Eigen::Ref<const Eigen::Matrix4f> &rigid_transform);

    inline Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> > pointsMatrixMap() { return Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)points.data(), 3, points.size()); }
    inline Eigen::Map<const Eigen::Matrix<float,3,Eigen::Dynamic> > pointsMatrixMap() const { return Eigen::Map<const Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)points.data(), 3, points.size()); }

    inline Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> > normalsMatrixMap() { return Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)normals.data(), 3, normals.size()); }
    inline Eigen::Map<const Eigen::Matrix<float,3,Eigen::Dynamic> > normalsMatrixMap() const { return Eigen::Map<const Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)normals.data(), 3, normals.size()); }

    inline Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> > colorsMatrixMap() { return Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)colors.data(), 3, colors.size()); }
    inline Eigen::Map<const Eigen::Matrix<float,3,Eigen::Dynamic> > colorsMatrixMap() const { return Eigen::Map<const Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)colors.data(), 3, colors.size()); }
};
