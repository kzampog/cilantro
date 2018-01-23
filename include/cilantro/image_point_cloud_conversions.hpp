#pragma once

#include <cilantro/point_cloud.hpp>
#include <pangolin/pangolin.h>

namespace cilantro {
    void depthImageToPoints(const pangolin::Image<unsigned short> &depth_img,
                            const Eigen::Matrix3f &intr,
                            VectorSet<float,3> &points,
                            bool keep_invalid = false);

    inline void depthImageToPointCloud(const pangolin::Image<unsigned short> &depth_img,
                                       const Eigen::Matrix3f &intr,
                                       PointCloud<float,3> &cloud,
                                       bool keep_invalid = false)
    {
        depthImageToPoints(depth_img, intr, cloud.points, keep_invalid);
    }

    void RGBDImagesToPointsColors(const pangolin::Image<Eigen::Matrix<unsigned char,3,1>> &rgb_img,
                                  const pangolin::Image<unsigned short> &depth_img,
                                  const Eigen::Matrix3f &intr,
                                  VectorSet<float,3> &points,
                                  VectorSet<float,3> &colors,
                                  bool keep_invalid = false);

    inline void RGBDImagesToPointCloud(const pangolin::Image<Eigen::Matrix<unsigned char,3,1>> &rgb_img,
                                       const pangolin::Image<unsigned short> &depth_img,
                                       const Eigen::Matrix3f &intr,
                                       PointCloud<float,3> &cloud,
                                       bool keep_invalid = false)
    {
        RGBDImagesToPointsColors(rgb_img, depth_img, intr, cloud.points, cloud.colors, keep_invalid);
    }

    void pointsToDepthImage(const VectorSet<float,3> &points,
                            const Eigen::Matrix3f &intr,
                            pangolin::Image<unsigned short> &depth_img);

    inline void pointCloudToDepthImage(const PointCloud<float,3> &cloud,
                                       const Eigen::Matrix3f &intr,
                                       pangolin::Image<unsigned short> &depth_img)
    {
        pointsToDepthImage(cloud.points, intr, depth_img);
    }

    void pointsToDepthImage(const VectorSet<float,3> &points,
                            const Eigen::Matrix3f &intr,
                            const Eigen::Matrix3f &rot_mat,
                            const Eigen::Vector3f &t_vec,
                            pangolin::Image<unsigned short> &depth_img);

    inline void pointCloudToDepthImage(const PointCloud<float,3> &cloud,
                                       const Eigen::Matrix3f &intr,
                                       const Eigen::Matrix3f &rot_mat,
                                       const Eigen::Vector3f &t_vec,
                                       pangolin::Image<unsigned short> &depth_img)
    {
        pointsToDepthImage(cloud.points, intr, rot_mat, t_vec, depth_img);
    }

    void pointsColorsToRGBDImages(const VectorSet<float,3> &points,
                                  const VectorSet<float,3> &colors,
                                  const Eigen::Matrix3f &intr,
                                  pangolin::Image<Eigen::Matrix<unsigned char,3,1>> &rgb_img,
                                  pangolin::Image<unsigned short> &depth_img);

    inline void pointCloudToRGBDImages(const PointCloud<float,3> &cloud,
                                       const Eigen::Matrix3f &intr,
                                       pangolin::Image<Eigen::Matrix<unsigned char,3,1>> &rgb_img,
                                       pangolin::Image<unsigned short> &depth_img)
    {
        pointsColorsToRGBDImages(cloud.points, cloud.colors, intr, rgb_img, depth_img);
    }


    void pointsColorsToRGBDImages(const VectorSet<float,3> &points,
                                  const VectorSet<float,3> &colors,
                                  const Eigen::Matrix3f &intr,
                                  const Eigen::Matrix3f &rot_mat,
                                  const Eigen::Vector3f &t_vec,
                                  pangolin::Image<Eigen::Matrix<unsigned char,3,1>> &rgb_img,
                                  pangolin::Image<unsigned short> &depth_img);

    inline void pointCloudToRGBDImages(const PointCloud<float,3> &cloud,
                                       const Eigen::Matrix3f &intr,
                                       const Eigen::Matrix3f &rot_mat,
                                       const Eigen::Vector3f &t_vec,
                                       pangolin::Image<Eigen::Matrix<unsigned char,3,1>> &rgb_img,
                                       pangolin::Image<unsigned short> &depth_img)
    {
        pointsColorsToRGBDImages(cloud.points, cloud.colors, intr, rot_mat, t_vec, rgb_img, depth_img);
    }

    void pointsToIndexMap(const VectorSet<float,3> &points,
                          const Eigen::Matrix3f &intr,
                          pangolin::Image<size_t> &index_map);

    inline void pointCloudToIndexMap(const PointCloud<float,3> &cloud,
                                     const Eigen::Matrix3f &intr,
                                     pangolin::Image<size_t> &index_map)
    {
        pointsToIndexMap(cloud.points, intr, index_map);
    }

    void pointsToIndexMap(const VectorSet<float,3> &points,
                          const Eigen::Matrix3f &intr,
                          const Eigen::Matrix3f &rot_mat,
                          const Eigen::Vector3f &t_vec,
                          pangolin::Image<size_t> &index_map);

    inline void pointCloudToIndexMap(const PointCloud<float,3> &cloud,
                                     const Eigen::Matrix3f &intr,
                                     const Eigen::Matrix3f &rot_mat,
                                     const Eigen::Vector3f &t_vec,
                                     pangolin::Image<size_t> &index_map)
    {
        pointsToIndexMap(cloud.points, intr, rot_mat, t_vec, index_map);
    }
}
