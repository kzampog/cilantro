#pragma once

#include <cilantro/point_cloud.hpp>
#include <pangolin/pangolin.h>

void depthImageToPoints(const pangolin::Image<unsigned short> &depth_img,
                        const Eigen::Matrix3f &intr,
                        std::vector<Eigen::Vector3f> &points,
                        bool keep_invalid = false);

inline void depthImageToPointCloud(const pangolin::Image<unsigned short> &depth_img,
                                   const Eigen::Matrix3f &intr,
                                   PointCloud &cloud,
                                   bool keep_invalid = false)
{
    depthImageToPoints(depth_img, intr, cloud.points, keep_invalid);
}

void RGBDImagesToPointsColors(const pangolin::Image<Eigen::Matrix<unsigned char,3,1> > &rgb_img,
                              const pangolin::Image<unsigned short> &depth_img,
                              const Eigen::Matrix3f &intr,
                              std::vector<Eigen::Vector3f> &points,
                              std::vector<Eigen::Vector3f> &colors,
                              bool keep_invalid = false);

inline void RGBDImagesToPointCloud(const pangolin::Image<Eigen::Matrix<unsigned char,3,1> > &rgb_img,
                                   const pangolin::Image<unsigned short> &depth_img,
                                   const Eigen::Matrix3f &intr,
                                   PointCloud &cloud,
                                   bool keep_invalid = false)
{
    RGBDImagesToPointsColors(rgb_img, depth_img, intr, cloud.points, cloud.colors, keep_invalid);
}

void pointsToDepthImage(const std::vector<Eigen::Vector3f> &points,
                        const Eigen::Matrix3f &intr,
                        pangolin::ManagedImage<unsigned short> &depth_img);

inline void pointCloudToDepthImage(const PointCloud &cloud,
                                   const Eigen::Matrix3f &intr,
                                   pangolin::ManagedImage<unsigned short> &depth_img)
{
    pointsToDepthImage(cloud.points, intr, depth_img);
}

void pointsToDepthImage(const std::vector<Eigen::Vector3f> &points,
                        const Eigen::Matrix3f &intr,
                        const Eigen::Matrix3f &rot_mat,
                        const Eigen::Vector3f &t_vec,
                        pangolin::ManagedImage<unsigned short> &depth_img);

inline void pointCloudToDepthImage(const PointCloud &cloud,
                                   const Eigen::Matrix3f &intr,
                                   const Eigen::Matrix3f &rot_mat,
                                   const Eigen::Vector3f &t_vec,
                                   pangolin::ManagedImage<unsigned short> &depth_img)
{
    pointsToDepthImage(cloud.points, intr, rot_mat, t_vec, depth_img);
}

void pointsColorsToRGBDImages(const std::vector<Eigen::Vector3f> &points,
                              const std::vector<Eigen::Vector3f> &colors,
                              const Eigen::Matrix3f &intr,
                              pangolin::ManagedImage<Eigen::Matrix<unsigned char,3,1> > &rgb_img,
                              pangolin::ManagedImage<unsigned short> &depth_img);

inline void pointCloudToRGBDImages(const PointCloud &cloud,
                                   const Eigen::Matrix3f &intr,
                                   pangolin::ManagedImage<Eigen::Matrix<unsigned char,3,1> > &rgb_img,
                                   pangolin::ManagedImage<unsigned short> &depth_img)
{
    pointsColorsToRGBDImages(cloud.points, cloud.colors, intr, rgb_img, depth_img);
}


void pointsColorsToRGBDImages(const std::vector<Eigen::Vector3f> &points,
                              const std::vector<Eigen::Vector3f> &colors,
                              const Eigen::Matrix3f &intr,
                              const Eigen::Matrix3f &rot_mat,
                              const Eigen::Vector3f &t_vec,
                              pangolin::ManagedImage<Eigen::Matrix<unsigned char,3,1> > &rgb_img,
                              pangolin::ManagedImage<unsigned short> &depth_img);

inline void pointCloudToRGBDImages(const PointCloud &cloud,
                                   const Eigen::Matrix3f &intr,
                                   const Eigen::Matrix3f &rot_mat,
                                   const Eigen::Vector3f &t_vec,
                                   pangolin::ManagedImage<Eigen::Matrix<unsigned char,3,1> > &rgb_img,
                                   pangolin::ManagedImage<unsigned short> &depth_img)
{
    pointsColorsToRGBDImages(cloud.points, cloud.colors, intr, rot_mat, t_vec, rgb_img, depth_img);
}
