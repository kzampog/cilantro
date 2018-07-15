#pragma once

#include <cilantro/space_transformations.hpp>

namespace cilantro {
    template <typename RawDepthT, typename MetricDepthT>
    struct DepthValueConverter {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef RawDepthT RawDepth;
        typedef MetricDepthT MetricDepth;

        DepthValueConverter() : scale((MetricDepthT)(1.0)), inverseScale((MetricDepthT)(1.0)) {}
        DepthValueConverter(MetricDepthT mult) : scale(mult), inverseScale((MetricDepthT)(1.0)/mult) {}

        inline MetricDepthT getMetricValue(RawDepthT val) const { return inverseScale*static_cast<MetricDepthT>(val); }
        inline RawDepthT getRawValue(MetricDepthT val) const { return static_cast<RawDepthT>(scale*val); }

        const MetricDepthT scale;
        const MetricDepthT inverseScale;
    };

    template <class DepthConverterT>
    void depthImageToPoints(const typename DepthConverterT::RawDepth* depth_data,
                            const DepthConverterT &depth_converter,
                            size_t image_w, size_t image_h,
                            const Eigen::Ref<const Eigen::Matrix<typename DepthConverterT::MetricDepth,3,3>> &intrinsics,
                            VectorSet<typename DepthConverterT::MetricDepth,3> &points,
                            bool keep_invalid = false)
    {
        if (keep_invalid) {
            points.resize(Eigen::NoChange, image_w*image_h);
            size_t k;
#pragma omp parallel for private (k)
            for (size_t y = 0; y < image_h; y++) {
                k = y*image_w;
                for (size_t x = 0; x < image_w; x++) {
                    typename DepthConverterT::MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                    points(0,k) = (x - intrinsics(0,2))*z/intrinsics(0,0);
                    points(1,k) = (y - intrinsics(1,2))*z/intrinsics(1,1);
                    points(2,k) = z;
                    k++;
                }
            }
        } else {
            VectorSet<typename DepthConverterT::MetricDepth,3> points_tmp(3, image_w*image_h);
            size_t k;
            size_t valid_count = 0;
#pragma omp parallel for private (k) reduction (+: valid_count)
            for (size_t y = 0; y < image_h; y++) {
                k = y*image_w;
                for (size_t x = 0; x < image_w; x++) {
                    typename DepthConverterT::MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                    points_tmp(0,k) = (x - intrinsics(0,2))*z/intrinsics(0,0);
                    points_tmp(1,k) = (y - intrinsics(1,2))*z/intrinsics(1,1);
                    points_tmp(2,k) = z;
                    k++;
                    valid_count += z > (typename DepthConverterT::MetricDepth)0.0;
                }
            }
            k = 0;
            points.resize(Eigen::NoChange, valid_count);
            for (size_t i = 0; i < points_tmp.cols(); i++) {
                if (points_tmp(2,i) > (typename DepthConverterT::MetricDepth)0.0) points.col(k++) = points_tmp.col(i);
            }
        }
    }

    template <class DepthConverterT>
    void depthImageToPoints(const typename DepthConverterT::RawDepth* depth_data,
                            const DepthConverterT &depth_converter,
                            size_t image_w, size_t image_h,
                            const Eigen::Ref<const Eigen::Matrix<typename DepthConverterT::MetricDepth,3,3>> &intrinsics,
                            const RigidTransformation<typename DepthConverterT::MetricDepth,3> &extrinsics,
                            VectorSet<typename DepthConverterT::MetricDepth,3> &points,
                            bool keep_invalid = false)
    {
        if (keep_invalid) {
            points.resize(Eigen::NoChange, image_w*image_h);
            size_t k;
#pragma omp parallel for private (k)
            for (size_t y = 0; y < image_h; y++) {
                k = y*image_w;
                for (size_t x = 0; x < image_w; x++) {
                    typename DepthConverterT::MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                    points.col(k++).noalias() = extrinsics*Vector<typename DepthConverterT::MetricDepth,3>((x - intrinsics(0,2))*z/intrinsics(0,0), (y - intrinsics(1,2))*z/intrinsics(1,1), z);
                }
            }
        } else {
            VectorSet<typename DepthConverterT::MetricDepth,3> points_tmp(3, image_w*image_h);
            size_t k;
            size_t valid_count = 0;
#pragma omp parallel for private (k) reduction (+: valid_count)
            for (size_t y = 0; y < image_h; y++) {
                k = y*image_w;
                for (size_t x = 0; x < image_w; x++) {
                    typename DepthConverterT::MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                    points_tmp(0,k) = (x - intrinsics(0,2))*z/intrinsics(0,0);
                    points_tmp(1,k) = (y - intrinsics(1,2))*z/intrinsics(1,1);
                    points_tmp(2,k) = z;
                    k++;
                    valid_count += z > (typename DepthConverterT::MetricDepth)0.0;
                }
            }
            k = 0;
            points.resize(Eigen::NoChange, valid_count);
            for (size_t i = 0; i < points_tmp.cols(); i++) {
                if (points_tmp(2,i) > (typename DepthConverterT::MetricDepth)0.0) points.col(k++) = points_tmp.col(i);
            }
#pragma omp parallel for
            for (size_t i = 0; i < points.cols(); i++) {
                points.col(i) = extrinsics*points.col(i);
            }
        }
    }

    template <class DepthConverterT>
    void RGBDImagesToPointsColors(const unsigned char* rgb_data,
                                  const typename DepthConverterT::RawDepth* depth_data,
                                  const DepthConverterT &depth_converter,
                                  size_t image_w, size_t image_h,
                                  const Eigen::Ref<const Eigen::Matrix<typename DepthConverterT::MetricDepth,3,3>> &intrinsics,
                                  VectorSet<typename DepthConverterT::MetricDepth,3> &points,
                                  VectorSet<float,3> &colors,
                                  bool keep_invalid = false)
    {
        const float color_mult = 1.0f/255.0f;

        if (keep_invalid) {
            points.resize(Eigen::NoChange, image_w*image_h);
            colors.resize(Eigen::NoChange, image_w*image_h);
            size_t k;
#pragma omp parallel for private (k)
            for (size_t y = 0; y < image_h; y++) {
                k = y*image_w;
                for (size_t x = 0; x < image_w; x++) {
                    typename DepthConverterT::MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                    points(0,k) = (x - intrinsics(0,2))*z/intrinsics(0,0);
                    points(1,k) = (y - intrinsics(1,2))*z/intrinsics(1,1);
                    points(2,k) = z;
                    colors(0,k) = color_mult*static_cast<float>(rgb_data[3*k]);
                    colors(1,k) = color_mult*static_cast<float>(rgb_data[3*k + 1]);
                    colors(2,k) = color_mult*static_cast<float>(rgb_data[3*k + 2]);
                    k++;
                }
            }
        } else {
            VectorSet<typename DepthConverterT::MetricDepth,3> points_tmp(3, image_w*image_h);
            VectorSet<float,3> colors_tmp(3, image_w*image_h);
            size_t k;
            size_t valid_count = 0;
#pragma omp parallel for private (k) reduction (+: valid_count)
            for (size_t y = 0; y < image_h; y++) {
                k = y*image_w;
                for (size_t x = 0; x < image_w; x++) {
                    typename DepthConverterT::MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                    points_tmp(0,k) = (x - intrinsics(0,2))*z/intrinsics(0,0);
                    points_tmp(1,k) = (y - intrinsics(1,2))*z/intrinsics(1,1);
                    points_tmp(2,k) = z;
                    colors_tmp(0,k) = color_mult*static_cast<float>(rgb_data[3*k]);
                    colors_tmp(1,k) = color_mult*static_cast<float>(rgb_data[3*k + 1]);
                    colors_tmp(2,k) = color_mult*static_cast<float>(rgb_data[3*k + 2]);
                    k++;
                    valid_count += z > (typename DepthConverterT::MetricDepth)0.0;
                }
            }
            k = 0;
            points.resize(Eigen::NoChange, valid_count);
            colors.resize(Eigen::NoChange, valid_count);
            for (size_t i = 0; i < points_tmp.cols(); i++) {
                if (points_tmp(2,i) > (typename DepthConverterT::MetricDepth)0.0) {
                    points.col(k) = points_tmp.col(i);
                    colors.col(k) = colors_tmp.col(i);
                    k++;
                }
            }
        }
    }

    template <class DepthConverterT>
    void RGBDImagesToPointsColors(const unsigned char* rgb_data,
                                  const typename DepthConverterT::RawDepth* depth_data,
                                  const DepthConverterT &depth_converter,
                                  size_t image_w, size_t image_h,
                                  const Eigen::Ref<const Eigen::Matrix<typename DepthConverterT::MetricDepth,3,3>> &intrinsics,
                                  const RigidTransformation<typename DepthConverterT::MetricDepth,3> &extrinsics,
                                  VectorSet<typename DepthConverterT::MetricDepth,3> &points,
                                  VectorSet<float,3> &colors,
                                  bool keep_invalid = false)
    {
        const float color_mult = 1.0f/255.0f;

        if (keep_invalid) {
            points.resize(Eigen::NoChange, image_w*image_h);
            colors.resize(Eigen::NoChange, image_w*image_h);
            size_t k;
#pragma omp parallel for private (k)
            for (size_t y = 0; y < image_h; y++) {
                k = y*image_w;
                for (size_t x = 0; x < image_w; x++) {
                    typename DepthConverterT::MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                    points.col(k).noalias() = extrinsics*Vector<typename DepthConverterT::MetricDepth,3>((x - intrinsics(0,2))*z/intrinsics(0,0), (y - intrinsics(1,2))*z/intrinsics(1,1), z);
                    colors(0,k) = color_mult*static_cast<float>(rgb_data[3*k]);
                    colors(1,k) = color_mult*static_cast<float>(rgb_data[3*k + 1]);
                    colors(2,k) = color_mult*static_cast<float>(rgb_data[3*k + 2]);
                    k++;
                }
            }
        } else {
            VectorSet<typename DepthConverterT::MetricDepth,3> points_tmp(3, image_w*image_h);
            VectorSet<float,3> colors_tmp(3, image_w*image_h);
            size_t k;
            size_t valid_count = 0;
#pragma omp parallel for private (k) reduction (+: valid_count)
            for (size_t y = 0; y < image_h; y++) {
                k = y*image_w;
                for (size_t x = 0; x < image_w; x++) {
                    typename DepthConverterT::MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                    points_tmp(0,k) = (x - intrinsics(0,2))*z/intrinsics(0,0);
                    points_tmp(1,k) = (y - intrinsics(1,2))*z/intrinsics(1,1);
                    points_tmp(2,k) = z;
                    colors_tmp(0,k) = color_mult*static_cast<float>(rgb_data[3*k]);
                    colors_tmp(1,k) = color_mult*static_cast<float>(rgb_data[3*k + 1]);
                    colors_tmp(2,k) = color_mult*static_cast<float>(rgb_data[3*k + 2]);
                    k++;
                    valid_count += z > (typename DepthConverterT::MetricDepth)0.0;
                }
            }
            k = 0;
            points.resize(Eigen::NoChange, valid_count);
            colors.resize(Eigen::NoChange, valid_count);
            for (size_t i = 0; i < points_tmp.cols(); i++) {
                if (points_tmp(2,i) > (typename DepthConverterT::MetricDepth)0.0) {
                    points.col(k) = points_tmp.col(i);
                    colors.col(k) = colors_tmp.col(i);
                    k++;
                }
            }
#pragma omp parallel for
            for (size_t i = 0; i < points.cols(); i++) {
                points.col(i) = extrinsics*points.col(i);
            }
        }
    }

    template <class DepthConverterT>
    void pointsToDepthImage(const ConstVectorSetMatrixMap<typename DepthConverterT::MetricDepth,3> &points,
                            const Eigen::Ref<const Eigen::Matrix<typename DepthConverterT::MetricDepth,3,3>> &intrinsics,
                            const DepthConverterT &depth_converter,
                            typename DepthConverterT::RawDepth* depth_data,
                            size_t image_w, size_t image_h)
    {
#pragma omp parallel for
        for (size_t i = 0; i < image_w*image_h; i++) {
            depth_data[i] = (typename DepthConverterT::RawDepth)0;
        }
#pragma omp parallel for
        for (size_t i = 0; i < points.cols(); i++) {
            if (points(2,i) <= (typename DepthConverterT::MetricDepth)0.0) continue;
            size_t x = (size_t)std::llround(points(0,i)*intrinsics(0,0)/points(2,i) + intrinsics(0,2));
            size_t y = (size_t)std::llround(points(1,i)*intrinsics(1,1)/points(2,i) + intrinsics(1,2));
            if (x >= image_w || y >= image_h) continue;
            size_t ind = y*image_w + x;
            typename DepthConverterT::RawDepth depth_val = depth_converter.getRawValue(points(2,i));
            if (depth_data[ind] == (typename DepthConverterT::RawDepth)0 || depth_val < depth_data[ind]) {
                depth_data[ind] = depth_val;
            }
        }
    }

    template <class DepthConverterT>
    void pointsToDepthImage(const ConstVectorSetMatrixMap<typename DepthConverterT::MetricDepth,3> &points,
                            const RigidTransformation<typename DepthConverterT::MetricDepth,3> &extrinsics,
                            const Eigen::Ref<const Eigen::Matrix<typename DepthConverterT::MetricDepth,3,3>> &intrinsics,
                            const DepthConverterT &depth_converter,
                            typename DepthConverterT::RawDepth* depth_data,
                            size_t image_w, size_t image_h)
    {
        const RigidTransformation<typename DepthConverterT::MetricDepth,3> to_cam(extrinsics.inverse());

#pragma omp parallel for
        for (size_t i = 0; i < image_w*image_h; i++) {
            depth_data[i] = (typename DepthConverterT::RawDepth)0;
        }

        Vector<typename DepthConverterT::MetricDepth,3> pt_cam;
#pragma omp parallel for private (pt_cam)
        for (size_t i = 0; i < points.cols(); i++) {
            pt_cam.noalias() = to_cam*points.col(i);
            if (pt_cam(2) <= (typename DepthConverterT::MetricDepth)0.0) continue;
            size_t x = (size_t)std::llround(pt_cam(0)*intrinsics(0,0)/pt_cam(2) + intrinsics(0,2));
            size_t y = (size_t)std::llround(pt_cam(1)*intrinsics(1,1)/pt_cam(2) + intrinsics(1,2));
            if (x >= image_w || y >= image_h) continue;
            size_t ind = y*image_w + x;
            typename DepthConverterT::RawDepth depth_val = depth_converter.getRawValue(pt_cam(2));
            if (depth_data[ind] == (typename DepthConverterT::RawDepth)0 || depth_val < depth_data[ind]) {
                depth_data[ind] = depth_val;
            }
        }
    }

    template <class DepthConverterT>
    void pointsColorsToRGBDImages(const ConstVectorSetMatrixMap<typename DepthConverterT::MetricDepth,3> &points,
                                  const ConstVectorSetMatrixMap<float,3> &colors,
                                  const Eigen::Ref<const Eigen::Matrix<typename DepthConverterT::MetricDepth,3,3>> &intrinsics,
                                  const DepthConverterT &depth_converter,
                                  unsigned char* rgb_data,
                                  typename DepthConverterT::RawDepth* depth_data,
                                  size_t image_w, size_t image_h)
    {
#pragma omp parallel for
        for (size_t i = 0; i < image_w*image_h; i++) {
            rgb_data[3*i] = (unsigned char)0;
            rgb_data[3*i + 1] = (unsigned char)0;
            rgb_data[3*i + 2] = (unsigned char)0;
            depth_data[i] = (typename DepthConverterT::RawDepth)0;
        }
#pragma omp parallel for
        for (size_t i = 0; i < points.cols(); i++) {
            if (points(2,i) <= (typename DepthConverterT::MetricDepth)0.0) continue;
            size_t x = (size_t)std::llround(points(0,i)*intrinsics(0,0)/points(2,i) + intrinsics(0,2));
            size_t y = (size_t)std::llround(points(1,i)*intrinsics(1,1)/points(2,i) + intrinsics(1,2));
            if (x >= image_w || y >= image_h) continue;
            size_t ind = y*image_w + x;
            typename DepthConverterT::RawDepth depth_val = depth_converter.getRawValue(points(2,i));
            if (depth_data[ind] == (typename DepthConverterT::RawDepth)0 || depth_val < depth_data[ind]) {
                rgb_data[3*ind] = static_cast<unsigned char>(255.0f*colors(0,i));
                rgb_data[3*ind + 1] = static_cast<unsigned char>(255.0f*colors(1,i));
                rgb_data[3*ind + 2] = static_cast<unsigned char>(255.0f*colors(2,i));
                depth_data[ind] = depth_val;
            }
        }
    }

    template <class DepthConverterT>
    void pointsColorsToRGBDImages(const ConstVectorSetMatrixMap<typename DepthConverterT::MetricDepth,3> &points,
                                  const ConstVectorSetMatrixMap<float,3> &colors,
                                  const RigidTransformation<typename DepthConverterT::MetricDepth,3> &extrinsics,
                                  const Eigen::Ref<const Eigen::Matrix<typename DepthConverterT::MetricDepth,3,3>> &intrinsics,
                                  const DepthConverterT &depth_converter,
                                  unsigned char* rgb_data,
                                  typename DepthConverterT::RawDepth* depth_data,
                                  size_t image_w, size_t image_h)
    {
        const RigidTransformation<typename DepthConverterT::MetricDepth,3> to_cam(extrinsics.inverse());

#pragma omp parallel for
        for (size_t i = 0; i < image_w*image_h; i++) {
            rgb_data[3*i] = (unsigned char)0;
            rgb_data[3*i + 1] = (unsigned char)0;
            rgb_data[3*i + 2] = (unsigned char)0;
            depth_data[i] = (typename DepthConverterT::RawDepth)0;
        }

        Vector<typename DepthConverterT::MetricDepth,3> pt_cam;
#pragma omp parallel for private (pt_cam)
        for (size_t i = 0; i < points.cols(); i++) {
            pt_cam.noalias() = to_cam*points.col(i);
            if (pt_cam(2) <= (typename DepthConverterT::MetricDepth)0.0) continue;
            size_t x = (size_t)std::llround(pt_cam(0)*intrinsics(0,0)/pt_cam(2) + intrinsics(0,2));
            size_t y = (size_t)std::llround(pt_cam(1)*intrinsics(1,1)/pt_cam(2) + intrinsics(1,2));
            if (x >= image_w || y >= image_h) continue;
            size_t ind = y*image_w + x;
            typename DepthConverterT::RawDepth depth_val = depth_converter.getRawValue(pt_cam(2));
            if (depth_data[ind] == (typename DepthConverterT::RawDepth)0 || depth_val < depth_data[ind]) {
                rgb_data[3*ind] = static_cast<unsigned char>(255.0f*colors(0,i));
                rgb_data[3*ind + 1] = static_cast<unsigned char>(255.0f*colors(1,i));
                rgb_data[3*ind + 2] = static_cast<unsigned char>(255.0f*colors(2,i));
                depth_data[ind] = depth_val;
            }
        }
    }

    template <typename PointT>
    void pointsToIndexMap(const ConstVectorSetMatrixMap<PointT,3> &points,
                          const Eigen::Ref<const Eigen::Matrix<PointT,3,3>> &intrinsics,
                          size_t* index_map_data,
                          size_t image_w, size_t image_h)
    {
        const size_t empty = std::numeric_limits<size_t>::max();

#pragma omp parallel for
        for (size_t i = 0; i < image_w*image_h; i++) {
            index_map_data[i] = empty;
        }
#pragma omp parallel for
        for (size_t i = 0; i < points.cols(); i++) {
            if (points(2,i) <= (PointT)0.0) continue;
            size_t x = (size_t)std::llround(points(0,i)*intrinsics(0,0)/points(2,i) + intrinsics(0,2));
            size_t y = (size_t)std::llround(points(1,i)*intrinsics(1,1)/points(2,i) + intrinsics(1,2));
            if (x >= image_w || y >= image_h) continue;
            size_t ind = y*image_w + x;
            if (index_map_data[ind] == empty || points(2,i) < points(2,index_map_data[ind])) {
                index_map_data[ind] = i;
            }
        }
    }

    template <typename PointT>
    void pointsToIndexMap(const ConstVectorSetMatrixMap<PointT,3> &points,
                          const RigidTransformation<PointT,3> &extrinsics,
                          const Eigen::Ref<const Eigen::Matrix<PointT,3,3>> &intrinsics,
                          size_t* index_map_data,
                          size_t image_w, size_t image_h)
    {
        const RigidTransformation<PointT,3> to_cam(extrinsics.inverse());
        const size_t empty = std::numeric_limits<size_t>::max();

#pragma omp parallel for
        for (size_t i = 0; i < image_w*image_h; i++) {
            index_map_data[i] = empty;
        }

        VectorSet<PointT,3> points_cam(3, points.cols());
#pragma omp parallel for
        for (size_t i = 0; i < points.cols(); i++) {
            points_cam.col(i).noalias() = to_cam*points.col(i);
        }

#pragma omp parallel for
        for (size_t i = 0; i < points.cols(); i++) {
            if (points_cam(2,i) <= (PointT)0.0) continue;
            size_t x = (size_t)std::llround(points_cam(0,i)*intrinsics(0,0)/points_cam(2,i) + intrinsics(0,2));
            size_t y = (size_t)std::llround(points_cam(1,i)*intrinsics(1,1)/points_cam(2,i) + intrinsics(1,2));
            if (x >= image_w || y >= image_h) continue;
            size_t ind = y*image_w + x;
            if (index_map_data[ind] == empty || points_cam(2,i) < points_cam(2,index_map_data[ind])) {
                index_map_data[ind] = i;
            }
        }
    }
}
