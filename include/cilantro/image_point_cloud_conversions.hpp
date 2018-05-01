#pragma once

#include <cilantro/data_containers.hpp>

namespace cilantro {
    template <typename RawDepthT, typename MetricDepthT>
    struct DepthValueConverter {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        DepthValueConverter() : scale((MetricDepthT)(1.0)), inverseScale((MetricDepthT)(1.0)) {}
        DepthValueConverter(MetricDepthT mult) : scale(mult), inverseScale((MetricDepthT)(1.0)/mult) {}

        inline MetricDepthT getMetricValue(RawDepthT val) const { return inverseScale*static_cast<MetricDepthT>(val); }
        inline RawDepthT getRawValue(MetricDepthT val) const { return static_cast<RawDepthT>(scale*val); }

        const MetricDepthT scale;
        const MetricDepthT inverseScale;
    };

    template <typename DepthT, typename PointT, class DepthConverterT = DepthValueConverter<DepthT,PointT>>
    void depthImageToPoints(const DepthT* depth_data,
                            size_t image_w, size_t image_h,
                            const Eigen::Ref<const Eigen::Matrix<PointT,3,3>> &intrinsics,
                            VectorSet<PointT,3> &points,
                            bool keep_invalid = false,
                            const DepthConverterT &depth_converter = DepthConverterT())
    {
        if (keep_invalid) {
            points.resize(Eigen::NoChange, image_w*image_h);
            size_t k;
#pragma omp parallel for private (k)
            for (size_t y = 0; y < image_h; y++) {
                k = y*image_w;
                for (size_t x = 0; x < image_w; x++) {
                    PointT z = depth_converter.getMetricValue(depth_data[k]);
                    points(0,k) = (x - intrinsics(0,2))*z/intrinsics(0,0);
                    points(1,k) = (y - intrinsics(1,2))*z/intrinsics(1,1);
                    points(2,k) = z;
                    k++;
                }
            }
        } else {
            VectorSet<PointT,3> points_tmp(3, image_w*image_h);
            size_t k;
            size_t valid_count = 0;
#pragma omp parallel for private (k) reduction (+: valid_count)
            for (size_t y = 0; y < image_h; y++) {
                k = y*image_w;
                for (size_t x = 0; x < image_w; x++) {
                    PointT z = depth_converter.getMetricValue(depth_data[k]);
                    points_tmp(0,k) = (x - intrinsics(0,2))*z/intrinsics(0,0);
                    points_tmp(1,k) = (y - intrinsics(1,2))*z/intrinsics(1,1);
                    points_tmp(2,k) = z;
                    k++;
                    valid_count += z > (PointT)0.0;
                }
            }
            k = 0;
            points.resize(Eigen::NoChange, valid_count);
            for (size_t i = 0; i < points_tmp.cols(); i++) {
                if (points_tmp(2,i) > (PointT)0.0) points.col(k++) = points_tmp.col(i);
            }
        }
    }

    template <typename DepthT, typename PointT, class DepthConverterT = DepthValueConverter<DepthT,PointT>>
    void RGBDImagesToPointsColors(const unsigned char* rgb_data,
                                  const DepthT* depth_data,
                                  size_t image_w, size_t image_h,
                                  const Eigen::Ref<const Eigen::Matrix<PointT,3,3>> &intrinsics,
                                  VectorSet<PointT,3> &points,
                                  VectorSet<float,3> &colors,
                                  bool keep_invalid = false,
                                  const DepthConverterT &depth_converter = DepthConverterT())
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
                    PointT z = depth_converter.getMetricValue(depth_data[k]);
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
            VectorSet<PointT,3> points_tmp(3, image_w*image_h);
            VectorSet<float,3> colors_tmp(3, image_w*image_h);
            size_t k;
            size_t valid_count = 0;
#pragma omp parallel for private (k) reduction (+: valid_count)
            for (size_t y = 0; y < image_h; y++) {
                k = y*image_w;
                for (size_t x = 0; x < image_w; x++) {
                    PointT z = depth_converter.getMetricValue(depth_data[k]);
                    points_tmp(0,k) = (x - intrinsics(0,2))*z/intrinsics(0,0);
                    points_tmp(1,k) = (y - intrinsics(1,2))*z/intrinsics(1,1);
                    points_tmp(2,k) = z;
                    colors_tmp(0,k) = color_mult*static_cast<float>(rgb_data[3*k]);
                    colors_tmp(1,k) = color_mult*static_cast<float>(rgb_data[3*k + 1]);
                    colors_tmp(2,k) = color_mult*static_cast<float>(rgb_data[3*k + 2]);
                    k++;
                    valid_count += z > (PointT)0.0;
                }
            }
            k = 0;
            points.resize(Eigen::NoChange, valid_count);
            colors.resize(Eigen::NoChange, valid_count);
            for (size_t i = 0; i < points_tmp.cols(); i++) {
                if (points_tmp(2,i) > (PointT)0.0) {
                    points.col(k) = points_tmp.col(i);
                    colors.col(k) = colors_tmp.col(i);
                    k++;
                }
            }
        }
    }

    template <typename PointT, typename DepthT, class DepthConverterT = DepthValueConverter<DepthT,PointT>>
    void pointsToDepthImage(const ConstVectorSetMatrixMap<PointT,3> &points,
                            const Eigen::Ref<const Eigen::Matrix<PointT,3,3>> &intrinsics,
                            DepthT* depth_data,
                            size_t image_w, size_t image_h,
                            const DepthConverterT &depth_converter = DepthConverterT())
    {
#pragma omp parallel for
        for (size_t i = 0; i < image_w*image_h; i++) {
            depth_data[i] = (DepthT)0;
        }
#pragma omp parallel for
        for (size_t i = 0; i < points.cols(); i++) {
            if (points(2,i) <= (PointT)0.0) continue;
            size_t x = (size_t)std::llround(points(0,i)*intrinsics(0,0)/points(2,i) + intrinsics(0,2));
            size_t y = (size_t)std::llround(points(1,i)*intrinsics(1,1)/points(2,i) + intrinsics(1,2));
            if (x >= image_w || y >= image_h) continue;
            size_t ind = y*image_w + x;
            DepthT depth_val = depth_converter.getRawValue(points(2,i));
            if (depth_data[ind] == (DepthT)0 || depth_val < depth_data[ind]) {
                depth_data[ind] = depth_val;
            }
        }
    }

    template <typename PointT, typename DepthT, class DepthConverterT = DepthValueConverter<DepthT,PointT>>
    void pointsColorsToRGBDImages(const ConstVectorSetMatrixMap<PointT,3> &points,
                                  const ConstVectorSetMatrixMap<float,3> &colors,
                                  const Eigen::Ref<const Eigen::Matrix<PointT,3,3>> &intrinsics,
                                  unsigned char* rgb_data,
                                  DepthT* depth_data,
                                  size_t image_w, size_t image_h,
                                  const DepthConverterT &depth_converter = DepthConverterT())
    {
#pragma omp parallel for
        for (size_t i = 0; i < image_w*image_h; i++) {
            rgb_data[3*i] = (unsigned char)0;
            rgb_data[3*i + 1] = (unsigned char)0;
            rgb_data[3*i + 2] = (unsigned char)0;
            depth_data[i] = (DepthT)0;
        }
#pragma omp parallel for
        for (size_t i = 0; i < points.cols(); i++) {
            if (points(2,i) <= (PointT)0.0) continue;
            size_t x = (size_t)std::llround(points(0,i)*intrinsics(0,0)/points(2,i) + intrinsics(0,2));
            size_t y = (size_t)std::llround(points(1,i)*intrinsics(1,1)/points(2,i) + intrinsics(1,2));
            if (x >= image_w || y >= image_h) continue;
            size_t ind = y*image_w + x;
            DepthT depth_val = depth_converter.getRawValue(points(2,i));
            if (depth_data[ind] == (DepthT)0 || depth_val < depth_data[ind]) {
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
        const size_t empty = std::numeric_limits<std::size_t>::max();

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
}
