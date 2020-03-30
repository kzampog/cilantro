#pragma once

#include <cilantro/core/space_transformations.hpp>

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

    template <typename RawDepthT, typename MetricDepthT>
    struct TruncatedDepthValueConverter {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef RawDepthT RawDepth;
        typedef MetricDepthT MetricDepth;

        TruncatedDepthValueConverter()
                : scale((MetricDepthT)(1.0)),
                  inverseScale((MetricDepthT)(1.0)),
                  maxDepth(std::numeric_limits<MetricDepthT>::max())
        {}

        TruncatedDepthValueConverter(MetricDepthT mult, MetricDepthT thresh)
                : scale(mult),
                  inverseScale((MetricDepthT)(1.0)/mult),
                  maxDepth(thresh)
        {}

        inline MetricDepthT getMetricValue(RawDepthT val) const {
            MetricDepthT res = inverseScale*static_cast<MetricDepthT>(val);
            return (res < maxDepth) ? res : (MetricDepthT)0.0;
        }

        inline RawDepthT getRawValue(MetricDepthT val) const {
            return (val < maxDepth) ? static_cast<RawDepthT>(scale*val) : (RawDepthT)0;
        }

        const MetricDepthT scale;
        const MetricDepthT inverseScale;
        const MetricDepthT maxDepth;
    };

    template <class DepthConverterT>
    void depthImageToPoints(const typename DepthConverterT::RawDepth* depth_data,
                            const DepthConverterT &depth_converter,
                            size_t image_w, size_t image_h,
                            const Eigen::Ref<const Eigen::Matrix<typename DepthConverterT::MetricDepth,3,3>> &intrinsics,
                            VectorSet<typename DepthConverterT::MetricDepth,3> &points,
                            bool keep_invalid = false)
    {
        using MetricDepth = typename DepthConverterT::MetricDepth;

        const Eigen::Matrix<MetricDepth,3,3,Eigen::RowMajor> intrinsics_inv = intrinsics.inverse();

        if (keep_invalid) {
            points.resize(3, image_w*image_h);
            size_t k;
#pragma omp parallel for private (k)
            for (size_t y = 0; y < image_h; y++) {
                k = y*image_w;
                for (size_t x = 0; x < image_w; x++) {
                    const MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                    points.col(k).noalias() = intrinsics_inv*Vector<MetricDepth,3>(z*x, z*y, z);
                    k++;
                }
            }
        } else {
            VectorSet<MetricDepth,3> points_tmp(3, image_w*image_h);
            size_t k;
            size_t valid_count = 0;
#pragma omp parallel for private (k) reduction (+: valid_count)
            for (size_t y = 0; y < image_h; y++) {
                k = y*image_w;
                for (size_t x = 0; x < image_w; x++) {
                    const MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                    points_tmp.col(k).noalias() = intrinsics_inv*Vector<MetricDepth,3>(z*x, z*y, z);
                    k++;
                    valid_count += z > (MetricDepth)0.0;
                }
            }

            k = 0;
            points.resize(3, valid_count);
            for (size_t i = 0; i < points_tmp.cols(); i++) {
                if (points_tmp(2,i) > (MetricDepth)0.0) points.col(k++) = points_tmp.col(i);
            }
        }
    }

    template <class DepthConverterT>
    void depthImageToPoints(const typename DepthConverterT::RawDepth* depth_data,
                            const DepthConverterT &depth_converter,
                            size_t image_w, size_t image_h,
                            const Eigen::Ref<const Eigen::Matrix<typename DepthConverterT::MetricDepth,3,3>> &intrinsics,
                            const RigidTransform<typename DepthConverterT::MetricDepth,3> &extrinsics,
                            VectorSet<typename DepthConverterT::MetricDepth,3> &points,
                            bool keep_invalid = false)
    {
        using MetricDepth = typename DepthConverterT::MetricDepth;

        const Eigen::Matrix<MetricDepth,3,3,Eigen::RowMajor> intrinsics_inv = intrinsics.inverse();

        if (keep_invalid) {
            points.resize(3, image_w*image_h);
            size_t k;
#pragma omp parallel for private (k)
            for (size_t y = 0; y < image_h; y++) {
                k = y*image_w;
                for (size_t x = 0; x < image_w; x++) {
                    const MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                    points.col(k++).noalias() = extrinsics*intrinsics_inv*Vector<MetricDepth,3>(z*x, z*y, z);
                }
            }
        } else {
            VectorSet<MetricDepth,3> points_tmp(3, image_w*image_h);
            size_t k;
            size_t valid_count = 0;
#pragma omp parallel for private (k) reduction (+: valid_count)
            for (size_t y = 0; y < image_h; y++) {
                k = y*image_w;
                for (size_t x = 0; x < image_w; x++) {
                    const MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                    points_tmp.col(k).noalias() = intrinsics_inv*Vector<MetricDepth,3>(z*x, z*y, z);
                    k++;
                    valid_count += z > (MetricDepth)0.0;
                }
            }

            k = 0;
            points.resize(3, valid_count);
            for (size_t i = 0; i < points_tmp.cols(); i++) {
                if (points_tmp(2,i) > (MetricDepth)0.0) points.col(k++) = points_tmp.col(i);
            }

#pragma omp parallel for
            for (size_t i = 0; i < points.cols(); i++) {
                points.col(i) = extrinsics*points.col(i);
            }
        }
    }

    template <class DepthConverterT>
    void depthImageToPointsNormals(const typename DepthConverterT::RawDepth* depth_data,
                                   const DepthConverterT &depth_converter,
                                   size_t image_w, size_t image_h,
                                   const Eigen::Ref<const Eigen::Matrix<typename DepthConverterT::MetricDepth,3,3>> &intrinsics,
                                   VectorSet<typename DepthConverterT::MetricDepth,3> &points,
                                   VectorSet<typename DepthConverterT::MetricDepth,3> &normals,
                                   bool keep_invalid = false)
    {
        using MetricDepth = typename DepthConverterT::MetricDepth;

        const Eigen::Matrix<MetricDepth,3,3,Eigen::RowMajor> intrinsics_inv = intrinsics.inverse();

        if (keep_invalid) {
            points.resize(3, image_w*image_h);
            normals.setConstant(3, points.cols(), std::numeric_limits<MetricDepth>::quiet_NaN());

#pragma omp parallel
            {
                size_t k;
#pragma omp for
                for (size_t y = 0; y < image_h; y++) {
                    k = y*image_w;
                    for (size_t x = 0; x < image_w; x++) {
                        const MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                        points.col(k).noalias() = intrinsics_inv*Vector<MetricDepth,3>(z*x, z*y, z);
                        k++;
                    }
                }

#pragma omp for
                for (size_t y = 1; y < image_h - 1; y++) {
                    for (size_t x = 1; x < image_w - 1; x++) {
                        k = y*image_w + x;
                        if (points(2,k) > (MetricDepth)0.0 &&
                            points(2,k+1) > (MetricDepth)0.0 &&
                            points(2,k-1) > (MetricDepth)0.0 &&
                            points(2,k+image_w) > (MetricDepth)0.0 &&
                            points(2,k-image_w) > (MetricDepth)0.0)
                        {
                            normals.col(k).noalias() = (points.col(k+image_w) - points.col(k-image_w)).cross(points.col(k+1) - points.col(k-1)).normalized();
                        }
                    }
                }
            }
        } else {
            VectorSet<MetricDepth,3> points_tmp(3, image_w*image_h);
            VectorSet<MetricDepth,3> normals_tmp(VectorSet<MetricDepth,3>::Constant(3, points_tmp.cols(), std::numeric_limits<MetricDepth>::quiet_NaN()));
            size_t valid_count = 0;

#pragma omp parallel
            {
                size_t k;
#pragma omp for
                for (size_t y = 0; y < image_h; y++) {
                    k = y*image_w;
                    for (size_t x = 0; x < image_w; x++) {
                        const MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                        points_tmp.col(k).noalias() = intrinsics_inv*Vector<MetricDepth,3>(z*x, z*y, z);
                        k++;
                    }
                }

#pragma omp for reduction (+: valid_count)
                for (size_t y = 1; y < image_h - 1; y++) {
                    for (size_t x = 1; x < image_w - 1; x++) {
                        k = y*image_w + x;
                        if (points_tmp(2,k) > (MetricDepth)0.0 &&
                            points_tmp(2,k+1) > (MetricDepth)0.0 &&
                            points_tmp(2,k-1) > (MetricDepth)0.0 &&
                            points_tmp(2,k+image_w) > (MetricDepth)0.0 &&
                            points_tmp(2,k-image_w) > (MetricDepth)0.0)
                        {
                            normals_tmp.col(k).noalias() = (points_tmp.col(k+image_w) - points_tmp.col(k-image_w)).cross(points_tmp.col(k+1) - points_tmp.col(k-1)).normalized();
                            valid_count += 1;
                        }
                    }
                }
            }

            size_t k = 0;
            points.resize(3, valid_count);
            normals.resize(3, valid_count);
            for (size_t i = 0; i < points_tmp.cols(); i++) {
                if (!std::isnan(normals_tmp(0,i))) {
                    points.col(k) = points_tmp.col(i);
                    normals.col(k) = normals_tmp.col(i);
                    k++;
                }
            }
        }
    }

    template <class DepthConverterT>
    void depthImageToPointsNormals(const typename DepthConverterT::RawDepth* depth_data,
                                   const DepthConverterT &depth_converter,
                                   size_t image_w, size_t image_h,
                                   const Eigen::Ref<const Eigen::Matrix<typename DepthConverterT::MetricDepth,3,3>> &intrinsics,
                                   const RigidTransform<typename DepthConverterT::MetricDepth,3> &extrinsics,
                                   VectorSet<typename DepthConverterT::MetricDepth,3> &points,
                                   VectorSet<typename DepthConverterT::MetricDepth,3> &normals,
                                   bool keep_invalid = false)
    {
        using MetricDepth = typename DepthConverterT::MetricDepth;

        const Eigen::Matrix<MetricDepth,3,3,Eigen::RowMajor> intrinsics_inv = intrinsics.inverse();

        if (keep_invalid) {
            points.resize(3, image_w*image_h);
            normals.setConstant(3, points.cols(), std::numeric_limits<MetricDepth>::quiet_NaN());

#pragma omp parallel
            {
                size_t k;
#pragma omp for
                for (size_t y = 0; y < image_h; y++) {
                    k = y*image_w;
                    for (size_t x = 0; x < image_w; x++) {
                        const MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                        points.col(k).noalias() = intrinsics_inv*Vector<MetricDepth,3>(z*x, z*y, z);
                        k++;
                    }
                }

#pragma omp for
                for (size_t y = 1; y < image_h - 1; y++) {
                    for (size_t x = 1; x < image_w - 1; x++) {
                        k = y*image_w + x;
                        if (points(2,k) > (MetricDepth)0.0 &&
                            points(2,k+1) > (MetricDepth)0.0 &&
                            points(2,k-1) > (MetricDepth)0.0 &&
                            points(2,k+image_w) > (MetricDepth)0.0 &&
                            points(2,k-image_w) > (MetricDepth)0.0)
                        {
                            normals.col(k).noalias() = (points.col(k+image_w) - points.col(k-image_w)).cross(points.col(k+1) - points.col(k-1)).normalized();
                        }
                    }
                }

#pragma omp for
                for (size_t i = 0; i < points.cols(); i++) {
                    points.col(i) = extrinsics*points.col(i);
                    normals.col(i) = extrinsics.linear()*normals.col(i);
                }
            }
        } else {
            VectorSet<MetricDepth,3> points_tmp(3, image_w*image_h);
            VectorSet<MetricDepth,3> normals_tmp(VectorSet<MetricDepth,3>::Constant(3, points_tmp.cols(), std::numeric_limits<MetricDepth>::quiet_NaN()));
            size_t valid_count = 0;

#pragma omp parallel
            {
                size_t k;
#pragma omp for
                for (size_t y = 0; y < image_h; y++) {
                    k = y*image_w;
                    for (size_t x = 0; x < image_w; x++) {
                        const MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                        points_tmp.col(k).noalias() = intrinsics_inv*Vector<MetricDepth,3>(z*x, z*y, z);
                        k++;
                    }
                }

#pragma omp for reduction (+: valid_count)
                for (size_t y = 1; y < image_h - 1; y++) {
                    for (size_t x = 1; x < image_w - 1; x++) {
                        k = y*image_w + x;
                        if (points_tmp(2,k) > (MetricDepth)0.0 &&
                            points_tmp(2,k+1) > (MetricDepth)0.0 &&
                            points_tmp(2,k-1) > (MetricDepth)0.0 &&
                            points_tmp(2,k+image_w) > (MetricDepth)0.0 &&
                            points_tmp(2,k-image_w) > (MetricDepth)0.0)
                        {
                            normals_tmp.col(k).noalias() = (points_tmp.col(k+image_w) - points_tmp.col(k-image_w)).cross(points_tmp.col(k+1) - points_tmp.col(k-1)).normalized();
                            valid_count += 1;
                        }
                    }
                }
            }

            size_t k = 0;
            points.resize(3, valid_count);
            normals.resize(3, valid_count);
            for (size_t i = 0; i < points_tmp.cols(); i++) {
                if (!std::isnan(normals_tmp(0,i))) {
                    points.col(k) = points_tmp.col(i);
                    normals.col(k) = normals_tmp.col(i);
                    k++;
                }
            }

#pragma omp parallel for
            for (size_t i = 0; i < points.cols(); i++) {
                points.col(i) = extrinsics*points.col(i);
                normals.col(i) = extrinsics.linear()*normals.col(i);
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
        using MetricDepth = typename DepthConverterT::MetricDepth;

        const Eigen::Matrix<MetricDepth,3,3,Eigen::RowMajor> intrinsics_inv = intrinsics.inverse();

        const float color_mult = 1.0f/255.0f;

        if (keep_invalid) {
            points.resize(3, image_w*image_h);
            colors.resize(3, image_w*image_h);
            size_t k;
#pragma omp parallel for private (k)
            for (size_t y = 0; y < image_h; y++) {
                k = y*image_w;
                for (size_t x = 0; x < image_w; x++) {
                    const MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                    points.col(k).noalias() = intrinsics_inv*Vector<MetricDepth,3>(z*x, z*y, z);
                    colors(0,k) = color_mult*static_cast<float>(rgb_data[3*k]);
                    colors(1,k) = color_mult*static_cast<float>(rgb_data[3*k + 1]);
                    colors(2,k) = color_mult*static_cast<float>(rgb_data[3*k + 2]);
                    k++;
                }
            }
        } else {
            VectorSet<MetricDepth,3> points_tmp(3, image_w*image_h);
            VectorSet<float,3> colors_tmp(3, image_w*image_h);
            size_t k;
            size_t valid_count = 0;
#pragma omp parallel for private (k) reduction (+: valid_count)
            for (size_t y = 0; y < image_h; y++) {
                k = y*image_w;
                for (size_t x = 0; x < image_w; x++) {
                    const MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                    points_tmp.col(k).noalias() = intrinsics_inv*Vector<MetricDepth,3>(z*x, z*y, z);
                    colors_tmp(0,k) = color_mult*static_cast<float>(rgb_data[3*k]);
                    colors_tmp(1,k) = color_mult*static_cast<float>(rgb_data[3*k + 1]);
                    colors_tmp(2,k) = color_mult*static_cast<float>(rgb_data[3*k + 2]);
                    k++;
                    valid_count += z > (MetricDepth)0.0;
                }
            }

            k = 0;
            points.resize(3, valid_count);
            colors.resize(3, valid_count);
            for (size_t i = 0; i < points_tmp.cols(); i++) {
                if (points_tmp(2,i) > (MetricDepth)0.0) {
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
                                  const RigidTransform<typename DepthConverterT::MetricDepth,3> &extrinsics,
                                  VectorSet<typename DepthConverterT::MetricDepth,3> &points,
                                  VectorSet<float,3> &colors,
                                  bool keep_invalid = false)
    {
        using MetricDepth = typename DepthConverterT::MetricDepth;

        const Eigen::Matrix<MetricDepth,3,3,Eigen::RowMajor> intrinsics_inv = intrinsics.inverse();

        const float color_mult = 1.0f/255.0f;

        if (keep_invalid) {
            points.resize(3, image_w*image_h);
            colors.resize(3, image_w*image_h);
            size_t k;
#pragma omp parallel for private (k)
            for (size_t y = 0; y < image_h; y++) {
                k = y*image_w;
                for (size_t x = 0; x < image_w; x++) {
                    const MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                    points.col(k).noalias() = extrinsics*intrinsics_inv*Vector<MetricDepth,3>(z*x, z*y, z);
                    colors(0,k) = color_mult*static_cast<float>(rgb_data[3*k]);
                    colors(1,k) = color_mult*static_cast<float>(rgb_data[3*k + 1]);
                    colors(2,k) = color_mult*static_cast<float>(rgb_data[3*k + 2]);
                    k++;
                }
            }
        } else {
            VectorSet<MetricDepth,3> points_tmp(3, image_w*image_h);
            VectorSet<float,3> colors_tmp(3, image_w*image_h);
            size_t k;
            size_t valid_count = 0;
#pragma omp parallel for private (k) reduction (+: valid_count)
            for (size_t y = 0; y < image_h; y++) {
                k = y*image_w;
                for (size_t x = 0; x < image_w; x++) {
                    const MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                    points_tmp.col(k).noalias() = intrinsics_inv*Vector<MetricDepth,3>(z*x, z*y, z);
                    colors_tmp(0,k) = color_mult*static_cast<float>(rgb_data[3*k]);
                    colors_tmp(1,k) = color_mult*static_cast<float>(rgb_data[3*k + 1]);
                    colors_tmp(2,k) = color_mult*static_cast<float>(rgb_data[3*k + 2]);
                    k++;
                    valid_count += z > (MetricDepth)0.0;
                }
            }

            k = 0;
            points.resize(3, valid_count);
            colors.resize(3, valid_count);
            for (size_t i = 0; i < points_tmp.cols(); i++) {
                if (points_tmp(2,i) > (MetricDepth)0.0) {
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
    void RGBDImagesToPointsNormalsColors(const unsigned char* rgb_data,
                                         const typename DepthConverterT::RawDepth* depth_data,
                                         const DepthConverterT &depth_converter,
                                         size_t image_w, size_t image_h,
                                         const Eigen::Ref<const Eigen::Matrix<typename DepthConverterT::MetricDepth,3,3>> &intrinsics,
                                         VectorSet<typename DepthConverterT::MetricDepth,3> &points,
                                         VectorSet<typename DepthConverterT::MetricDepth,3> &normals,
                                         VectorSet<float,3> &colors,
                                         bool keep_invalid = false)
    {
        using MetricDepth = typename DepthConverterT::MetricDepth;

        const Eigen::Matrix<MetricDepth,3,3,Eigen::RowMajor> intrinsics_inv = intrinsics.inverse();

        const float color_mult = 1.0f/255.0f;

        if (keep_invalid) {
            points.resize(3, image_w*image_h);
            colors.resize(3, points.cols());
            normals.setConstant(3, points.cols(), std::numeric_limits<MetricDepth>::quiet_NaN());

#pragma omp parallel
            {
                size_t k;
#pragma omp for
                for (size_t y = 0; y < image_h; y++) {
                    k = y*image_w;
                    for (size_t x = 0; x < image_w; x++) {
                        const MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                        points.col(k).noalias() = intrinsics_inv*Vector<MetricDepth,3>(z*x, z*y, z);
                        colors(0,k) = color_mult*static_cast<float>(rgb_data[3*k]);
                        colors(1,k) = color_mult*static_cast<float>(rgb_data[3*k + 1]);
                        colors(2,k) = color_mult*static_cast<float>(rgb_data[3*k + 2]);
                        k++;
                    }
                }
#pragma omp for
                for (size_t y = 1; y < image_h - 1; y++) {
                    for (size_t x = 1; x < image_w - 1; x++) {
                        k = y*image_w + x;
                        if (points(2,k) > (MetricDepth)0.0 &&
                            points(2,k+1) > (MetricDepth)0.0 &&
                            points(2,k-1) > (MetricDepth)0.0 &&
                            points(2,k+image_w) > (MetricDepth)0.0 &&
                            points(2,k-image_w) > (MetricDepth)0.0)
                        {
                            normals.col(k).noalias() = (points.col(k+image_w) - points.col(k-image_w)).cross(points.col(k+1) - points.col(k-1)).normalized();
                        }
                    }
                }
            }
        } else {
            VectorSet<MetricDepth,3> points_tmp(3, image_w*image_h);
            VectorSet<float,3> colors_tmp(3, points_tmp.cols());
            VectorSet<MetricDepth,3> normals_tmp(VectorSet<MetricDepth,3>::Constant(3, points_tmp.cols(), std::numeric_limits<MetricDepth>::quiet_NaN()));
            size_t valid_count = 0;

#pragma omp parallel
            {
                size_t k;
#pragma omp for
                for (size_t y = 0; y < image_h; y++) {
                    k = y*image_w;
                    for (size_t x = 0; x < image_w; x++) {
                        const MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                        points_tmp.col(k).noalias() = intrinsics_inv*Vector<MetricDepth,3>(z*x, z*y, z);
                        colors_tmp(0,k) = color_mult*static_cast<float>(rgb_data[3*k]);
                        colors_tmp(1,k) = color_mult*static_cast<float>(rgb_data[3*k + 1]);
                        colors_tmp(2,k) = color_mult*static_cast<float>(rgb_data[3*k + 2]);
                        k++;
                    }
                }

#pragma omp for reduction (+: valid_count)
                for (size_t y = 1; y < image_h - 1; y++) {
                    for (size_t x = 1; x < image_w - 1; x++) {
                        k = y*image_w + x;
                        if (points_tmp(2,k) > (MetricDepth)0.0 &&
                            points_tmp(2,k+1) > (MetricDepth)0.0 &&
                            points_tmp(2,k-1) > (MetricDepth)0.0 &&
                            points_tmp(2,k+image_w) > (MetricDepth)0.0 &&
                            points_tmp(2,k-image_w) > (MetricDepth)0.0)
                        {
                            normals_tmp.col(k).noalias() = (points_tmp.col(k+image_w) - points_tmp.col(k-image_w)).cross(points_tmp.col(k+1) - points_tmp.col(k-1)).normalized();
                            valid_count += 1;
                        }
                    }
                }
            }

            size_t k = 0;
            points.resize(3, valid_count);
            normals.resize(3, valid_count);
            colors.resize(3, valid_count);
            for (size_t i = 0; i < points_tmp.cols(); i++) {
                if (!std::isnan(normals_tmp(0,i))) {
                    points.col(k) = points_tmp.col(i);
                    normals.col(k) = normals_tmp.col(i);
                    colors.col(k) = colors_tmp.col(i);
                    k++;
                }
            }
        }
    }

    template <class DepthConverterT>
    void RGBDImagesToPointsNormalsColors(const unsigned char* rgb_data,
                                         const typename DepthConverterT::RawDepth* depth_data,
                                         const DepthConverterT &depth_converter,
                                         size_t image_w, size_t image_h,
                                         const Eigen::Ref<const Eigen::Matrix<typename DepthConverterT::MetricDepth,3,3>> &intrinsics,
                                         const RigidTransform<typename DepthConverterT::MetricDepth,3> &extrinsics,
                                         VectorSet<typename DepthConverterT::MetricDepth,3> &points,
                                         VectorSet<typename DepthConverterT::MetricDepth,3> &normals,
                                         VectorSet<float,3> &colors,
                                         bool keep_invalid = false)
    {
        using MetricDepth = typename DepthConverterT::MetricDepth;

        const Eigen::Matrix<MetricDepth,3,3,Eigen::RowMajor> intrinsics_inv = intrinsics.inverse();

        const float color_mult = 1.0f/255.0f;

        if (keep_invalid) {
            points.resize(3, image_w*image_h);
            colors.resize(3, points.cols());
            normals.setConstant(3, points.cols(), std::numeric_limits<MetricDepth>::quiet_NaN());

#pragma omp parallel
            {
                size_t k;
#pragma omp for
                for (size_t y = 0; y < image_h; y++) {
                    k = y*image_w;
                    for (size_t x = 0; x < image_w; x++) {
                        const MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                        points.col(k).noalias() = intrinsics_inv*Vector<MetricDepth,3>(z*x, z*y, z);
                        colors(0,k) = color_mult*static_cast<float>(rgb_data[3*k]);
                        colors(1,k) = color_mult*static_cast<float>(rgb_data[3*k + 1]);
                        colors(2,k) = color_mult*static_cast<float>(rgb_data[3*k + 2]);
                        k++;
                    }
                }

#pragma omp for
                for (size_t y = 1; y < image_h - 1; y++) {
                    for (size_t x = 1; x < image_w - 1; x++) {
                        k = y*image_w + x;
                        if (points(2,k) > (MetricDepth)0.0 &&
                            points(2,k+1) > (MetricDepth)0.0 &&
                            points(2,k-1) > (MetricDepth)0.0 &&
                            points(2,k+image_w) > (MetricDepth)0.0 &&
                            points(2,k-image_w) > (MetricDepth)0.0)
                        {
                            normals.col(k).noalias() = (points.col(k+image_w) - points.col(k-image_w)).cross(points.col(k+1) - points.col(k-1)).normalized();
                        }
                    }
                }

#pragma omp for
                for (size_t i = 0; i < points.cols(); i++) {
                    points.col(i) = extrinsics*points.col(i);
                    normals.col(i) = extrinsics.linear()*normals.col(i);
                }
            }
        } else {
            VectorSet<MetricDepth,3> points_tmp(3, image_w*image_h);
            VectorSet<float,3> colors_tmp(3, points_tmp.cols());
            VectorSet<MetricDepth,3> normals_tmp(VectorSet<MetricDepth,3>::Constant(3, points_tmp.cols(), std::numeric_limits<MetricDepth>::quiet_NaN()));
            size_t valid_count = 0;

#pragma omp parallel
            {
                size_t k;
#pragma omp for
                for (size_t y = 0; y < image_h; y++) {
                    k = y*image_w;
                    for (size_t x = 0; x < image_w; x++) {
                        const MetricDepth z = depth_converter.getMetricValue(depth_data[k]);
                        points_tmp.col(k).noalias() = intrinsics_inv*Vector<MetricDepth,3>(z*x, z*y, z);
                        colors_tmp(0,k) = color_mult*static_cast<float>(rgb_data[3*k]);
                        colors_tmp(1,k) = color_mult*static_cast<float>(rgb_data[3*k + 1]);
                        colors_tmp(2,k) = color_mult*static_cast<float>(rgb_data[3*k + 2]);
                        k++;
                    }
                }

#pragma omp for reduction (+: valid_count)
                for (size_t y = 1; y < image_h - 1; y++) {
                    for (size_t x = 1; x < image_w - 1; x++) {
                        k = y*image_w + x;
                        if (points_tmp(2,k) > (MetricDepth)0.0 &&
                            points_tmp(2,k+1) > (MetricDepth)0.0 &&
                            points_tmp(2,k-1) > (MetricDepth)0.0 &&
                            points_tmp(2,k+image_w) > (MetricDepth)0.0 &&
                            points_tmp(2,k-image_w) > (MetricDepth)0.0)
                        {
                            normals_tmp.col(k).noalias() = (points_tmp.col(k+image_w) - points_tmp.col(k-image_w)).cross(points_tmp.col(k+1) - points_tmp.col(k-1)).normalized();
                            valid_count += 1;
                        }
                    }
                }
            }

            size_t k = 0;
            points.resize(3, valid_count);
            normals.resize(3, valid_count);
            colors.resize(3, valid_count);
            for (size_t i = 0; i < points_tmp.cols(); i++) {
                if (!std::isnan(normals_tmp(0,i))) {
                    points.col(k) = points_tmp.col(i);
                    normals.col(k) = normals_tmp.col(i);
                    colors.col(k) = colors_tmp.col(i);
                    k++;
                }
            }

#pragma omp parallel for
            for (size_t i = 0; i < points.cols(); i++) {
                points.col(i) = extrinsics*points.col(i);
                normals.col(i) = extrinsics.linear()*normals.col(i);
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
        using MetricDepth = typename DepthConverterT::MetricDepth;
        using RawDepth = typename DepthConverterT::RawDepth;

        const Vector<MetricDepth,3> intr0 = intrinsics.row(0);
        const Vector<MetricDepth,3> intr1 = intrinsics.row(1);

#pragma omp parallel
        {
#pragma omp for
            for (size_t i = 0; i < image_w*image_h; i++) {
                depth_data[i] = (RawDepth)0;
            }

#pragma omp for schedule(dynamic, 256)
            for (size_t i = 0; i < points.cols(); i++) {
                if (points(2,i) <= (MetricDepth)0.0) continue;
                const MetricDepth inv_z = (MetricDepth)1.0/points(2,i);
                const size_t x = (size_t)std::llround(inv_z*intr0.dot(points.col(i)));
                const size_t y = (size_t)std::llround(inv_z*intr1.dot(points.col(i)));
                if (x >= image_w || y >= image_h) continue;
                const size_t ind = y*image_w + x;
                const RawDepth depth_val = depth_converter.getRawValue(points(2,i));
                if (depth_val > (RawDepth)0 &&
                    (depth_data[ind] == (RawDepth)0 || depth_val < depth_data[ind]))
                {
                    depth_data[ind] = depth_val;
                }
            }
        }
    }

    template <class DepthConverterT>
    void pointsToDepthImage(const ConstVectorSetMatrixMap<typename DepthConverterT::MetricDepth,3> &points,
                            const RigidTransform<typename DepthConverterT::MetricDepth,3> &extrinsics,
                            const Eigen::Ref<const Eigen::Matrix<typename DepthConverterT::MetricDepth,3,3>> &intrinsics,
                            const DepthConverterT &depth_converter,
                            typename DepthConverterT::RawDepth* depth_data,
                            size_t image_w, size_t image_h)
    {
        using MetricDepth = typename DepthConverterT::MetricDepth;
        using RawDepth = typename DepthConverterT::RawDepth;

        const Vector<MetricDepth,3> intr0 = intrinsics.row(0);
        const Vector<MetricDepth,3> intr1 = intrinsics.row(1);

        const RigidTransform<MetricDepth,3> to_cam(extrinsics.inverse());

#pragma omp parallel
        {
#pragma omp for
            for (size_t i = 0; i < image_w*image_h; i++) {
                depth_data[i] = (RawDepth)0;
            }

            Vector<MetricDepth,3> pt_cam;
#pragma omp for schedule(dynamic, 256)
            for (size_t i = 0; i < points.cols(); i++) {
                pt_cam.noalias() = to_cam*points.col(i);
                if (pt_cam[2] <= (MetricDepth)0.0) continue;
                const MetricDepth inv_z = (MetricDepth)1.0/pt_cam[2];
                const size_t x = (size_t)std::llround(inv_z*intr0.dot(pt_cam));
                const size_t y = (size_t)std::llround(inv_z*intr1.dot(pt_cam));
                if (x >= image_w || y >= image_h) continue;
                const size_t ind = y*image_w + x;
                const RawDepth depth_val = depth_converter.getRawValue(pt_cam[2]);
                if (depth_val > (RawDepth)0 &&
                    (depth_data[ind] == (RawDepth)0 || depth_val < depth_data[ind]))
                {
                    depth_data[ind] = depth_val;
                }
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
        using MetricDepth = typename DepthConverterT::MetricDepth;
        using RawDepth = typename DepthConverterT::RawDepth;

        const Vector<MetricDepth,3> intr0 = intrinsics.row(0);
        const Vector<MetricDepth,3> intr1 = intrinsics.row(1);

#pragma omp parallel
        {
#pragma omp for
            for (size_t i = 0; i < image_w*image_h; i++) {
                rgb_data[3*i] = (unsigned char)0;
                rgb_data[3*i + 1] = (unsigned char)0;
                rgb_data[3*i + 2] = (unsigned char)0;
                depth_data[i] = (RawDepth)0;
            }

#pragma omp for schedule(dynamic, 256)
            for (size_t i = 0; i < points.cols(); i++) {
                if (points(2,i) <= (MetricDepth)0.0) continue;
                const MetricDepth inv_z = (MetricDepth)1.0/points(2,i);
                const size_t x = (size_t)std::llround(inv_z*intr0.dot(points.col(i)));
                const size_t y = (size_t)std::llround(inv_z*intr1.dot(points.col(i)));
                if (x >= image_w || y >= image_h) continue;
                const size_t ind = y*image_w + x;
                const RawDepth depth_val = depth_converter.getRawValue(points(2,i));
                if (depth_val > (RawDepth)0 &&
                    (depth_data[ind] == (RawDepth)0 || depth_val < depth_data[ind]))
                {
                    rgb_data[3*ind] = static_cast<unsigned char>(255.0f*colors(0,i));
                    rgb_data[3*ind + 1] = static_cast<unsigned char>(255.0f*colors(1,i));
                    rgb_data[3*ind + 2] = static_cast<unsigned char>(255.0f*colors(2,i));
                    depth_data[ind] = depth_val;
                }
            }
        }
    }

    template <class DepthConverterT>
    void pointsColorsToRGBDImages(const ConstVectorSetMatrixMap<typename DepthConverterT::MetricDepth,3> &points,
                                  const ConstVectorSetMatrixMap<float,3> &colors,
                                  const RigidTransform<typename DepthConverterT::MetricDepth,3> &extrinsics,
                                  const Eigen::Ref<const Eigen::Matrix<typename DepthConverterT::MetricDepth,3,3>> &intrinsics,
                                  const DepthConverterT &depth_converter,
                                  unsigned char* rgb_data,
                                  typename DepthConverterT::RawDepth* depth_data,
                                  size_t image_w, size_t image_h)
    {
        using MetricDepth = typename DepthConverterT::MetricDepth;
        using RawDepth = typename DepthConverterT::RawDepth;

        const Vector<MetricDepth,3> intr0 = intrinsics.row(0);
        const Vector<MetricDepth,3> intr1 = intrinsics.row(1);

        const RigidTransform<MetricDepth,3> to_cam(extrinsics.inverse());

#pragma omp parallel
        {
#pragma omp for
            for (size_t i = 0; i < image_w*image_h; i++) {
                rgb_data[3*i] = (unsigned char)0;
                rgb_data[3*i + 1] = (unsigned char)0;
                rgb_data[3*i + 2] = (unsigned char)0;
                depth_data[i] = (RawDepth)0;
            }

            Vector<MetricDepth,3> pt_cam;
#pragma omp for schedule(dynamic, 256)
            for (size_t i = 0; i < points.cols(); i++) {
                pt_cam.noalias() = to_cam*points.col(i);
                if (pt_cam[2] <= (MetricDepth)0.0) continue;
                const MetricDepth inv_z = (MetricDepth)1.0/pt_cam[2];
                const size_t x = (size_t)std::llround(inv_z*intr0.dot(pt_cam));
                const size_t y = (size_t)std::llround(inv_z*intr1.dot(pt_cam));
                if (x >= image_w || y >= image_h) continue;
                const size_t ind = y*image_w + x;
                const RawDepth depth_val = depth_converter.getRawValue(pt_cam[2]);
                if (depth_val > (RawDepth)0 &&
                    (depth_data[ind] == (RawDepth)0 || depth_val < depth_data[ind]))
                {
                    rgb_data[3*ind] = static_cast<unsigned char>(255.0f*colors(0,i));
                    rgb_data[3*ind + 1] = static_cast<unsigned char>(255.0f*colors(1,i));
                    rgb_data[3*ind + 2] = static_cast<unsigned char>(255.0f*colors(2,i));
                    depth_data[ind] = depth_val;
                }
            }
        }
    }

    template <typename PointT, typename IndexT = size_t>
    void pointsToIndexMap(const ConstVectorSetMatrixMap<PointT,3> &points,
                          const Eigen::Ref<const Eigen::Matrix<PointT,3,3>> &intrinsics,
                          IndexT* index_map_data,
                          size_t image_w, size_t image_h)
    {
        const IndexT empty = std::numeric_limits<IndexT>::max();

        const Vector<PointT,3> intr0 = intrinsics.row(0);
        const Vector<PointT,3> intr1 = intrinsics.row(1);

#pragma omp parallel
        {
#pragma omp for
            for (size_t i = 0; i < image_w*image_h; i++) {
                index_map_data[i] = empty;
            }

#pragma omp for schedule(dynamic, 256)
            for (size_t i = 0; i < points.cols(); i++) {
                if (points(2,i) <= (PointT)0.0) continue;
                const PointT inv_z = (PointT)1.0/points(2,i);
                const size_t x = (size_t)std::llround(inv_z*intr0.dot(points.col(i)));
                const size_t y = (size_t)std::llround(inv_z*intr1.dot(points.col(i)));
                if (x >= image_w || y >= image_h) continue;
                const size_t ind = y*image_w + x;
                if (index_map_data[ind] == empty || points(2,i) < points(2,index_map_data[ind])) {
                    index_map_data[ind] = static_cast<IndexT>(i);
                }
            }
        }
    }

    template <typename PointT, typename IndexT = size_t>
    void pointsToIndexMap(const ConstVectorSetMatrixMap<PointT,3> &points,
                          const RigidTransform<PointT,3> &extrinsics,
                          const Eigen::Ref<const Eigen::Matrix<PointT,3,3>> &intrinsics,
                          IndexT* index_map_data,
                          size_t image_w, size_t image_h)
    {
        const IndexT empty = std::numeric_limits<IndexT>::max();

        const Vector<PointT,3> intr0 = intrinsics.row(0);
        const Vector<PointT,3> intr1 = intrinsics.row(1);

        const RigidTransform<PointT,3> to_cam(extrinsics.inverse());

        VectorSet<PointT,3> points_cam(3, points.cols());
#pragma omp parallel
        {
#pragma omp for
            for (size_t i = 0; i < image_w*image_h; i++) {
                index_map_data[i] = empty;
            }

#pragma omp for
            for (size_t i = 0; i < points.cols(); i++) {
                points_cam.col(i).noalias() = to_cam*points.col(i);
            }

#pragma omp for schedule(dynamic, 256)
            for (size_t i = 0; i < points.cols(); i++) {
                if (points_cam(2,i) <= (PointT)0.0) continue;
                const PointT inv_z = (PointT)1.0/points_cam(2,i);
                const size_t x = (size_t)std::llround(inv_z*intr0.dot(points_cam.col(i)));
                const size_t y = (size_t)std::llround(inv_z*intr1.dot(points_cam.col(i)));
                if (x >= image_w || y >= image_h) continue;
                const size_t ind = y*image_w + x;
                if (index_map_data[ind] == empty || points_cam(2,i) < points_cam(2,index_map_data[ind])) {
                    index_map_data[ind] = static_cast<IndexT>(i);
                }
            }
        }
    }
}
