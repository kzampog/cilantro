#pragma once

#include <cilantro/data_containers.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class PointsProximityEvaluator {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PointsProximityEvaluator(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                 ScalarT dist_thresh)
                : points_(points), max_distance_(dist_thresh)
        {}

        inline bool operator()(size_t, size_t, ScalarT dist) const { return dist < max_distance_; }

    private:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> points_;
        ScalarT max_distance_;
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    class NormalsProximityEvaluator {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        NormalsProximityEvaluator(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &normals,
                                  ScalarT angle_thresh)
                : normals_(normals), max_angle_(angle_thresh)
        {}

        inline bool operator()(size_t i, size_t j, ScalarT) const {
            const ScalarT angle = std::acos(normals_.col(i).dot(normals_.col(j)));
            if (max_angle_ >= (ScalarT)0.0) {
                return angle <= max_angle_;
            } else {
                return std::min(angle, (ScalarT)M_PI - angle) <= -max_angle_;
            }
        }

    private:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> normals_;
        ScalarT max_angle_;
    };

    template <typename ScalarT>
    class ColorsProximityEvaluator {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ColorsProximityEvaluator(const ConstVectorSetMatrixMap<float,3> &colors,
                                 float dist_thresh)
                : colors_(colors), max_color_diff_(dist_thresh*dist_thresh)
        {}

        inline bool operator()(size_t i, size_t j, ScalarT) const { return (colors_.col(i) - colors_.col(j)).squaredNorm() < max_color_diff_; }

    private:
        ConstVectorSetMatrixMap<float,3> colors_;
        float max_color_diff_;
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    class PointsNormalsProximityEvaluator {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PointsNormalsProximityEvaluator(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                        const ConstVectorSetMatrixMap<ScalarT,EigenDim> &normals,
                                        ScalarT dist_thresh,
                                        ScalarT angle_thresh)
                : points_(points), normals_(normals),
                  max_distance_(dist_thresh), max_angle_(angle_thresh)
        {}

        inline bool operator()(size_t i, size_t j, ScalarT dist) const {
            if (dist >= max_distance_) return false;
            const ScalarT angle = std::acos(normals_.col(i).dot(normals_.col(j)));
            if (max_angle_ >= (ScalarT)0.0) {
                return angle < max_angle_;
            } else {
                return std::min(angle, (ScalarT)M_PI - angle) < -max_angle_;
            }
        }

    private:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> points_;
        ConstVectorSetMatrixMap<ScalarT,EigenDim> normals_;
        ScalarT max_distance_;
        ScalarT max_angle_;
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    class PointsColorsProximityEvaluator {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PointsColorsProximityEvaluator(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                       const ConstVectorSetMatrixMap<float,3> &colors,
                                       ScalarT dist_thresh,
                                       float color_thresh)
                : points_(points), colors_(colors),
                  max_distance_(dist_thresh), max_color_diff_(color_thresh*color_thresh)
        {}

        inline bool operator()(size_t i, size_t j, ScalarT dist) const {
            return (dist < max_distance_) && ((colors_.col(i) - colors_.col(j)).squaredNorm() < max_color_diff_);
        }

    private:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> points_;
        ConstVectorSetMatrixMap<float,3> colors_;
        ScalarT max_distance_;
        float max_color_diff_;
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    class NormalsColorsProximityEvaluator {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        NormalsColorsProximityEvaluator(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &normals,
                                        const ConstVectorSetMatrixMap<float,3> &colors,
                                        ScalarT angle_thresh,
                                        float color_thresh)
                : normals_(normals), colors_(colors),
                  max_angle_(angle_thresh), max_color_diff_(color_thresh*color_thresh)
        {}

        inline bool operator()(size_t i, size_t j, ScalarT) const {
            if ((colors_.col(i) - colors_.col(j)).squaredNorm() >= max_color_diff_) return false;
            const ScalarT angle = std::acos(normals_.col(i).dot(normals_.col(j)));
            if (max_angle_ >= (ScalarT)0.0) {
                return angle < max_angle_;
            } else {
                return std::min(angle, (ScalarT)M_PI - angle) < -max_angle_;
            }
        }

    private:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> normals_;
        ConstVectorSetMatrixMap<float,3> colors_;
        ScalarT max_angle_;
        float max_color_diff_;
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    class PointsNormalsColorsProximityEvaluator {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PointsNormalsColorsProximityEvaluator(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                              const ConstVectorSetMatrixMap<ScalarT,EigenDim> &normals,
                                              const ConstVectorSetMatrixMap<float,3> &colors,
                                              ScalarT dist_thresh,
                                              ScalarT angle_thresh,
                                              float color_thresh)
                : points_(points), normals_(normals), colors_(colors),
                  max_distance_(dist_thresh), max_angle_(angle_thresh), max_color_diff_(color_thresh*color_thresh)
        {}

        inline bool operator()(size_t i, size_t j, ScalarT dist) const {
            if (dist >= max_distance_ || (colors_.col(i) - colors_.col(j)).squaredNorm() >= max_color_diff_) return false;
            const ScalarT angle = std::acos(normals_.col(i).dot(normals_.col(j)));
            if (max_angle_ >= (ScalarT)0.0) {
                return angle < max_angle_;
            } else {
                return std::min(angle, (ScalarT)M_PI - angle) < -max_angle_;
            }
        }

    private:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> points_;
        ConstVectorSetMatrixMap<ScalarT,EigenDim> normals_;
        ConstVectorSetMatrixMap<float,3> colors_;
        ScalarT max_distance_;
        ScalarT max_angle_;
        float max_color_diff_;
    };
}
