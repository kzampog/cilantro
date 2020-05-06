#pragma once

#include <cilantro/core/data_containers.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cilantro {
    // Weight evaluators (return a scalar weight)

    template <typename ValueT, typename WeightT = ValueT>
    class IdentityWeightEvaluator {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ValueT InputScalar;
        typedef WeightT OutputScalar;

        inline WeightT operator()(ValueT val) const { return static_cast<WeightT>(val); }

        template <class PointT>
        inline WeightT operator()(const PointT&, const PointT&, ValueT val) const { return static_cast<WeightT>(val); }

        inline WeightT operator()(size_t, size_t, ValueT val) const { return static_cast<WeightT>(val); }
    };

    template <typename ValueT, typename WeightT = ValueT>
    class UnityWeightEvaluator {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ValueT InputScalar;
        typedef WeightT OutputScalar;

        inline constexpr WeightT operator()(ValueT) const { return (WeightT)1; }

        template <class PointT>
        inline constexpr WeightT operator()(const PointT&, const PointT&, ValueT) const { return (WeightT)1; }

        inline constexpr WeightT operator()(size_t, size_t, ValueT) const { return (WeightT)1; }
    };

    template <typename ValueT, typename WeightT = ValueT, bool distances_are_squared = true>
    class RBFKernelWeightEvaluator {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ValueT InputScalar;
        typedef WeightT OutputScalar;

        RBFKernelWeightEvaluator() : coeff_(-(WeightT)(0.5)) {}

        RBFKernelWeightEvaluator(ValueT sigma) : coeff_(-(WeightT)(0.5)/(sigma*sigma)) {}

        inline RBFKernelWeightEvaluator& setSigma(ValueT sigma) {
            coeff_ = -(WeightT)(0.5)/(sigma*sigma);
            return *this;
        }

        template <bool dist_sq = distances_are_squared>
        inline typename std::enable_if<dist_sq,WeightT>::type operator()(ValueT dist) const {
            return std::exp(coeff_*static_cast<WeightT>(dist));
        }

        template <bool dist_sq = distances_are_squared>
        inline typename std::enable_if<!dist_sq,WeightT>::type operator()(ValueT dist) const {
            return std::exp(coeff_*static_cast<WeightT>(dist*dist));
        }

        template <class PointT, bool dist_sq = distances_are_squared>
        inline typename std::enable_if<dist_sq,WeightT>::type operator()(const PointT&, const PointT&, ValueT dist) const {
            return std::exp(coeff_*static_cast<WeightT>(dist));
        }

        template <class PointT, bool dist_sq = distances_are_squared>
        inline typename std::enable_if<!dist_sq,WeightT>::type operator()(const PointT&, const PointT&, ValueT dist) const {
            return std::exp(coeff_*static_cast<WeightT>(dist*dist));
        }

        template <bool dist_sq = distances_are_squared>
        inline typename std::enable_if<dist_sq,WeightT>::type operator()(size_t, size_t, ValueT dist) const {
            return std::exp(coeff_*static_cast<WeightT>(dist));
        }

        template <bool dist_sq = distances_are_squared>
        inline typename std::enable_if<!dist_sq,WeightT>::type operator()(size_t, size_t, ValueT dist) const {
            return std::exp(coeff_*static_cast<WeightT>(dist*dist));
        }

    private:
        WeightT coeff_;
    };

    template <typename ValueT, typename WeightT = ValueT>
    using DistanceEvaluator = IdentityWeightEvaluator<ValueT,WeightT>;

    template <typename ValueT, typename WeightT = ValueT>
    using AdjacencyEvaluator = UnityWeightEvaluator<ValueT,WeightT>;

    template <typename ValueT>
    using AlwaysTrueEvaluator = UnityWeightEvaluator<ValueT,bool>;

    // Proximity evaluators (return bool)

    template <typename ScalarT>
    class PointsProximityEvaluator {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT InputScalar;
        typedef bool OutputScalar;

        PointsProximityEvaluator(ScalarT dist_thresh) : max_distance_(dist_thresh) {}

        inline bool operator()(size_t, size_t, ScalarT dist) const { return dist < max_distance_; }

    private:
        ScalarT max_distance_;
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    class NormalsProximityEvaluator {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT InputScalar;
        typedef bool OutputScalar;

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

        typedef ScalarT InputScalar;
        typedef bool OutputScalar;

        ColorsProximityEvaluator(const ConstVectorSetMatrixMap<float,3> &colors,
                                 float dist_thresh)
                : colors_(colors), max_color_diff_(dist_thresh*dist_thresh)
        {}

        inline bool operator()(size_t i, size_t j, ScalarT) const {
            return (colors_.col(i) - colors_.col(j)).squaredNorm() < max_color_diff_;
        }

    private:
        ConstVectorSetMatrixMap<float,3> colors_;
        float max_color_diff_;
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    class PointsNormalsProximityEvaluator {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT InputScalar;
        typedef bool OutputScalar;

        PointsNormalsProximityEvaluator(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &normals,
                                        ScalarT dist_thresh,
                                        ScalarT angle_thresh)
                : normals_(normals), max_distance_(dist_thresh), max_angle_(angle_thresh)
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
        ConstVectorSetMatrixMap<ScalarT,EigenDim> normals_;
        ScalarT max_distance_;
        ScalarT max_angle_;
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    class PointsColorsProximityEvaluator {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT InputScalar;
        typedef bool OutputScalar;

        PointsColorsProximityEvaluator(const ConstVectorSetMatrixMap<float,3> &colors,
                                       ScalarT dist_thresh,
                                       float color_thresh)
                : colors_(colors), max_distance_(dist_thresh), max_color_diff_(color_thresh*color_thresh)
        {}

        inline bool operator()(size_t i, size_t j, ScalarT dist) const {
            return (dist < max_distance_) && ((colors_.col(i) - colors_.col(j)).squaredNorm() < max_color_diff_);
        }

    private:
        ConstVectorSetMatrixMap<float,3> colors_;
        ScalarT max_distance_;
        float max_color_diff_;
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    class NormalsColorsProximityEvaluator {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT InputScalar;
        typedef bool OutputScalar;

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

        typedef ScalarT InputScalar;
        typedef bool OutputScalar;

        PointsNormalsColorsProximityEvaluator(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &normals,
                                              const ConstVectorSetMatrixMap<float,3> &colors,
                                              ScalarT dist_thresh,
                                              ScalarT angle_thresh,
                                              float color_thresh)
                : normals_(normals), colors_(colors),
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
        ConstVectorSetMatrixMap<ScalarT,EigenDim> normals_;
        ConstVectorSetMatrixMap<float,3> colors_;
        ScalarT max_distance_;
        ScalarT max_angle_;
        float max_color_diff_;
    };
}
