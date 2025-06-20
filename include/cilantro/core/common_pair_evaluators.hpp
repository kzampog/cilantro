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
  using InputScalar = ValueT;
  using OutputScalar = WeightT;

  inline WeightT operator()(ValueT val) const { return static_cast<WeightT>(val); }

  template <class PointT>
  inline WeightT operator()(const PointT&, const PointT&, ValueT val) const {
    return static_cast<WeightT>(val);
  }

  inline WeightT operator()(size_t, size_t, ValueT val) const { return static_cast<WeightT>(val); }
};

template <typename ValueT, typename WeightT = ValueT>
class UnityWeightEvaluator {
public:
  using InputScalar = ValueT;
  using OutputScalar = WeightT;

  inline constexpr WeightT operator()(ValueT) const { return (WeightT)1; }

  template <class PointT>
  inline constexpr WeightT operator()(const PointT&, const PointT&, ValueT) const {
    return (WeightT)1;
  }

  inline constexpr WeightT operator()(size_t, size_t, ValueT) const { return (WeightT)1; }
};

template <typename ValueT, typename WeightT = ValueT, bool distances_are_squared = true>
class RBFKernelWeightEvaluator {
public:
  using InputScalar = ValueT;
  using OutputScalar = WeightT;

  RBFKernelWeightEvaluator() : coeff_(-(WeightT)(0.5)) {}

  RBFKernelWeightEvaluator(ValueT sigma) : coeff_(-(WeightT)(0.5) / (sigma * sigma)) {}

  inline RBFKernelWeightEvaluator& setSigma(ValueT sigma) {
    coeff_ = -(WeightT)(0.5) / (sigma * sigma);
    return *this;
  }

  inline WeightT operator()(ValueT dist) const {
    if constexpr (distances_are_squared) {
      return std::exp(coeff_ * static_cast<WeightT>(dist));
    } else {
      return std::exp(coeff_ * static_cast<WeightT>(dist * dist));
    }
  }

  template <class PointT>
  inline WeightT operator()(const PointT&, const PointT&, ValueT dist) const {
    if constexpr (distances_are_squared) {
      return std::exp(coeff_ * static_cast<WeightT>(dist));
    } else {
      return std::exp(coeff_ * static_cast<WeightT>(dist * dist));
    }
  }

private:
  WeightT coeff_;
};

template <typename ValueT, typename WeightT = ValueT>
using DistanceEvaluator = IdentityWeightEvaluator<ValueT, WeightT>;

template <typename ValueT, typename WeightT = ValueT>
using AdjacencyEvaluator = UnityWeightEvaluator<ValueT, WeightT>;

template <typename ValueT>
using AlwaysTrueEvaluator = UnityWeightEvaluator<ValueT, bool>;

// Proximity evaluators (return bool)

template <typename ScalarT>
class PointsProximityEvaluator {
public:
  using InputScalar = ScalarT;
  using OutputScalar = bool;

  PointsProximityEvaluator(ScalarT dist_thresh) : max_distance_(dist_thresh) {}

  inline bool operator()(size_t, size_t, ScalarT dist) const { return dist < max_distance_; }

private:
  ScalarT max_distance_;
};

template <typename ScalarT, ptrdiff_t EigenDim>
class NormalsProximityEvaluator {
public:
  using InputScalar = ScalarT;
  using OutputScalar = bool;

  NormalsProximityEvaluator(const ConstVectorSetMatrixMap<ScalarT, EigenDim>& normals,
                            ScalarT angle_thresh)
      : normals_(normals), max_angle_(angle_thresh) {}

  inline bool operator()(size_t i, size_t j, ScalarT) const {
    const ScalarT angle = std::acos(normals_.col(i).dot(normals_.col(j)));
    if (max_angle_ >= (ScalarT)0.0) {
      return angle <= max_angle_;
    } else {
      return std::min(angle, (ScalarT)M_PI - angle) <= -max_angle_;
    }
  }

private:
  ConstVectorSetMatrixMap<ScalarT, EigenDim> normals_;
  ScalarT max_angle_;
};

template <typename ScalarT>
class ColorsProximityEvaluator {
public:
  using InputScalar = ScalarT;
  using OutputScalar = bool;

  ColorsProximityEvaluator(const ConstVectorSetMatrixMap<float, 3>& colors, float dist_thresh)
      : colors_(colors), max_color_diff_(dist_thresh * dist_thresh) {}

  inline bool operator()(size_t i, size_t j, ScalarT) const {
    return (colors_.col(i) - colors_.col(j)).squaredNorm() < max_color_diff_;
  }

private:
  ConstVectorSetMatrixMap<float, 3> colors_;
  float max_color_diff_;
};

template <typename ScalarT, ptrdiff_t EigenDim>
class PointsNormalsProximityEvaluator {
public:
  using InputScalar = ScalarT;
  using OutputScalar = bool;

  PointsNormalsProximityEvaluator(const ConstVectorSetMatrixMap<ScalarT, EigenDim>& normals,
                                  ScalarT dist_thresh, ScalarT angle_thresh)
      : normals_(normals), max_distance_(dist_thresh), max_angle_(angle_thresh) {}

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
  ConstVectorSetMatrixMap<ScalarT, EigenDim> normals_;
  ScalarT max_distance_;
  ScalarT max_angle_;
};

template <typename ScalarT, ptrdiff_t EigenDim>
class PointsColorsProximityEvaluator {
public:
  using InputScalar = ScalarT;
  using OutputScalar = bool;

  PointsColorsProximityEvaluator(const ConstVectorSetMatrixMap<float, 3>& colors,
                                 ScalarT dist_thresh, float color_thresh)
      : colors_(colors), max_distance_(dist_thresh), max_color_diff_(color_thresh * color_thresh) {}

  inline bool operator()(size_t i, size_t j, ScalarT dist) const {
    return (dist < max_distance_) &&
           ((colors_.col(i) - colors_.col(j)).squaredNorm() < max_color_diff_);
  }

private:
  ConstVectorSetMatrixMap<float, 3> colors_;
  ScalarT max_distance_;
  float max_color_diff_;
};

template <typename ScalarT, ptrdiff_t EigenDim>
class NormalsColorsProximityEvaluator {
public:
  using InputScalar = ScalarT;
  using OutputScalar = bool;

  NormalsColorsProximityEvaluator(const ConstVectorSetMatrixMap<ScalarT, EigenDim>& normals,
                                  const ConstVectorSetMatrixMap<float, 3>& colors,
                                  ScalarT angle_thresh, float color_thresh)
      : normals_(normals),
        colors_(colors),
        max_angle_(angle_thresh),
        max_color_diff_(color_thresh * color_thresh) {}

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
  ConstVectorSetMatrixMap<ScalarT, EigenDim> normals_;
  ConstVectorSetMatrixMap<float, 3> colors_;
  ScalarT max_angle_;
  float max_color_diff_;
};

template <typename ScalarT, ptrdiff_t EigenDim>
class PointsNormalsColorsProximityEvaluator {
public:
  using InputScalar = ScalarT;
  using OutputScalar = bool;

  PointsNormalsColorsProximityEvaluator(const ConstVectorSetMatrixMap<ScalarT, EigenDim>& normals,
                                        const ConstVectorSetMatrixMap<float, 3>& colors,
                                        ScalarT dist_thresh, ScalarT angle_thresh,
                                        float color_thresh)
      : normals_(normals),
        colors_(colors),
        max_distance_(dist_thresh),
        max_angle_(angle_thresh),
        max_color_diff_(color_thresh * color_thresh) {}

  inline bool operator()(size_t i, size_t j, ScalarT dist) const {
    if (dist >= max_distance_ || (colors_.col(i) - colors_.col(j)).squaredNorm() >= max_color_diff_)
      return false;
    const ScalarT angle = std::acos(normals_.col(i).dot(normals_.col(j)));
    if (max_angle_ >= (ScalarT)0.0) {
      return angle < max_angle_;
    } else {
      return std::min(angle, (ScalarT)M_PI - angle) < -max_angle_;
    }
  }

private:
  ConstVectorSetMatrixMap<ScalarT, EigenDim> normals_;
  ConstVectorSetMatrixMap<float, 3> colors_;
  ScalarT max_distance_;
  ScalarT max_angle_;
  float max_color_diff_;
};

}  // namespace cilantro
