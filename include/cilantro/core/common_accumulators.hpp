#pragma once

#include <cilantro/core/data_containers.hpp>

namespace cilantro {

template <typename IndexT = size_t>
struct IndexAccumulator {
  using Index = IndexT;

  inline IndexAccumulator() {}

  inline IndexAccumulator(IndexT i) : indices(1, i) {}

  inline IndexAccumulator& mergeWith(const IndexAccumulator& other) {
    indices.insert(indices.end(), other.indices.begin(), other.indices.end());
    return *this;
  }

  std::vector<IndexT> indices;
};

template <typename IndexT = size_t>
class IndexAccumulatorProxy {
public:
  using Accumulator = IndexAccumulator<IndexT>;

  inline Accumulator buildAccumulator(IndexT i) const { return Accumulator(i); }

  inline Accumulator& addToAccumulator(Accumulator& accum, IndexT i) const {
    accum.indices.emplace_back(i);
    return accum;
  }
};

template <typename ScalarT, ptrdiff_t EigenDim>
struct PointSumAccumulator {
  using Scalar = ScalarT;

  enum { Dimension = EigenDim };

  inline PointSumAccumulator(size_t dim = 0)
      : pointSum(Vector<ScalarT, EigenDim>::Zero(dim, 1)), pointCount(0) {}

  inline PointSumAccumulator(const Eigen::Ref<const Vector<ScalarT, EigenDim>>& point)
      : pointSum(point), pointCount(1) {}

  inline PointSumAccumulator& mergeWith(const PointSumAccumulator& other) {
    pointSum += other.pointSum;
    pointCount += other.pointCount;
    return *this;
  }

  Vector<ScalarT, EigenDim> pointSum;
  size_t pointCount;
};

template <typename ScalarT, ptrdiff_t EigenDim>
class PointSumAccumulatorProxy {
public:
  using Accumulator = PointSumAccumulator<ScalarT, EigenDim>;

  inline PointSumAccumulatorProxy(const ConstVectorSetMatrixMap<ScalarT, EigenDim>& points)
      : points_(points) {}

  inline Accumulator buildAccumulator(size_t i) const { return Accumulator(points_.col(i)); }

  inline Accumulator& addToAccumulator(Accumulator& accum, size_t i) const {
    accum.pointSum += points_.col(i);
    accum.pointCount++;
    return accum;
  }

private:
  ConstVectorSetMatrixMap<ScalarT, EigenDim> points_;
};

template <typename ScalarT, ptrdiff_t EigenDim>
struct PointNormalSumAccumulator {
  using Scalar = ScalarT;

  enum { Dimension = EigenDim };

  inline PointNormalSumAccumulator(size_t dim = 0)
      : pointSum(Vector<ScalarT, EigenDim>::Zero(dim, 1)),
        normalSum(Vector<ScalarT, EigenDim>::Zero(dim, 1)),
        pointCount(0) {}

  inline PointNormalSumAccumulator(const Eigen::Ref<const Vector<ScalarT, EigenDim>>& point,
                                   const Eigen::Ref<const Vector<ScalarT, EigenDim>>& normal)
      : pointSum(point), normalSum(normal), pointCount(1) {}

  inline PointNormalSumAccumulator& mergeWith(const PointNormalSumAccumulator& other) {
    pointSum += other.pointSum;
    if (normalSum.dot(other.normalSum) < (ScalarT)0.0) {
      normalSum -= other.normalSum;
    } else {
      normalSum += other.normalSum;
    }
    pointCount += other.pointCount;
    return *this;
  }

  Vector<ScalarT, EigenDim> pointSum;
  Vector<ScalarT, EigenDim> normalSum;
  size_t pointCount;
};

template <typename ScalarT, ptrdiff_t EigenDim>
class PointNormalSumAccumulatorProxy {
public:
  using Accumulator = PointNormalSumAccumulator<ScalarT, EigenDim>;

  inline PointNormalSumAccumulatorProxy(const ConstVectorSetMatrixMap<ScalarT, EigenDim>& points,
                                        const ConstVectorSetMatrixMap<ScalarT, EigenDim>& normals)
      : points_(points), normals_(normals) {}

  inline Accumulator buildAccumulator(size_t i) const {
    return Accumulator(points_.col(i), normals_.col(i));
  }

  inline Accumulator& addToAccumulator(Accumulator& accum, size_t i) const {
    accum.pointSum += points_.col(i);
    if (accum.normalSum.dot(normals_.col(i)) < (ScalarT)0.0) {
      accum.normalSum -= normals_.col(i);
    } else {
      accum.normalSum += normals_.col(i);
    }
    accum.pointCount++;
    return accum;
  }

private:
  ConstVectorSetMatrixMap<ScalarT, EigenDim> points_;
  ConstVectorSetMatrixMap<ScalarT, EigenDim> normals_;
};

template <typename ScalarT, ptrdiff_t EigenDim>
struct PointColorSumAccumulator {
  using Scalar = ScalarT;

  enum { Dimension = EigenDim };

  inline PointColorSumAccumulator(size_t dim = 0)
      : pointSum(Vector<ScalarT, EigenDim>::Zero(dim, 1)),
        colorSum(Vector<float, 3>::Zero()),
        pointCount(0) {}

  inline PointColorSumAccumulator(const Eigen::Ref<const Vector<ScalarT, EigenDim>>& point,
                                  const Eigen::Ref<const Vector<float, 3>>& color)
      : pointSum(point), colorSum(color), pointCount(1) {}

  inline PointColorSumAccumulator& mergeWith(const PointColorSumAccumulator& other) {
    pointSum += other.pointSum;
    colorSum += other.colorSum;
    pointCount += other.pointCount;
    return *this;
  }

  Vector<ScalarT, EigenDim> pointSum;
  Vector<float, 3> colorSum;
  size_t pointCount;
};

template <typename ScalarT, ptrdiff_t EigenDim>
class PointColorSumAccumulatorProxy {
public:
  using Accumulator = PointColorSumAccumulator<ScalarT, EigenDim>;

  inline PointColorSumAccumulatorProxy(const ConstVectorSetMatrixMap<ScalarT, EigenDim>& points,
                                       const ConstVectorSetMatrixMap<float, 3>& colors)
      : points_(points), colors_(colors) {}

  inline Accumulator buildAccumulator(size_t i) const {
    return Accumulator(points_.col(i), colors_.col(i));
  }

  inline Accumulator& addToAccumulator(Accumulator& accum, size_t i) const {
    accum.pointSum += points_.col(i);
    accum.colorSum += colors_.col(i);
    accum.pointCount++;
    return accum;
  }

private:
  ConstVectorSetMatrixMap<ScalarT, EigenDim> points_;
  ConstVectorSetMatrixMap<float, 3> colors_;
};

template <typename ScalarT, ptrdiff_t EigenDim>
struct PointNormalColorSumAccumulator {
  using Scalar = ScalarT;

  enum { Dimension = EigenDim };

  inline PointNormalColorSumAccumulator(size_t dim = 0)
      : pointSum(Vector<ScalarT, EigenDim>::Zero(dim, 1)),
        normalSum(Vector<ScalarT, EigenDim>::Zero(dim, 1)),
        colorSum(Vector<float, 3>::Zero()),
        pointCount(0) {}

  inline PointNormalColorSumAccumulator(const Eigen::Ref<const Vector<ScalarT, EigenDim>>& point,
                                        const Eigen::Ref<const Vector<ScalarT, EigenDim>>& normal,
                                        const Eigen::Ref<const Vector<float, 3>>& color)
      : pointSum(point), normalSum(normal), colorSum(color), pointCount(1) {}

  inline PointNormalColorSumAccumulator& mergeWith(const PointNormalColorSumAccumulator& other) {
    pointSum += other.pointSum;
    if (normalSum.dot(other.normalSum) < (ScalarT)0.0) {
      normalSum -= other.normalSum;
    } else {
      normalSum += other.normalSum;
    }
    colorSum += other.colorSum;
    pointCount += other.pointCount;
    return *this;
  }

  Vector<ScalarT, EigenDim> pointSum;
  Vector<ScalarT, EigenDim> normalSum;
  Vector<float, 3> colorSum;
  size_t pointCount;
};

template <typename ScalarT, ptrdiff_t EigenDim>
class PointNormalColorSumAccumulatorProxy {
public:
  using Accumulator = PointNormalColorSumAccumulator<ScalarT, EigenDim>;

  inline PointNormalColorSumAccumulatorProxy(
      const ConstVectorSetMatrixMap<ScalarT, EigenDim>& points,
      const ConstVectorSetMatrixMap<ScalarT, EigenDim>& normals,
      const ConstVectorSetMatrixMap<float, 3>& colors)
      : points_(points), normals_(normals), colors_(colors) {}

  inline Accumulator buildAccumulator(size_t i) const {
    return Accumulator(points_.col(i), normals_.col(i), colors_.col(i));
  }

  inline Accumulator& addToAccumulator(Accumulator& accum, size_t i) const {
    accum.pointSum += points_.col(i);
    if (accum.normalSum.dot(normals_.col(i)) < (ScalarT)0.0) {
      accum.normalSum -= normals_.col(i);
    } else {
      accum.normalSum += normals_.col(i);
    }
    accum.colorSum += colors_.col(i);
    accum.pointCount++;
    return accum;
  }

private:
  ConstVectorSetMatrixMap<ScalarT, EigenDim> points_;
  ConstVectorSetMatrixMap<ScalarT, EigenDim> normals_;
  ConstVectorSetMatrixMap<float, 3> colors_;
};

}  // namespace cilantro
