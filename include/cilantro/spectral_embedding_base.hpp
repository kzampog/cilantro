#pragma once

#include <cilantro/data_containers.hpp>

namespace cilantro {
    // CRTP base class that holds computed embedding and accessors
    template <typename ScalarT, ptrdiff_t EigenDim = Eigen::Dynamic, typename Derived = void>
    struct SpectralEmbeddingBase {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum { Dimension = EigenDim };

        VectorSet<ScalarT,EigenDim> embeddedPoints;

        Vector<ScalarT,EigenDim> computedEigenvalues;

        inline const VectorSet<ScalarT,EigenDim>& getEmbeddedPoints() const { return embeddedPoints; }

        inline const Vector<ScalarT,EigenDim>& getComputedEigenValues() const { return computedEigenvalues; }

        inline size_t getEmbeddingDimension() const { return embeddedPoints.rows(); }
    };
}
