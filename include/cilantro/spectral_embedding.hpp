#pragma once

#include <cilantro/data_containers.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim = Eigen::Dynamic>
    struct SpectralEmbedding {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum { Dimension = EigenDim };

        VectorSet<ScalarT,EigenDim> embeddedPoints;

        Vector<ScalarT,EigenDim> computedEigenvalues;

        inline const VectorSet<ScalarT,EigenDim>& getEmbeddedPoints() const { return embeddedPoints; }

        inline const Vector<ScalarT,EigenDim>& getComputedEigenValues() const { return computedEigenvalues; }
    };
}
