#pragma once

#include <cilantro/core/data_containers.hpp>

namespace cilantro {
    // CRTP base class that holds computed embedding and accessors
    template <typename Derived, typename ScalarT, ptrdiff_t EigenDim = Eigen::Dynamic>
    class SpectralEmbeddingBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum { Dimension = EigenDim };

        inline const VectorSet<ScalarT,EigenDim>& getEmbeddedPoints() const { return embedded_points_; }

        inline const Vector<ScalarT,EigenDim>& getComputedEigenValues() const { return computed_eigenvalues_; }

        inline size_t getEmbeddingDimension() const { return embedded_points_.rows(); }

    protected:
        VectorSet<ScalarT,EigenDim> embedded_points_;
        Vector<ScalarT,EigenDim> computed_eigenvalues_;
    };
}
