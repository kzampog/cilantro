#pragma once

#include <cilantro/3rd_party/spectra/SymEigsSolver.h>
#include <cilantro/data_containers.hpp>

namespace cilantro {
    // If positive, EigenDim is the embedding dimension. Set to Eigen::Dynamic for runtime setting.
    template <typename ScalarT, ptrdiff_t EigenDim = Eigen::Dynamic>
    class MultidimensionalScaling {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Embedding dimension set at compile time (EigenDim template parameter)
        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        MultidimensionalScaling(const Eigen::Ref<const Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>> &distances,
                                bool distances_are_squared = false)
        {
            compute_(distances, EigenDim, false, distances_are_squared);
        }

        // Embedding dimension set at runtime (EigenDim == Eigen::Dynamic)
        // If estimate_dim == true, chooses embedding dimension in [1, max_dim] based on eigenvalue distribution.
        // Otherwise, returns embedding of dimension max_dim.
        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim == Eigen::Dynamic>::type>
        MultidimensionalScaling(const Eigen::Ref<const Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>> &distances,
                                size_t max_dim,
                                bool estimate_dim = false,
                                bool distances_are_squared = false)
        {
            if (max_dim > 0 && max_dim < distances.rows()) {
                compute_(distances, max_dim, estimate_dim, distances_are_squared);
            } else {
                compute_(distances, 2, false, distances_are_squared);
            }
        }

        ~MultidimensionalScaling() {}

        inline const VectorSet<ScalarT,EigenDim>& getEmbeddedPoints() const { return embedded_points_; }

        inline const Vector<ScalarT,Eigen::Dynamic>& getComputedEigenValues() const { return eigenvalues_; }

    private:
        Vector<ScalarT,Eigen::Dynamic> eigenvalues_;
        VectorSet<ScalarT,EigenDim> embedded_points_;

        inline void compute_(const Eigen::Ref<const Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>> &distances,
                             size_t max_dim, bool estimate_dim, bool distances_are_squared)
        {
            size_t dim = max_dim;
            const size_t num_eigenvalues = (estimate_dim) ? std::min(max_dim+1, (size_t)(distances.rows()-1)) : max_dim;

            const ScalarT conv_tol = (std::is_same<ScalarT,float>::value) ? 1e-7 : 1e-10;
            size_t n_conv = 0;
            size_t max_iter = 1000;

            Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> J = Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>::Identity(distances.rows(),distances.cols()) - Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>::Constant(distances.rows(),distances.cols(),(ScalarT)(1.0)/distances.rows());
            Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> B;
            if (distances_are_squared) {
                B.noalias() = -(ScalarT)(0.5)*J*distances*J;
            } else {
                B.noalias() = -(ScalarT)(0.5)*J*distances.array().square().matrix()*J;
            }

            Spectra::DenseSymMatProd<ScalarT> op(B);
            Spectra::SymEigsSolver<ScalarT, Spectra::LARGEST_MAGN, Spectra::DenseSymMatProd<ScalarT>> eig(&op, num_eigenvalues, std::min(2*num_eigenvalues, (size_t)distances.rows()));
            eig.init();
            do {
                n_conv = eig.compute(max_iter, conv_tol, Spectra::LARGEST_MAGN);
                max_iter *= 2;
            } while (n_conv != num_eigenvalues);

            eigenvalues_ = eig.eigenvalues();
            for (size_t i = 0; i < eigenvalues_.rows(); i++) {
                if (eigenvalues_[i] < (ScalarT)0.0) eigenvalues_[i] = (ScalarT)0.0;
            }

            if (estimate_dim) {
                dim = estimate_embedding_dimension_(eigenvalues_, max_dim);
            }

            embedded_points_ = eigenvalues_.head(dim).cwiseSqrt().asDiagonal() * eig.eigenvectors(dim).transpose();
//            embedded_points_ = (eig.eigenvectors(dim)*eigenvalues_.head(dim).cwiseSqrt().asDiagonal()).transpose();
        }

        size_t estimate_embedding_dimension_(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,1>> &eigenvalues, size_t max_dim) {
            ScalarT min_val = std::numeric_limits<ScalarT>::infinity();
            ScalarT max_val = -std::numeric_limits<ScalarT>::infinity();
            ScalarT max_diff = -std::numeric_limits<ScalarT>::infinity();
            size_t max_ind = 0;
            for (size_t i = 0; i < eigenvalues.rows() - 1; i++) {
                ScalarT diff = eigenvalues[i] - eigenvalues[i+1];
                if (diff > max_diff) {
                    max_diff = diff;
                    max_ind = i;
                }
                if (eigenvalues[i] < min_val) min_val = eigenvalues[i];
                if (eigenvalues[i] > max_val) max_val = eigenvalues[i];
            }
            if (eigenvalues[eigenvalues.rows()-1] < min_val) min_val = eigenvalues[eigenvalues.rows()-1];
            if (eigenvalues[eigenvalues.rows()-1] > max_val) max_val = eigenvalues[eigenvalues.rows()-1];

            if (max_val - min_val < std::numeric_limits<ScalarT>::epsilon()) return max_dim;
            return max_ind + 1;
        }
    };

    typedef MultidimensionalScaling<float,2> MultidimensionalScaling2f;
    typedef MultidimensionalScaling<double,2> MultidimensionalScaling2d;
    typedef MultidimensionalScaling<float,3> MultidimensionalScaling3f;
    typedef MultidimensionalScaling<double,3> MultidimensionalScaling3d;
    typedef MultidimensionalScaling<float,Eigen::Dynamic> MultidimensionalScalingXf;
    typedef MultidimensionalScaling<double,Eigen::Dynamic> MultidimensionalScalingXd;
}
