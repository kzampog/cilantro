#pragma once

#include <memory>
#include <cilantro/3rd_party/spectra/SymEigsSolver.h>
#include <cilantro/3rd_party/spectra/SymGEigsSolver.h>
#include <cilantro/kmeans.hpp>

namespace cilantro {
    template <typename ScalarT>
    class SpectraDiagonalInverseBop {
    private:
        const int dim_;
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> diag_vals_;
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> diag_vals_inv_;
    public:
        SpectraDiagonalInverseBop(const Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>& D)
                : dim_(D.rows()),
                  diag_vals_(D.diagonal()),
                  diag_vals_inv_(diag_vals_.cwiseInverse())
        {}

        SpectraDiagonalInverseBop(const Eigen::SparseMatrix<ScalarT>& D)
                : dim_(D.rows()),
                  diag_vals_(D.diagonal()),
                  diag_vals_inv_(diag_vals_.cwiseInverse())
        {}

        inline int rows() const { return dim_; }
        inline int cols() const { return dim_; }

        // y_out = inv(B) * x_in
        void solve(const ScalarT* x_in, ScalarT* y_out) const {
            Eigen::Map<const Eigen::Matrix<ScalarT,Eigen::Dynamic,1>> x(x_in, dim_);
            Eigen::Map<Eigen::Matrix<ScalarT,Eigen::Dynamic,1>> y(y_out, dim_);
            y.noalias() = diag_vals_inv_.cwiseProduct(x);
        }

        // y_out = B * x_in
        void mat_prod(const ScalarT* x_in, ScalarT* y_out) const {
            Eigen::Map<const Eigen::Matrix<ScalarT,Eigen::Dynamic,1>> x(x_in, dim_);
            Eigen::Map<Eigen::Matrix<ScalarT,Eigen::Dynamic,1>> y(y_out, dim_);
            y.noalias() = diag_vals_.cwiseProduct(x);
        }
    };

    enum struct GraphLaplacianType {UNNORMALIZED, NORMALIZED_SYMMETRIC, NORMALIZED_RANDOM_WALK};

    // If positive, EigenDim is the embedding dimension (and also the number of clusters).
    // Set to Eigen::Dynamic for runtime setting.
    template <typename ScalarT, ptrdiff_t EigenDim = Eigen::Dynamic>
    class SpectralClustering {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Dense input
        // Number of clusters (embedding dimension) set at compile time (EigenDim template parameter)
        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        SpectralClustering(const Eigen::Ref<const Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>> &affinities,
                           const GraphLaplacianType &laplacian_type = GraphLaplacianType::NORMALIZED_RANDOM_WALK,
                           size_t kmeans_max_iter = 100,
                           ScalarT kmeans_conv_tol = std::numeric_limits<ScalarT>::epsilon(),
                           bool kmeans_use_kd_tree = false)
        {
            compute_dense_(affinities, EigenDim, false, laplacian_type, kmeans_max_iter, kmeans_conv_tol, kmeans_use_kd_tree);
        }

        // Dense input
        // Number of clusters (embedding dimension) set at runtime (EigenDim == Eigen::Dynamic)
        // If estimate_num_clusters == true, chooses number of clusters in [1, max_num_clusters] based on eigenvalue
        // distribution. Otherwise, returns max_num_clusters clusters.
        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim == Eigen::Dynamic>::type>
        SpectralClustering(const Eigen::Ref<const Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>> &affinities,
                           size_t max_num_clusters,
                           bool estimate_num_clusters = false,
                           const GraphLaplacianType &laplacian_type = GraphLaplacianType::NORMALIZED_RANDOM_WALK,
                           size_t kmeans_max_iter = 100,
                           ScalarT kmeans_conv_tol = std::numeric_limits<ScalarT>::epsilon(),
                           bool kmeans_use_kd_tree = false)
        {
            if (max_num_clusters > 0 && max_num_clusters < affinities.rows()) {
                compute_dense_(affinities, max_num_clusters, estimate_num_clusters, laplacian_type, kmeans_max_iter, kmeans_conv_tol, kmeans_use_kd_tree);
            } else {
                compute_dense_(affinities, 2, false, laplacian_type, kmeans_max_iter, kmeans_conv_tol, kmeans_use_kd_tree);
            }
        }

        // Sparse input
        // Number of clusters (embedding dimension) set at compile time (EigenDim template parameter)
        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        SpectralClustering(const Eigen::SparseMatrix<ScalarT> &affinities,
                           const GraphLaplacianType &laplacian_type = GraphLaplacianType::NORMALIZED_RANDOM_WALK,
                           size_t kmeans_max_iter = 100,
                           ScalarT kmeans_conv_tol = std::numeric_limits<ScalarT>::epsilon(),
                           bool kmeans_use_kd_tree = false)
        {
            compute_sparse_(affinities, EigenDim, false, laplacian_type, kmeans_max_iter, kmeans_conv_tol, kmeans_use_kd_tree);
        }

        // Sparse input
        // Number of clusters (embedding dimension) set at runtime (EigenDim == Eigen::Dynamic)
        // If estimate_num_clusters == true, chooses number of clusters in [1, max_num_clusters] based on eigenvalue
        // distribution. Otherwise, returns max_num_clusters clusters.
        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim == Eigen::Dynamic>::type>
        SpectralClustering(const Eigen::SparseMatrix<ScalarT> &affinities,
                           size_t max_num_clusters,
                           bool estimate_num_clusters = false,
                           const GraphLaplacianType &laplacian_type = GraphLaplacianType::NORMALIZED_RANDOM_WALK,
                           size_t kmeans_max_iter = 100,
                           ScalarT kmeans_conv_tol = std::numeric_limits<ScalarT>::epsilon(),
                           bool kmeans_use_kd_tree = false)
        {
            if (max_num_clusters > 0 && max_num_clusters < affinities.rows()) {
                compute_sparse_(affinities, max_num_clusters, estimate_num_clusters, laplacian_type, kmeans_max_iter, kmeans_conv_tol, kmeans_use_kd_tree);
            } else {
                compute_sparse_(affinities, 2, false, laplacian_type, kmeans_max_iter, kmeans_conv_tol, kmeans_use_kd_tree);
            }
        }

        ~SpectralClustering() {}

        inline const VectorSet<ScalarT,EigenDim>& getEmbeddedPoints() const { return embedded_points_; }

        inline const Vector<ScalarT,Eigen::Dynamic>& getComputedEigenValues() const { return eigenvalues_; }

        inline const std::vector<std::vector<size_t>>& getClusterPointIndices() const { return clusterer_->getClusterPointIndices(); }

        inline const std::vector<size_t>& getClusterIndexMap() const { return clusterer_->getClusterIndexMap(); }

        inline size_t getNumberOfClusters() const { return embedded_points_.rows(); }

        inline const KMeans<ScalarT,EigenDim>& getClusterer() const { return *clusterer_; }

    private:
        Vector<ScalarT,Eigen::Dynamic> eigenvalues_;
        VectorSet<ScalarT,EigenDim> embedded_points_;
        std::shared_ptr<KMeans<ScalarT,EigenDim>> clusterer_;

        void compute_dense_(const Eigen::Ref<const Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>> &affinities,
                            size_t max_num_clusters,
                            bool estimate_num_clusters,
                            const GraphLaplacianType &laplacian_type,
                            size_t kmeans_max_iter,
                            ScalarT kmeans_conv_tol,
                            bool kmeans_use_kd_tree)
        {
            size_t num_clusters = max_num_clusters;
            const size_t num_eigenvalues = (estimate_num_clusters) ? std::min(max_num_clusters+1, (size_t)(affinities.rows()-1)) : max_num_clusters;

            ScalarT conv_tol = (std::is_same<ScalarT,float>::value) ? 1e-7 : 1e-10;
            size_t n_conv = 0;
            size_t max_iter = 1000;

            switch (laplacian_type) {
                case GraphLaplacianType::UNNORMALIZED: {
                    Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> L = affinities.colwise().sum().asDiagonal();
                    L -=  affinities;

                    Spectra::DenseSymMatProd<ScalarT> op(L);
                    Spectra::SymEigsSolver<ScalarT, Spectra::SMALLEST_MAGN, Spectra::DenseSymMatProd<ScalarT>> eig(&op, num_eigenvalues, std::min(2*num_eigenvalues, (size_t)affinities.rows()));
                    eig.init();
                    do {
                        n_conv = eig.compute(max_iter, conv_tol, Spectra::SMALLEST_MAGN);
                        max_iter *= 2;
                    } while (n_conv != num_eigenvalues);

                    eigenvalues_ = eig.eigenvalues();
                    for (size_t i = 0; i < eigenvalues_.rows(); i++) {
                        if (eigenvalues_[i] < (ScalarT)0.0) eigenvalues_[i] = (ScalarT)0.0;
                    }

                    if (estimate_num_clusters) {
                        num_clusters = estimate_number_of_clusters_(eigenvalues_, max_num_clusters);
                    }

                    embedded_points_ = eig.eigenvectors(num_clusters).transpose();

                    break;
                }
                case GraphLaplacianType::NORMALIZED_SYMMETRIC: {
                    Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> Dtm12 = affinities.colwise().sum().array().rsqrt().matrix().asDiagonal();
                    Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> L = Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>::Identity(affinities.rows(),affinities.cols()) - Dtm12*affinities*Dtm12;

                    Spectra::DenseSymMatProd<ScalarT> op(L);
                    Spectra::SymEigsSolver<ScalarT, Spectra::SMALLEST_MAGN, Spectra::DenseSymMatProd<ScalarT>> eig(&op, num_eigenvalues, std::min(2*num_eigenvalues, (size_t)affinities.rows()));
                    eig.init();
                    do {
                        n_conv = eig.compute(max_iter, conv_tol, Spectra::SMALLEST_MAGN);
                        max_iter *= 2;
                    } while (n_conv != num_eigenvalues);

                    eigenvalues_ = eig.eigenvalues();
                    for (size_t i = 0; i < eigenvalues_.rows(); i++) {
                        if (eigenvalues_[i] < (ScalarT)0.0) eigenvalues_[i] = (ScalarT)0.0;
                    }

                    if (estimate_num_clusters) {
                        num_clusters = estimate_number_of_clusters_(eigenvalues_, max_num_clusters);
                    }

                    embedded_points_ = eig.eigenvectors(num_clusters).transpose();

                    for (size_t i = 0; i < embedded_points_.cols(); i++) {
                        ScalarT scale = (ScalarT)(1.0)/embedded_points_.col(i).norm();
                        if (std::isfinite(scale)) embedded_points_.col(i) *= scale;
                    }

                    break;
                }
                case GraphLaplacianType::NORMALIZED_RANDOM_WALK: {
                    Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> D = affinities.colwise().sum().asDiagonal();
                    Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> L = D - affinities;

                    Spectra::DenseSymMatProd<ScalarT> op(L);
                    SpectraDiagonalInverseBop<ScalarT> Bop(D);
                    Spectra::SymGEigsSolver<ScalarT, Spectra::SMALLEST_MAGN, Spectra::DenseSymMatProd<ScalarT>, SpectraDiagonalInverseBop<ScalarT>, Spectra::GEIGS_REGULAR_INVERSE> eig(&op, &Bop, num_eigenvalues, std::min(2*num_eigenvalues, (size_t)affinities.rows()));

                    eig.init();
                    do {
                        n_conv = eig.compute(max_iter, conv_tol, Spectra::SMALLEST_MAGN);
                        max_iter *= 2;
                    } while (n_conv != num_eigenvalues);

                    eigenvalues_ = eig.eigenvalues();
                    for (size_t i = 0; i < eigenvalues_.rows(); i++) {
                        if (eigenvalues_[i] < (ScalarT)0.0) eigenvalues_[i] = (ScalarT)0.0;
                    }

                    if (estimate_num_clusters) {
                        num_clusters = estimate_number_of_clusters_(eigenvalues_, max_num_clusters);
                    }

                    embedded_points_ = eig.eigenvectors(num_clusters).transpose();

                    break;
                }
            }

            clusterer_.reset(new KMeans<ScalarT,EigenDim>(embedded_points_));
            clusterer_->cluster(num_clusters, kmeans_max_iter, kmeans_conv_tol, kmeans_use_kd_tree);
        }

        void compute_sparse_(const Eigen::SparseMatrix<ScalarT> &affinities,
                             size_t max_num_clusters,
                             bool estimate_num_clusters,
                             const GraphLaplacianType &laplacian_type,
                             size_t kmeans_max_iter,
                             ScalarT kmeans_conv_tol,
                             bool kmeans_use_kd_tree)
        {
            size_t num_clusters = max_num_clusters;
            const size_t num_eigenvalues = (estimate_num_clusters) ? std::min(max_num_clusters+1, (size_t)(affinities.rows()-1)) : max_num_clusters;

            ScalarT conv_tol = (std::is_same<ScalarT,float>::value) ? 1e-7 : 1e-10;
            size_t n_conv = 0;
            size_t max_iter = 1000;

            switch (laplacian_type) {
                case GraphLaplacianType::UNNORMALIZED: {
                    Eigen::SparseMatrix<ScalarT> D(affinities.rows(),affinities.cols());
                    D.reserve(Eigen::VectorXi::Ones(affinities.rows()));
                    for (size_t i = 0; i < affinities.cols(); i++) {
                        D.insert(i,i) = affinities.col(i).sum();
                    }
                    Eigen::SparseMatrix<ScalarT> L = D - affinities;

                    Spectra::SparseSymMatProd<ScalarT> op(L);
                    Spectra::SymEigsSolver<ScalarT, Spectra::SMALLEST_MAGN, Spectra::SparseSymMatProd<ScalarT>> eig(&op, num_eigenvalues, std::min(2*num_eigenvalues, (size_t)affinities.rows()));
                    eig.init();
                    do {
                        n_conv = eig.compute(max_iter, conv_tol, Spectra::SMALLEST_MAGN);
                        max_iter *= 2;
                    } while (n_conv != num_eigenvalues);

                    eigenvalues_ = eig.eigenvalues();
                    for (size_t i = 0; i < eigenvalues_.rows(); i++) {
                        if (eigenvalues_[i] < (ScalarT)0.0) eigenvalues_[i] = (ScalarT)0.0;
                    }

                    if (estimate_num_clusters) {
                        num_clusters = estimate_number_of_clusters_(eigenvalues_, max_num_clusters);
                    }

                    embedded_points_ = eig.eigenvectors(num_clusters).transpose();

                    break;
                }
                case GraphLaplacianType::NORMALIZED_SYMMETRIC: {
                    Eigen::SparseMatrix<ScalarT> Dtm12(affinities.rows(),affinities.cols());
                    Dtm12.reserve(Eigen::VectorXi::Ones(affinities.rows()));
                    for (size_t i = 0; i < affinities.cols(); i++) {
                        Dtm12.insert(i,i) = (ScalarT)(1.0)/std::sqrt(affinities.col(i).sum());
                    }
                    Eigen::SparseMatrix<ScalarT> L(affinities.rows(),affinities.cols());
                    L.setIdentity();
                    L -= Dtm12*affinities*Dtm12;

                    Spectra::SparseSymMatProd<ScalarT> op(L);
                    Spectra::SymEigsSolver<ScalarT, Spectra::SMALLEST_MAGN, Spectra::SparseSymMatProd<ScalarT>> eig(&op, num_eigenvalues, std::min(2*num_eigenvalues, (size_t)affinities.rows()));
                    eig.init();
                    do {
                        n_conv = eig.compute(max_iter, conv_tol, Spectra::SMALLEST_MAGN);
                        max_iter *= 2;
                    } while (n_conv != num_eigenvalues);

                    eigenvalues_ = eig.eigenvalues();
                    for (size_t i = 0; i < eigenvalues_.rows(); i++) {
                        if (eigenvalues_[i] < (ScalarT)0.0) eigenvalues_[i] = (ScalarT)0.0;
                    }

                    if (estimate_num_clusters) {
                        num_clusters = estimate_number_of_clusters_(eigenvalues_, max_num_clusters);
                    }

                    embedded_points_ = eig.eigenvectors(num_clusters).transpose();

                    for (size_t i = 0; i < embedded_points_.cols(); i++) {
                        ScalarT scale = (ScalarT)(1.0)/embedded_points_.col(i).norm();
                        if (std::isfinite(scale)) embedded_points_.col(i) *= scale;
                    }

                    break;
                }
                case GraphLaplacianType::NORMALIZED_RANDOM_WALK: {
                    Eigen::SparseMatrix<ScalarT> D(affinities.rows(),affinities.cols());
                    D.reserve(Eigen::VectorXi::Ones(affinities.rows()));
                    for (size_t i = 0; i < affinities.cols(); i++) {
                        D.insert(i,i) = affinities.col(i).sum();
                    }
                    Eigen::SparseMatrix<ScalarT> L = D - affinities;

                    Spectra::SparseSymMatProd<ScalarT> op(L);
                    SpectraDiagonalInverseBop<ScalarT> Bop(D);
                    Spectra::SymGEigsSolver<ScalarT, Spectra::SMALLEST_MAGN, Spectra::SparseSymMatProd<ScalarT>, SpectraDiagonalInverseBop<ScalarT>, Spectra::GEIGS_REGULAR_INVERSE> eig(&op, &Bop, num_eigenvalues, std::min(2*num_eigenvalues, (size_t)affinities.rows()));

                    eig.init();
                    do {
                        n_conv = eig.compute(max_iter, conv_tol, Spectra::SMALLEST_MAGN);
                        max_iter *= 2;
                    } while (n_conv != num_eigenvalues);

                    eigenvalues_ = eig.eigenvalues();
                    for (size_t i = 0; i < eigenvalues_.rows(); i++) {
                        if (eigenvalues_[i] < (ScalarT)0.0) eigenvalues_[i] = (ScalarT)0.0;
                    }

                    if (estimate_num_clusters) {
                        num_clusters = estimate_number_of_clusters_(eigenvalues_, max_num_clusters);
                    }

                    embedded_points_ = eig.eigenvectors(num_clusters).transpose();

                    break;
                }
            }

            clusterer_.reset(new KMeans<ScalarT,EigenDim>(embedded_points_));
            clusterer_->cluster(num_clusters, kmeans_max_iter, kmeans_conv_tol, kmeans_use_kd_tree);
        }

        size_t estimate_number_of_clusters_(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,1>> &eigenvalues,
                                            size_t max_num_clusters)
        {
            ScalarT min_val = std::numeric_limits<ScalarT>::infinity();
            ScalarT max_val = -std::numeric_limits<ScalarT>::infinity();
            ScalarT max_diff = eigenvalues[0];
            size_t max_ind = 0;
            for (size_t i = 0; i < eigenvalues.rows() - 1; i++) {
                ScalarT diff = eigenvalues[i+1] - eigenvalues[i];
                if (diff > max_diff) {
                    max_diff = diff;
                    max_ind = i;
                }
                if (eigenvalues[i] < min_val) min_val = eigenvalues[i];
                if (eigenvalues[i] > max_val) max_val = eigenvalues[i];
            }
            if (eigenvalues[eigenvalues.rows()-1] < min_val) min_val = eigenvalues[eigenvalues.rows()-1];
            if (eigenvalues[eigenvalues.rows()-1] > max_val) max_val = eigenvalues[eigenvalues.rows()-1];

            if (max_val - min_val < std::numeric_limits<ScalarT>::epsilon()) return max_num_clusters;
            return max_ind + 1;
        }
    };
}
