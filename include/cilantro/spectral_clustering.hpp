#pragma once

#include <memory>
#include <cilantro/kmeans.hpp>

namespace cilantro {

    enum struct GraphLaplacianType {UNNORMALIZED, NORMALIZED_SYMMETRIC, NORMALIZED_RANDOM_WALK};

    // EigenDim is the embedding dimension (and also the number of clusters); set to Eigen::Dynamic for runtime setting
    template <typename ScalarT, ptrdiff_t EigenDim = Eigen::Dynamic>
    class SpectralClustering {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Number of clusters (embedding dimension) set at compile time
        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        SpectralClustering(const Eigen::Ref<const Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>> &affinities,
                           const GraphLaplacianType &laplacian_type = GraphLaplacianType::NORMALIZED_RANDOM_WALK,
                           size_t kmeans_max_iter = 100,
                           ScalarT kmeans_conv_tol = std::numeric_limits<ScalarT>::epsilon(),
                           bool kmeans_use_kd_tree = false)
        {
            compute_dense_(affinities, EigenDim, laplacian_type, kmeans_max_iter, kmeans_conv_tol, kmeans_use_kd_tree);
        }

        // Number of clusters (embedding dimension) set at runtime
        // Figures out number of clusters based of eigenvalue distribution if num_clusters == 0
        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim == Eigen::Dynamic>::type>
        SpectralClustering(const Eigen::Ref<const Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>> &affinities,
                           size_t num_clusters,
                           const GraphLaplacianType &laplacian_type = GraphLaplacianType::NORMALIZED_RANDOM_WALK,
                           size_t kmeans_max_iter = 100,
                           ScalarT kmeans_conv_tol = std::numeric_limits<ScalarT>::epsilon(),
                           bool kmeans_use_kd_tree = false)
        {
            compute_dense_(affinities, num_clusters, laplacian_type, kmeans_max_iter, kmeans_conv_tol, kmeans_use_kd_tree);
        }

        ~SpectralClustering() {}

        inline const VectorSet<ScalarT,EigenDim>& getEmbeddedPoints() const { return embedded_points_; }

        inline const std::vector<std::vector<size_t>>& getClusterPointIndices() const { return clusterer_->getClusterPointIndices(); }

        inline const std::vector<size_t>& getClusterIndexMap() const { return clusterer_->getClusterIndexMap(); }

        inline size_t getNumberOfClusters() const { return clusterer_->getNumberOfCluster(); }

        inline const KMeans<ScalarT,EigenDim>& getClusterer() const { return *clusterer_; }

    private:
        VectorSet<ScalarT,EigenDim> embedded_points_;
        std::shared_ptr<KMeans<ScalarT,EigenDim>> clusterer_;

        void compute_dense_(const Eigen::Ref<const Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>> &affinities,
                            size_t num_clusters_in, const GraphLaplacianType &laplacian_type,
                            size_t kmeans_max_iter, ScalarT kmeans_conv_tol, bool kmeans_use_kd_tree)
        {
            size_t num_clusters = num_clusters_in;

            switch (laplacian_type) {
                case GraphLaplacianType::UNNORMALIZED: {
                    Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> L = affinities.rowwise().sum().asDiagonal() - affinities;

                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>> eig(L);
                    if (num_clusters == 0) {
                        // TODO :)
                    }
                    embedded_points_ = eig.eigenvectors().leftCols(num_clusters).transpose();

                    clusterer_.reset(new KMeans<ScalarT,EigenDim>(embedded_points_));
                    clusterer_->cluster(num_clusters, kmeans_max_iter, kmeans_conv_tol, kmeans_use_kd_tree);

                    break;
                }
                case GraphLaplacianType::NORMALIZED_SYMMETRIC: {
                    Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> Dtm12 = affinities.rowwise().sum().rsqrt().asDiagonal();
                    Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> L = Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>::Identity(affinities.rows(),affinities.cols()) - Dtm12*affinities*Dtm12;

                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>> eig(L);
                    if (num_clusters == 0) {
                        // TODO :)
                    }
                    embedded_points_ = eig.eigenvectors().leftCols(num_clusters).transpose();

                    for (size_t i = 0; i < embedded_points_.cols(); i++) {
                        ScalarT scale = 1.0/embedded_points_.col(i).norm();
                        embedded_points_.col(i) *= scale;
                    }

                    clusterer_.reset(new KMeans<ScalarT,EigenDim>(embedded_points_));
                    clusterer_->cluster(num_clusters, kmeans_max_iter, kmeans_conv_tol, kmeans_use_kd_tree);

                    break;
                }
                case GraphLaplacianType::NORMALIZED_RANDOM_WALK: {
                    Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> D = affinities.rowwise().sum().asDiagonal();
                    Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic> L = D - affinities;

                    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<ScalarT,Eigen::Dynamic,Eigen::Dynamic>> geig(L,D);
                    if (num_clusters == 0) {
                        // TODO :)
                    }
                    embedded_points_ = geig.eigenvectors().leftCols(num_clusters).transpose();

                    clusterer_.reset(new KMeans<ScalarT,EigenDim>(embedded_points_));
                    clusterer_->cluster(num_clusters, kmeans_max_iter, kmeans_conv_tol, kmeans_use_kd_tree);

                    break;
                }
            }
        }

//        size_t estimate_optimal_number_of_clusters() {
//
//        }

    };
}
