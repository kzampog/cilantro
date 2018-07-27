#pragma once

#include <random>
#include <cilantro/kd_tree.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    class KMeans {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum { Dimension = EigenDim };

        KMeans(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &data)
                : data_map_(data),
                  iteration_count_(0)
        {}

        ~KMeans() {}

        KMeans& cluster(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &centroids,
                        size_t max_iter = 100,
                        ScalarT tol = std::numeric_limits<ScalarT>::epsilon(),
                        bool use_kd_tree = false)
        {
            cluster_centroids_ = centroids;
            cluster_(max_iter, tol, use_kd_tree);
            return *this;
        }

        KMeans& cluster(size_t num_clusters,
                        size_t max_iter = 100,
                        ScalarT tol = std::numeric_limits<ScalarT>::epsilon(),
                        bool use_kd_tree = false)
        {
            cluster_centroids_.resize(data_map_.rows(), std::max((size_t)1, std::min(num_clusters, (size_t)data_map_.cols())));

            std::vector<size_t> range(data_map_.cols());
            for (size_t i = 0; i < range.size(); i++) range[i] = i;

            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_int_distribution<size_t> dist;
            size_t prev_size = range.size();
            for (size_t i = 0; i < cluster_centroids_.cols(); i++) {
                size_t rand_ind = dist(rng) % prev_size;
                cluster_centroids_.col(i) = data_map_.col(range[rand_ind]);
                prev_size--;
                std::swap(range[rand_ind], range[prev_size]);
            }

            cluster_(max_iter, tol, use_kd_tree);
            return *this;
        }

        inline const VectorSet<ScalarT,EigenDim>& getClusterCentroids() const { return cluster_centroids_; }

        inline const std::vector<std::vector<size_t>>& getClusterPointIndices() const { return cluster_point_indices_; }

        inline const std::vector<size_t>& getClusterIndexMap() const { return cluster_index_map_; }

        inline size_t getNumberOfClusters() const { return cluster_centroids_.cols(); }

        inline size_t getNumberOfPerformedIterations() const { return iteration_count_; }

    private:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> data_map_;

        VectorSet<ScalarT,EigenDim> cluster_centroids_;
        std::vector<std::vector<size_t>> cluster_point_indices_;
        std::vector<size_t> cluster_index_map_;

        size_t iteration_count_;

        inline void cluster_(size_t max_iter, ScalarT tol, bool use_kd_tree) {
            const size_t num_clusters = cluster_centroids_.cols();
            const size_t num_points = data_map_.cols();
            const ScalarT tol_sq = tol*tol;

            cluster_index_map_.resize(num_points);

            size_t extr_dist_ind;
            ScalarT extr_dist;

            ScalarT dist;

            VectorSet<ScalarT,EigenDim> centroids_old;

            KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim> data_adaptor(data_map_);
            DistAdaptor<KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim>> dist_adaptor(data_adaptor);

            iteration_count_ = 0;
            while (iteration_count_ < max_iter) {
                bool assignments_unchanged = true;

                // Update assignments
                if (use_kd_tree) {
                    Neighbor<ScalarT> nn;
                    KDTree<ScalarT,EigenDim,DistAdaptor> tree(cluster_centroids_);
#pragma omp parallel for shared (assignments_unchanged) private (nn)
                    for (size_t i = 0; i < num_points; i++) {
                        tree.nearestNeighborSearch(data_map_.col(i), nn);
                        if (cluster_index_map_[i] != nn.index) assignments_unchanged = false;
                        cluster_index_map_[i] = nn.index;
                    }
                } else {
#pragma omp parallel for shared (assignments_unchanged) private (extr_dist, extr_dist_ind, dist)
                    for (size_t i = 0; i < num_points; i++) {
                        extr_dist = std::numeric_limits<ScalarT>::infinity();
                        for (size_t j = 0; j < num_clusters; j++) {
                            // Resolved at compile time
                            if (std::is_same<DistAdaptor<KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim>>, KDTreeDistanceAdaptors::L2<KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim>>>::value ||
                                std::is_same<DistAdaptor<KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim>>, KDTreeDistanceAdaptors::L2Simple<KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim>>>::value)
                            {
                                dist = (cluster_centroids_.col(j) - data_map_.col(i)).squaredNorm();
                            } else {
                                dist = dist_adaptor.evalMetric(&(cluster_centroids_.col(j)[0]), i, data_map_.rows());
                            }
                            if (dist < extr_dist) {
                                extr_dist = dist;
                                extr_dist_ind = j;
                            }
                        }
                        if (cluster_index_map_[i] != extr_dist_ind) assignments_unchanged = false;
                        cluster_index_map_[i] = extr_dist_ind;
                    }
                }

                if (assignments_unchanged && iteration_count_ > 0) break;
                if (tol > (ScalarT)0.0) centroids_old = cluster_centroids_;

                // Update centroids
                cluster_centroids_.setZero();
                std::vector<size_t> point_count(num_clusters, 0);
                for (size_t i = 0; i < num_points; i++) {
                    cluster_centroids_.col(cluster_index_map_[i]) += data_map_.col(i);
                    point_count[cluster_index_map_[i]]++;
                }

                // Handle empty clusters
                for (size_t i = 0; i < num_clusters; i++) {
                    if (point_count[i] != 0) continue;

                    // Find largest cluster
                    size_t max_ind = 0;
                    for (size_t j = 1; j < num_clusters; j++) {
                        if (point_count[j] > point_count[max_ind]) max_ind = j;
                    }

                    // Find furthest point from (old) centroid of previously found cluster
                    Vector<ScalarT,EigenDim> old_centroid(cluster_centroids_.col(max_ind)*(ScalarT)(1.0)/point_count[max_ind]);
                    extr_dist = (ScalarT)(-1.0);
#pragma omp parallel for shared (extr_dist, extr_dist_ind) private (dist)
                    for (size_t j = 0; j < num_points; j++) {
                        if (cluster_index_map_[j] == max_ind) {
                            // Resolved at compile time
                            if (std::is_same<DistAdaptor<KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim>>, KDTreeDistanceAdaptors::L2<KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim>>>::value ||
                                std::is_same<DistAdaptor<KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim>>, KDTreeDistanceAdaptors::L2Simple<KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim>>>::value)
                            {
                                dist = (old_centroid - data_map_.col(j)).squaredNorm();
                            } else {
                                dist = dist_adaptor.evalMetric(&(old_centroid[0]), j, data_map_.rows());
                            }
#pragma omp critical
                            if (dist > extr_dist) {
                                extr_dist = dist;
                                extr_dist_ind = j;
                            }
                        }
                    }

                    // Move previously found point to current (empty) cluster
                    cluster_index_map_[extr_dist_ind] = i;
                    cluster_centroids_.col(max_ind) -= data_map_.col(extr_dist_ind);
                    point_count[max_ind]--;
                    point_count[i]++;
                }

                // Compute new centroids
                for (size_t i = 0; i < num_clusters; i++) {
                    cluster_centroids_.col(i) *= (ScalarT)(1.0)/point_count[i];
                }

                iteration_count_++;

                // Check for convergence of centroids
                if (tol > (ScalarT)0.0 && (cluster_centroids_ - centroids_old).colwise().squaredNorm().maxCoeff() < tol_sq) break;
            }

            cluster_point_indices_.resize(num_clusters);
            for (size_t i = 0; i < num_points; i++) {
                cluster_point_indices_[cluster_index_map_[i]].emplace_back(i);
            }
        }
    };

    typedef KMeans<float,2,KDTreeDistanceAdaptors::L2> KMeans2f;
    typedef KMeans<double,2,KDTreeDistanceAdaptors::L2> KMeans2d;
    typedef KMeans<float,3,KDTreeDistanceAdaptors::L2> KMeans3f;
    typedef KMeans<double,3,KDTreeDistanceAdaptors::L2> KMeans3d;
    typedef KMeans<float,Eigen::Dynamic,KDTreeDistanceAdaptors::L2> KMeansXf;
    typedef KMeans<double,Eigen::Dynamic,KDTreeDistanceAdaptors::L2> KMeansXd;
}
