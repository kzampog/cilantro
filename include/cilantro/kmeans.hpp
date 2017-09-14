#pragma once

#include <random>
#include <Eigen/Dense>

template <typename ScalarT, ptrdiff_t EigenDim>
class KMeans {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KMeans(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> > &points)
            : data_map_(points.data(), EigenDim, points.cols()),
              iteration_count_(0)
    {}

    KMeans(const std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > &points)
            : data_map_((const ScalarT *)points.data(), EigenDim, points.size()),
              iteration_count_(0)
    {}

    KMeans& cluster(const std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > &centroids, size_t max_iter = 100) {
        cluster_centroids_ = centroids;
        cluster_(max_iter);
        return *this;
    }

    KMeans& cluster(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> > &centroids, size_t max_iter = 100) {
        cluster_centroids_.resize(centroids.cols());
        Eigen::Map<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> >((ScalarT *)cluster_centroids_.data(), EigenDim, cluster_centroids_.size()) = centroids;
        cluster_(max_iter);
        return *this;
    }

    KMeans& cluster(size_t num_clusters, size_t max_iter = 100) {
        cluster_centroids_.resize((num_clusters > data_map_.cols()) ? data_map_.cols() : num_clusters);

        std::vector<size_t> range(data_map_.cols());
        for (size_t i = 0; i < range.size(); i++) range[i] = i;

        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<size_t> dist;
        for (size_t i = 0; i < cluster_centroids_.size(); i++) {
            size_t prev_size = range.size();
            size_t rand_ind = dist(rng) % prev_size;
            cluster_centroids_[i] = data_map_.col(range[rand_ind]);
            std::swap(range[rand_ind], range[prev_size-1]);
            range.resize(prev_size-1);
        }

        cluster_(max_iter);
        return *this;
    }

    inline const std::vector<Eigen::Matrix<ScalarT,EigenDim,1> >& getClusterCentroids() const { return cluster_centroids_; }
    inline Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> > getClusterCentroidsMatrixMap() const { return Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> >((ScalarT *)cluster_centroids_.data(), EigenDim, cluster_centroids_.size()); }
    inline const std::vector<std::vector<size_t> >& getClusterPointIndices() const { return cluster_point_indices_; }
    inline const std::vector<size_t>& getClusterIndexMap() const { return cluster_index_map_; }
    inline size_t getNumberOfClusters() const { return cluster_centroids_.size(); }
    inline size_t getNumberOfNonEmptyClusters() const {
        size_t nz_count = 0;
        for (size_t i = 0; i < cluster_point_indices_.size(); i++) {
            if (!cluster_point_indices_[i].empty()) nz_count++;
        }
        return nz_count;
    }
    inline size_t getPerformedIterationsCount() const { return iteration_count_; }

private:
    Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> > data_map_;

    std::vector<Eigen::Matrix<ScalarT,EigenDim,1> > cluster_centroids_;
    std::vector<std::vector<size_t> > cluster_point_indices_;
    std::vector<size_t> cluster_index_map_;

    size_t iteration_count_;

    void cluster_(size_t max_iter) {
        size_t num_clusters = cluster_centroids_.size();
        size_t num_points = data_map_.cols();

        cluster_index_map_.resize(num_points);

        iteration_count_ = 0;
        size_t min_ind;
        ScalarT min_dist, dist;
        while (iteration_count_ < max_iter) {
            bool assignments_unchanged = true;

            // Update assignments
            for (size_t i = 0; i < num_points; i++) {
                min_dist = std::numeric_limits<ScalarT>::infinity();
                for (size_t j = 0; j < num_clusters; j++) {
                    dist = (cluster_centroids_[j] - data_map_.col(i)).squaredNorm();
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_ind = j;
                    }
                }
                if (cluster_index_map_[i] != min_ind) assignments_unchanged = false;
                cluster_index_map_[i] = min_ind;
            }

            if (assignments_unchanged) break;

            // Update centroids
            Eigen::Map<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> >((ScalarT *)cluster_centroids_.data(), EigenDim, num_clusters).setZero();
            std::vector<size_t> point_count(num_clusters, 0);
            for (size_t i = 0; i < num_points; i++) {
                cluster_centroids_[cluster_index_map_[i]] += data_map_.col(i);
                point_count[cluster_index_map_[i]]++;
            }
            for (size_t i = 0; i < num_clusters; i++) {
                cluster_centroids_[i] /= point_count[i];
            }

            iteration_count_++;
        }

        cluster_point_indices_.resize(num_clusters);
        for (size_t i = 0; i < num_points; i++) {
            cluster_point_indices_[cluster_index_map_[i]].emplace_back(i);
        }

    }

//    static inline bool vector_size_comparator_(const std::vector<size_t> &a, const std::vector<size_t> &b) { return a.size() > b.size(); }
};

typedef KMeans<float,3> KMeans3D;
