#pragma once

#include <vector>
#include <Eigen/Dense>

namespace cilantro {
    template <typename ClusterIndexT, typename PointIndexT>
    std::vector<ClusterIndexT> getPointToClusterIndexMap(const std::vector<std::vector<PointIndexT>> &cluster_to_point,
                                                         size_t num_points)
    {
        // cluster_to_point.size() signifies unlabeled point
        std::vector<ClusterIndexT> point_to_segment(num_points, cluster_to_point.size());
        for (size_t i = 0; i < cluster_to_point.size(); i++) {
            for (size_t j = 0; j < cluster_to_point[i].size(); j++) {
                point_to_segment[cluster_to_point[i][j]] = static_cast<ClusterIndexT>(i);
            }
        }
        return point_to_segment;
    }

    // Cluster indices in point_to_cluster are in [0, num_clusters - 1];
    // a cluster index >= num_clusters signifies unlabeled point
    template <typename PointIndexT, typename ClusterIndexT>
    std::vector<std::vector<PointIndexT>> getClusterToPointIndicesMap(const std::vector<ClusterIndexT> &point_to_cluster,
                                                                      size_t num_clusters)
    {
        const ClusterIndexT first_invalid_cluster_idx = static_cast<ClusterIndexT>(num_clusters);
        std::vector<std::vector<PointIndexT>> cluster_to_point(num_clusters);
        for (size_t i = 0; i < point_to_cluster.size(); i++) {
            if (point_to_cluster[i] < first_invalid_cluster_idx) cluster_to_point[point_to_cluster[i]].emplace_back(i);
        }
        return cluster_to_point;
    }

    // Cluster indices in point_to_cluster are in [0, num_clusters - 1];
    // a cluster index >= num_clusters signifies unlabeled point
    template <typename PointIndexT, typename ClusterIndexT>
    std::vector<PointIndexT> getLabeledPointIndices(const std::vector<ClusterIndexT> &point_to_cluster,
                                                    size_t num_clusters)
    {
        const ClusterIndexT first_invalid_cluster_idx = static_cast<ClusterIndexT>(num_clusters);
        std::vector<PointIndexT> res;
        for (size_t i = 0; i < point_to_cluster.size(); i++) {
            if (point_to_cluster[i] < first_invalid_cluster_idx) res.emplace_back(i);
        }
        return res;
    }

    // Cluster indices in point_to_cluster are in [0, num_clusters - 1];
    // a cluster index >= num_clusters signifies unlabeled point
    template <typename PointIndexT, typename ClusterIndexT>
    std::vector<PointIndexT> getUnlabeledPointIndices(const std::vector<ClusterIndexT> &point_to_cluster,
                                                      size_t num_clusters)
    {
        const ClusterIndexT first_invalid_cluster_idx = static_cast<ClusterIndexT>(num_clusters);
        std::vector<PointIndexT> res;
        for (size_t i = 0; i < point_to_cluster.size(); i++) {
            if (point_to_cluster[i] >= first_invalid_cluster_idx) res.emplace_back(i);
        }
        return res;
    }

    // CRTP base class that holds clustering results and accessors
    template <typename Derived, typename PointIndexT, typename ClusterIndexT>
    class ClusteringBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef PointIndexT PointIndex;
        typedef ClusterIndexT ClusterIndex;

        typedef std::vector<std::vector<PointIndexT>> ClusterToPointIndicesMap;
        typedef std::vector<ClusterIndexT> PointToClusterIndexMap;

        inline const ClusterToPointIndicesMap& getClusterToPointIndicesMap() const { return cluster_to_point_indices_map_; }

        inline const PointToClusterIndexMap& getPointToClusterIndexMap() const { return point_to_cluster_index_map_; }

        inline size_t getNumberOfClusters() const { return cluster_to_point_indices_map_.size(); }

        inline size_t getNumberOfPoints() const { return point_to_cluster_index_map_.size(); }

        template <typename IndexT = PointIndexT>
        inline std::vector<IndexT> getLabeledPointIndices() const {
            return cilantro::getLabeledPointIndices<IndexT>(point_to_cluster_index_map_, cluster_to_point_indices_map_.size());
        }

        template <typename IndexT = PointIndexT>
        inline std::vector<IndexT> getUnlabeledPointIndices() const {
            return cilantro::getUnlabeledPointIndices<IndexT>(point_to_cluster_index_map_, cluster_to_point_indices_map_.size());
        }

    protected:
        ClusterToPointIndicesMap cluster_to_point_indices_map_;
        PointToClusterIndexMap point_to_cluster_index_map_;
    };
}
