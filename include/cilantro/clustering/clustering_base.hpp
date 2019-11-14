#pragma once

#include <vector>
#include <Eigen/Dense>

namespace cilantro {
    static std::vector<size_t> getPointToClusterIndexMap(const std::vector<std::vector<size_t>> &cluster_to_point,
                                                         size_t num_points)
    {
        std::vector<size_t> point_to_segment(num_points, cluster_to_point.size());
        for (size_t i = 0; i < cluster_to_point.size(); i++) {
            for (size_t j = 0; j < cluster_to_point[i].size(); j++) {
                point_to_segment[cluster_to_point[i][j]] = i;
            }
        }
        return point_to_segment;
    }

    static std::vector<std::vector<size_t>> getClusterToPointIndicesMap(const std::vector<size_t> &point_to_cluster,
                                                                        size_t num_clusters)
    {
        std::vector<std::vector<size_t>> segment_to_point(num_clusters);
        for (size_t i = 0; i < point_to_cluster.size(); i++) {
            segment_to_point[point_to_cluster[i]].emplace_back(i);
        }
        return segment_to_point;
    }

    static std::vector<size_t> getUnlabeledPointIndices(const std::vector<std::vector<size_t>> &segment_to_point_map,
                                                        const std::vector<size_t> &point_to_segment_map)
    {
        const size_t no_label = segment_to_point_map.size();
        std::vector<size_t> res;
        res.reserve(point_to_segment_map.size());
        for (size_t i = 0; i < point_to_segment_map.size(); i++) {
            if (point_to_segment_map[i] == no_label) res.emplace_back(i);
        }
        return res;
    }

    // CRTP base class that holds clustering results and accessors
    template <typename Derived = void>
    class ClusteringBase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        inline const std::vector<std::vector<size_t>>& getClusterToPointIndicesMap() const { return cluster_to_point_indices_map_; }

        inline const std::vector<size_t>& getPointToClusterIndexMap() const { return point_to_cluster_index_map_; }

        inline size_t getNumberOfClusters() const { return cluster_to_point_indices_map_.size(); }

        inline size_t getNumberOfPoints() const { return point_to_cluster_index_map_.size(); }

        inline std::vector<size_t> getUnlabeledPointIndices() const {
            return cilantro::getUnlabeledPointIndices(cluster_to_point_indices_map_, point_to_cluster_index_map_);
        }

    protected:
        std::vector<std::vector<size_t>> cluster_to_point_indices_map_;
        std::vector<size_t> point_to_cluster_index_map_;
    };
}
