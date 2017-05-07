#pragma once

//#include <unordered_map>
#include <map>
#include <sisyphus/point_cloud.hpp>

class VoxelGrid {
public:
    VoxelGrid(const PointCloud &cloud, float bin_size);
    ~VoxelGrid() {}

    PointCloud getDownsampledCloud(int min_points_in_bin = 1);

    std::vector<int> getGridBinNeighbors(const Eigen::Vector3f &point);
    std::vector<int> getGridBinNeighbors(int point_ind);

private:

//    class EigenVector3iHasher_ {
//    public:
//        inline size_t operator()(const Eigen::Vector3i &p) const {
//            size_t seed = 0;
//            for (size_t i = 0; i < 3; ++i) {
//                seed ^= std::hash<float>()(p(i)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
//            }
//            return seed;
//        }
//    };

    class EigenVector3iComparator_ {
    public:
        inline bool operator()(const Eigen::Vector3i &p1, const Eigen::Vector3i &p2) const {
            if (p1(0) < p2(0)) return true;
            if (p1(0) == p2(0)) {
                if (p1(1) < p2(1)) return true;
                if (p1(1) == p2(1)) {
                    return p1(2) < p2(2);
                } else return false;
            } else return false;
        }
    };

    const PointCloud &cloud_ref_;
    int num_points_;
    Eigen::Vector3f min_pt_;
    float bin_size_;

    std::map<Eigen::Vector3i, std::vector<int>, EigenVector3iComparator_> grid_lookup_table_;
//    std::unordered_map<Eigen::Vector3i, std::vector<int>, EigenVector3iHasher_> grid_lookup_table_;

};
