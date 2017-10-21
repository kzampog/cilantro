#pragma once

//#include <unordered_map>
#include <map>
#include <cilantro/point_cloud.hpp>

namespace cilantro {
    class VoxelGrid {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        VoxelGrid(const std::vector<Eigen::Vector3f> &points, float bin_size);
        VoxelGrid(const PointCloud &cloud, float bin_size);
        ~VoxelGrid() {}

        std::vector<Eigen::Vector3f> getDownsampledPoints(size_t min_points_in_bin = 1) const;
        std::vector<Eigen::Vector3f> getDownsampledNormals(size_t min_points_in_bin = 1) const;
        std::vector<Eigen::Vector3f> getDownsampledColors(size_t min_points_in_bin = 1) const;

        PointCloud getDownsampledCloud(size_t min_points_in_bin = 1) const;

        const std::vector<size_t>& getGridBinNeighbors(const Eigen::Vector3f &point) const;
        const std::vector<size_t>& getGridBinNeighbors(size_t point_ind) const;

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

//    class EigenVector3iComparator_ {
//    public:
//        inline bool operator()(const Eigen::Vector3i &p1, const Eigen::Vector3i &p2) const {
//            if (p1[0] < p2[0]) return true;
//            if (p1[0] == p2[0]) {
//                if (p1[1] < p2[1]) return true;
//                if (p1[1] == p2[1]) {
//                    return p1[2] < p2[2];
//                } else return false;
//            } else return false;
//        }
//    };

        const std::vector<Eigen::Vector3f> * input_points_;
        const std::vector<Eigen::Vector3f> * input_normals_;
        const std::vector<Eigen::Vector3f> * input_colors_;

        float bin_size_;
//    Eigen::Vector3f min_pt_;
        std::array<float,3> min_pt_;

        std::vector<size_t> empty_indices_;

        std::map<std::array<int,3>,std::vector<size_t> > grid_lookup_table_;
//    std::map<Eigen::Vector3i, std::vector<size_t>, EigenVector3iComparator_> grid_lookup_table_;
//    std::unordered_map<Eigen::Vector3i, std::vector<size_t>, EigenVector3iHasher_> grid_lookup_table_;

        std::vector<std::map<std::array<int,3>,std::vector<size_t> >::iterator> map_iterators_;

        void build_lookup_table_();
    };
}
