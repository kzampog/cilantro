#pragma once

#include <cilantro/cartesian_grid.hpp>
#include <cilantro/point_cloud.hpp>

namespace cilantro {
    class VoxelGrid : public CartesianGrid3D {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        VoxelGrid(const PointCloud &cloud, float bin_size);

        std::vector<Eigen::Vector3f> getDownsampledPoints(size_t min_points_in_bin = 1) const;

        std::vector<Eigen::Vector3f> getDownsampledNormals(size_t min_points_in_bin = 1) const;

        std::vector<Eigen::Vector3f> getDownsampledColors(size_t min_points_in_bin = 1) const;

        PointCloud getDownsampledCloud(size_t min_points_in_bin = 1) const;

    private:
        const PointCloud& input_cloud_;
    };
}
