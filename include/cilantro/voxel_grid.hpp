#pragma once

#include <cilantro/cartesian_grid.hpp>
#include <cilantro/point_cloud.hpp>

namespace cilantro {
    class VoxelGrid : public CartesianGrid3D {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        VoxelGrid(const PointCloud<float,3> &cloud, float bin_size);

        VectorSet<float,3> getDownsampledPoints(size_t min_points_in_bin = 1) const;

        VectorSet<float,3> getDownsampledNormals(size_t min_points_in_bin = 1) const;

        VectorSet<float,3> getDownsampledColors(size_t min_points_in_bin = 1) const;

        PointCloud<float,3> getDownsampledCloud(size_t min_points_in_bin = 1) const;

    private:
        const PointCloud<float,3>& input_cloud_;
    };
}
