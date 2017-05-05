#include <sisyphus/voxel_grid.hpp>

VoxelGrid::VoxelGrid(const PointCloud &cloud, float bin_size)
        : bin_size_(bin_size),
          cloud_ptr_(&cloud),
          num_points_(cloud.points.size())
{
    min_pt_ = Eigen::Map<Eigen::MatrixXf>((float *)cloud.points.data(), 3, num_points_).rowwise().minCoeff();
    min_pt_ -= Eigen::Vector3f(bin_size_, bin_size_, bin_size_)/2.0f;

    Eigen::MatrixXi grid_coords = ((Eigen::Map<Eigen::MatrixXf>((float *)cloud.points.data(), 3, num_points_).colwise()-min_pt_)/bin_size_).array().floor().cast<int>();

    // Build lookup table
    for (int i = 0; i < num_points_; i++) {
        auto it = grid_lookup_table_.find(grid_coords.col(i));
        if (it == grid_lookup_table_.end()) {
            grid_lookup_table_.insert(std::pair<Eigen::Vector3i,std::vector<int> >(grid_coords.col(i), std::vector<int>(1, i)));
        } else {
            it->second.push_back(i);
        }
    }
}

PointCloud VoxelGrid::getDownsampledCloud(int min_points_in_bin) {
    PointCloud res;

    res.points.reserve(grid_lookup_table_.size());
    for (auto it = grid_lookup_table_.begin(); it != grid_lookup_table_.end(); ++it) {
        if (it->second.size() < min_points_in_bin) continue;
        Eigen::MatrixXf bin_points(3, it->second.size());
        for (int i = 0; i < it->second.size(); i++) {
            bin_points.col(i) = cloud_ptr_->points[it->second[i]];
        }
        res.points.push_back(bin_points.rowwise().mean());
    }

    if (cloud_ptr_->points.size() == cloud_ptr_->normals.size()) {
        res.normals.reserve(grid_lookup_table_.size());
        for (auto it = grid_lookup_table_.begin(); it != grid_lookup_table_.end(); ++it) {
            if (it->second.size() < min_points_in_bin) continue;
            Eigen::MatrixXf bin_normals(3, it->second.size());
            for (int i = 0; i < it->second.size(); i++) {
                bin_normals.col(i) = cloud_ptr_->normals[it->second[i]];
            }
            res.normals.push_back(bin_normals.rowwise().mean().normalized());
        }
    }

    if (cloud_ptr_->points.size() == cloud_ptr_->colors.size()) {
        res.colors.reserve(grid_lookup_table_.size());
        for (auto it = grid_lookup_table_.begin(); it != grid_lookup_table_.end(); ++it) {
            if (it->second.size() < min_points_in_bin) continue;
            Eigen::MatrixXf bin_colors(3, it->second.size());
            for (int i = 0; i < it->second.size(); i++) {
                bin_colors.col(i) = cloud_ptr_->colors[it->second[i]];
            }
            res.colors.push_back(bin_colors.rowwise().mean());
        }
    }

    return res;
}

std::vector<int> VoxelGrid::getGridBinNeighbors(const Eigen::Vector3f &point) {
    Eigen::Vector3i grid_coords = ((point-min_pt_)/bin_size_).array().floor().cast<int>();
    if ((grid_coords.array() < Eigen::Vector3i::Zero().array()).any()) return std::vector<int>(0);

    auto it = grid_lookup_table_.find(grid_coords);
    if (it == grid_lookup_table_.end()) return std::vector<int>(0);

    return it->second;
}

std::vector<int> VoxelGrid::getGridBinNeighbors(int point_ind) {
    return VoxelGrid::getGridBinNeighbors(cloud_ptr_->points[point_ind]);
}
