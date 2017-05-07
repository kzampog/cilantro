#include <sisyphus/voxel_grid.hpp>
//#include <sisyphus/pca.hpp>

VoxelGrid::VoxelGrid(const PointCloud &cloud, float bin_size)
        : bin_size_(bin_size),
          cloud_ref_(cloud),
          num_points_(cloud.points.size())
{
    min_pt_ = Eigen::Map<Eigen::MatrixXf>((float *)cloud.points.data(), 3, num_points_).rowwise().minCoeff();

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
            bin_points.col(i) = cloud_ref_.points[it->second[i]];
        }
        res.points.push_back(bin_points.rowwise().mean());
    }

//    if (cloud_ref_.points.size() == cloud_ref_.normals.size()) {
//        res.normals.reserve(grid_lookup_table_.size());
//        for (auto it = grid_lookup_table_.begin(); it != grid_lookup_table_.end(); ++it) {
//            if (it->second.size() < min_points_in_bin) continue;
//
//            Eigen::MatrixXf bin_normals(3, it->second.size()*2);
//            Eigen::Vector3f ref_dir = cloud_ref_.normals[it->second[0]];
//            size_t pos = 0, neg = 0;
//            for (int i = 0; i < it->second.size(); i++) {
//                const Eigen::Vector3f& curr_normal = cloud_ref_.normals[it->second[i]];
//                if (ref_dir.dot(curr_normal) < 0.0f) neg++; else pos++;
//                bin_normals.col(2*i) = -curr_normal;
//                bin_normals.col(2*i+1) = curr_normal;
//            }
//            if (neg > pos) ref_dir = -ref_dir;
//
//            PCA pca(bin_normals.data(), it->second.size()*2);
//            Eigen::Vector3f avg = pca.getEigenVectors().col(0);
//            if (ref_dir.dot(avg) < 0.0f) {
//                res.normals.push_back(-avg);
//            } else {
//                res.normals.push_back(avg);
//            }
//        }
//    }

    if (cloud_ref_.points.size() == cloud_ref_.normals.size()) {
        res.normals.reserve(grid_lookup_table_.size());
        for (auto it = grid_lookup_table_.begin(); it != grid_lookup_table_.end(); ++it) {
            if (it->second.size() < min_points_in_bin) continue;

            Eigen::MatrixXf bin_normals(3, it->second.size());
            Eigen::Vector3f ref_dir = cloud_ref_.normals[it->second[0]];
            size_t pos = 0, neg = 0;
            for (int i = 0; i < it->second.size(); i++) {
                const Eigen::Vector3f& curr_normal = cloud_ref_.normals[it->second[i]];
                if (ref_dir.dot(curr_normal) < 0.0f) {
                    bin_normals.col(i) = -curr_normal;
                    neg++;
                } else {
                    bin_normals.col(i) = curr_normal;
                    pos++;
                }
            }

            Eigen::Vector3f avg = bin_normals.rowwise().mean().normalized();
            if (neg > pos) {
                res.normals.push_back(-avg);
            } else {
                res.normals.push_back(avg);
            }
        }
    }

    if (cloud_ref_.points.size() == cloud_ref_.colors.size()) {
        res.colors.reserve(grid_lookup_table_.size());
        for (auto it = grid_lookup_table_.begin(); it != grid_lookup_table_.end(); ++it) {
            if (it->second.size() < min_points_in_bin) continue;

            Eigen::MatrixXf bin_colors(3, it->second.size());
            for (int i = 0; i < it->second.size(); i++) {
                bin_colors.col(i) = cloud_ref_.colors[it->second[i]];
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
    return VoxelGrid::getGridBinNeighbors(cloud_ref_.points[point_ind]);
}
