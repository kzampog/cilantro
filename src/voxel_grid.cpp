#include <cilantro/voxel_grid.hpp>
//#include <cilantro/pca.hpp>

VoxelGrid::VoxelGrid(const std::vector<Eigen::Vector3f> &points, float bin_size)
        : input_points_(&points),
          input_normals_(NULL),
          input_colors_(NULL),
          bin_size_(bin_size),
          empty_indices_(0)
{
    build_lookup_table_();
}

VoxelGrid::VoxelGrid(const PointCloud &cloud, float bin_size)
        : input_points_(&cloud.points),
          input_normals_(cloud.hasNormals()?&cloud.normals:NULL),
          input_colors_(cloud.hasColors()?&cloud.colors:NULL),
          bin_size_(bin_size),
          empty_indices_(0)
{
    build_lookup_table_();
}

std::vector<Eigen::Vector3f> VoxelGrid::getDownsampledPoints(size_t min_points_in_bin) const {
    std::vector<Eigen::Vector3f> points;
    points.reserve(grid_lookup_table_.size());
    for (auto it = grid_lookup_table_.begin(); it != grid_lookup_table_.end(); ++it) {
        if (it->second.size() < min_points_in_bin) continue;

        Eigen::MatrixXf bin_points(3, it->second.size());
        for (size_t i = 0; i < it->second.size(); i++) {
            bin_points.col(i) = (*input_points_)[it->second[i]];
        }
        points.emplace_back(bin_points.rowwise().mean());
    }

    return points;
}

std::vector<Eigen::Vector3f> VoxelGrid::getDownsampledNormals(size_t min_points_in_bin) const {
    std::vector<Eigen::Vector3f> normals;
    if (input_normals_ == NULL) {
        return normals;
    }

//    normals.reserve(grid_lookup_table_.size());
//    for (auto it = grid_lookup_table_.begin(); it != grid_lookup_table_.end(); ++it) {
//        if (it->second.size() < min_points_in_bin) continue;
//
//        Eigen::MatrixXf bin_normals(3, it->second.size()*2);
//        Eigen::Vector3f ref_dir = (*input_normals_)[it->second[0]];
//        size_t pos = 0, neg = 0;
//        for (size_t i = 0; i < it->second.size(); i++) {
//            const Eigen::Vector3f& curr_normal = (*input_normals_)[it->second[i]];
//            if (ref_dir.dot(curr_normal) < 0.0f) neg++; else pos++;
//            bin_normals.col(2*i) = -curr_normal;
//            bin_normals.col(2*i+1) = curr_normal;
//        }
//        if (neg > pos) ref_dir = -ref_dir;
//
//        PCA3D pca(bin_normals.data(), 3, it->second.size()*2);
//        Eigen::Vector3f avg = pca.getEigenVectorsMatrix().col(0);
//        if (ref_dir.dot(avg) < 0.0f) {
//            normals.emplace_back(-avg);
//        } else {
//            normals.emplace_back(avg);
//        }
//    }

    normals.reserve(grid_lookup_table_.size());
    for (auto it = grid_lookup_table_.begin(); it != grid_lookup_table_.end(); ++it) {
        if (it->second.size() < min_points_in_bin) continue;

        Eigen::MatrixXf bin_normals(3, it->second.size());
        Eigen::Vector3f ref_dir = (*input_normals_)[it->second[0]];
        size_t pos = 0, neg = 0;
        for (size_t i = 0; i < it->second.size(); i++) {
            const Eigen::Vector3f& curr_normal = (*input_normals_)[it->second[i]];
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
            normals.emplace_back(-avg);
        } else {
            normals.emplace_back(avg);
        }
    }

    return normals;
}

std::vector<Eigen::Vector3f> VoxelGrid::getDownsampledColors(size_t min_points_in_bin) const {
    std::vector<Eigen::Vector3f> colors;
    if (input_colors_ == NULL) return colors;

    colors.reserve(grid_lookup_table_.size());
    for (auto it = grid_lookup_table_.begin(); it != grid_lookup_table_.end(); ++it) {
        if (it->second.size() < min_points_in_bin) continue;

        Eigen::MatrixXf bin_colors(3, it->second.size());
        for (size_t i = 0; i < it->second.size(); i++) {
            bin_colors.col(i) = (*input_colors_)[it->second[i]];
        }
        colors.emplace_back(bin_colors.rowwise().mean());
    }

    return colors;
}

PointCloud VoxelGrid::getDownsampledCloud(size_t min_points_in_bin) const {
    return PointCloud(getDownsampledPoints(min_points_in_bin), getDownsampledNormals(min_points_in_bin), getDownsampledColors(min_points_in_bin));
}

const std::vector<size_t>& VoxelGrid::getGridBinNeighbors(const Eigen::Vector3f &point) const {
    Eigen::Vector3i grid_coords = ((point-min_pt_)/bin_size_).array().floor().cast<int>();
    if ((grid_coords.array() < 0).any()) return empty_indices_;

    auto it = grid_lookup_table_.find(grid_coords);
    if (it == grid_lookup_table_.end()) return empty_indices_;

    return it->second;
}

const std::vector<size_t>& VoxelGrid::getGridBinNeighbors(size_t point_ind) const {
    return VoxelGrid::getGridBinNeighbors((*input_points_)[point_ind]);
}

void VoxelGrid::build_lookup_table_() {
    if (input_points_->empty()) return;

    min_pt_ = Eigen::Map<Eigen::MatrixXf>((float *)input_points_->data(), 3, input_points_->size()).rowwise().minCoeff();
    Eigen::MatrixXi grid_coords = ((Eigen::Map<Eigen::MatrixXf>((float *)input_points_->data(), 3, input_points_->size()).colwise()-min_pt_)/bin_size_).array().floor().cast<int>();

    // Build lookup table
    for (size_t i = 0; i < input_points_->size(); i++) {
        auto it = grid_lookup_table_.find(grid_coords.col(i));
        if (it == grid_lookup_table_.end()) {
            grid_lookup_table_.insert(std::pair<Eigen::Vector3i,std::vector<size_t> >(grid_coords.col(i), std::vector<size_t>(1, i)));
        } else {
            it->second.emplace_back(i);
        }
    }
}
