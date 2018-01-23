#include <cilantro/voxel_grid.hpp>

namespace cilantro {
    VoxelGrid::VoxelGrid(const PointCloud<float,3> &cloud, float bin_size)
            : CartesianGrid3D(cloud.points, bin_size),
              input_cloud_(cloud)
    {}

    VectorSet<float,3> VoxelGrid::getDownsampledPoints(size_t min_points_in_bin) const {
        VectorSet<float,3> points(3, grid_lookup_table_.size());

        float scale;
        size_t ind = 0;
        for (size_t k = 0; k < bin_iterators_.size(); k++) {
            auto it = bin_iterators_[k];

            const std::vector<size_t>& bin_ind(it->second);
            if (bin_ind.size() < min_points_in_bin) continue;

            scale = 1.0f/bin_ind.size();
            points.col(ind).setZero();
            for (size_t i = 0; i < bin_ind.size(); i++) {
                points.col(ind) += input_cloud_.points.col(bin_ind[i]);
            }
            points.col(ind) *= scale;

            ind++;
        }

        points.conservativeResize(Eigen::NoChange, ind);
        return points;
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

    VectorSet<float,3> VoxelGrid::getDownsampledNormals(size_t min_points_in_bin) const {
        if (!input_cloud_.hasNormals()) return VectorSet<float,3>();

        VectorSet<float,3> normals(3, grid_lookup_table_.size());

        size_t ind = 0;
        for (size_t k = 0; k < bin_iterators_.size(); k++) {
            auto it = bin_iterators_[k];

            const std::vector<size_t>& bin_ind(it->second);
            if (bin_ind.size() < min_points_in_bin) continue;

            normals.col(ind).setZero();
            const Eigen::Vector3f& ref_dir = input_cloud_.normals.col(bin_ind[0]);
            size_t pos = 0, neg = 0;
            for (size_t i = 0; i < bin_ind.size(); i++) {
                const Eigen::Vector3f& curr_normal = input_cloud_.normals.col(bin_ind[i]);
                if (ref_dir.dot(curr_normal) < 0.0f) {
                    normals.col(ind) -= curr_normal;
                    neg++;
                } else {
                    normals.col(ind) += curr_normal;
                    pos++;
                }
            }
            if (neg > pos) normals.col(ind) *= -1.0f;
            normals.col(ind).normalize();

            ind++;
        }

        normals.conservativeResize(Eigen::NoChange, ind);
        return normals;
    }

    VectorSet<float,3> VoxelGrid::getDownsampledColors(size_t min_points_in_bin) const {
        if (!input_cloud_.hasColors()) return VectorSet<float,3>();

        VectorSet<float,3> colors(3, grid_lookup_table_.size());

        float scale;
        size_t ind = 0;
        for (size_t k = 0; k < bin_iterators_.size(); k++) {
            auto it = bin_iterators_[k];

            const std::vector<size_t>& bin_ind(it->second);
            if (bin_ind.size() < min_points_in_bin) continue;

            scale = 1.0f/bin_ind.size();

            colors.col(ind).setZero();
            for (size_t i = 0; i < bin_ind.size(); i++) {
                colors.col(ind) += input_cloud_.colors.col(bin_ind[i]);
            }
            colors.col(ind) *= scale;

            ind++;
        }

        colors.conservativeResize(Eigen::NoChange, ind);
        return colors;
    }

    PointCloud<float,3> VoxelGrid::getDownsampledCloud(size_t min_points_in_bin) const {
        PointCloud<float,3> res_cloud;

        bool do_normals = input_cloud_.hasNormals();
        bool do_colors = input_cloud_.hasColors();

        res_cloud.points.resize(3, grid_lookup_table_.size());
        if (do_normals) res_cloud.normals.resize(3, grid_lookup_table_.size());
        if (do_colors) res_cloud.colors.resize(3, grid_lookup_table_.size());

        float scale;
        size_t ind = 0;
//    for (auto it = grid_lookup_table_.begin(); it != grid_lookup_table_.end(); ++it) {
//#pragma omp parallel for shared (points, normals, colors) private (point, normal, color, scale) num_threads(4)
        for (size_t k = 0; k < bin_iterators_.size(); k++) {
            auto it = bin_iterators_[k];

            const std::vector<size_t>& bin_ind(it->second);
            if (bin_ind.size() < min_points_in_bin) continue;

            scale = 1.0f/bin_ind.size();

            res_cloud.points.col(ind).setZero();
            for (size_t i = 0; i < bin_ind.size(); i++) {
                res_cloud.points.col(ind) += input_cloud_.points.col(bin_ind[i]);
            }
            res_cloud.points.col(ind) *= scale;

            if (do_normals) {
                res_cloud.normals.col(ind).setZero();
                const Eigen::Vector3f& ref_dir = input_cloud_.normals.col(bin_ind[0]);
                size_t pos = 0, neg = 0;
                for (size_t i = 0; i < bin_ind.size(); i++) {
                    const Eigen::Vector3f& curr_normal = input_cloud_.normals.col(bin_ind[i]);
                    if (ref_dir.dot(curr_normal) < 0.0f) {
                        res_cloud.normals.col(ind) -= curr_normal;
                        neg++;
                    } else {
                        res_cloud.normals.col(ind) += curr_normal;
                        pos++;
                    }
                }
                if (neg > pos) res_cloud.normals.col(ind) *= -1.0f;
                res_cloud.normals.col(ind).normalize();
            }

            if (do_colors) {
                res_cloud.colors.col(ind).setZero();
                for (size_t i = 0; i < bin_ind.size(); i++) {
                    res_cloud.colors.col(ind) += input_cloud_.colors.col(bin_ind[i]);
                }
                res_cloud.colors.col(ind) *= scale;
            }

            ind++;

//#pragma omp critical
//        {
//            points.emplace_back(scale*point);
//            if (do_normals) normals.emplace_back(normal.normalized());
//            if (do_colors) colors.emplace_back(scale*color);
//        }

        }

        res_cloud.points.conservativeResize(Eigen::NoChange, ind);
        res_cloud.normals.conservativeResize(Eigen::NoChange, ind);
        res_cloud.colors.conservativeResize(Eigen::NoChange, ind);

        return res_cloud;
    }
}
