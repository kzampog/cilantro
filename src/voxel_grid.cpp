#include <cilantro/voxel_grid.hpp>

namespace cilantro {
    VoxelGrid::VoxelGrid(const PointCloud &cloud, float bin_size)
            : CartesianGrid3D(cloud.points, bin_size),
              input_cloud_(cloud)
    {}

    std::vector<Eigen::Vector3f> VoxelGrid::getDownsampledPoints(size_t min_points_in_bin) const {
        std::vector<Eigen::Vector3f> points;
        points.reserve(grid_lookup_table_.size());

        Eigen::Vector3f point;
        float scale;

        for (size_t k = 0; k < bin_iterators_.size(); k++) {
            auto it = bin_iterators_[k];

            const std::vector<size_t>& bin_ind(it->second);
            if (bin_ind.size() < min_points_in_bin) continue;

            scale = 1.0f/bin_ind.size();

            point.setZero();
            for (size_t i = 0; i < bin_ind.size(); i++) {
                point += input_cloud_.points[bin_ind[i]];
            }

            points.emplace_back(scale*point);
        }

        return points;
    }

    std::vector<Eigen::Vector3f> VoxelGrid::getDownsampledNormals(size_t min_points_in_bin) const {
        std::vector<Eigen::Vector3f> normals;
        if (!input_cloud_.hasNormals()) {
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

        Eigen::Vector3f normal;

        for (size_t k = 0; k < bin_iterators_.size(); k++) {
            auto it = bin_iterators_[k];

            const std::vector<size_t>& bin_ind(it->second);
            if (bin_ind.size() < min_points_in_bin) continue;

            normal.setZero();
            Eigen::Vector3f ref_dir = input_cloud_.normals[bin_ind[0]];
            size_t pos = 0, neg = 0;
            for (size_t i = 0; i < bin_ind.size(); i++) {
                const Eigen::Vector3f& curr_normal = input_cloud_.normals[bin_ind[i]];
                if (ref_dir.dot(curr_normal) < 0.0f) {
                    normal -= curr_normal;
                    neg++;
                } else {
                    normal += curr_normal;
                    pos++;
                }
            }
            if (neg > pos) normal *= -1.0f;

            normals.emplace_back(normal.normalized());
        }

        return normals;
    }

    std::vector<Eigen::Vector3f> VoxelGrid::getDownsampledColors(size_t min_points_in_bin) const {
        std::vector<Eigen::Vector3f> colors;
        if (!input_cloud_.hasColors()) return colors;

        colors.reserve(grid_lookup_table_.size());

        Eigen::Vector3f color;
        float scale;

        for (size_t k = 0; k < bin_iterators_.size(); k++) {
            auto it = bin_iterators_[k];

            const std::vector<size_t>& bin_ind(it->second);
            if (bin_ind.size() < min_points_in_bin) continue;

            scale = 1.0f/bin_ind.size();

            color.setZero();
            for (size_t i = 0; i < bin_ind.size(); i++) {
                color += input_cloud_.colors[bin_ind[i]];
            }

            colors.emplace_back(scale*color);
        }

        return colors;
    }

    PointCloud VoxelGrid::getDownsampledCloud(size_t min_points_in_bin) const {
        std::vector<Eigen::Vector3f> points, normals, colors;
        Eigen::Vector3f point, normal, color;

        bool do_normals = input_cloud_.hasNormals();
        bool do_colors = input_cloud_.hasColors();

        points.reserve(grid_lookup_table_.size());
        if (do_normals) normals.reserve(grid_lookup_table_.size());
        if (do_colors) colors.reserve(grid_lookup_table_.size());

        float scale;

//    for (auto it = grid_lookup_table_.begin(); it != grid_lookup_table_.end(); ++it) {
//#pragma omp parallel for shared (points, normals, colors) private (point, normal, color, scale) num_threads(4)
        for (size_t k = 0; k < bin_iterators_.size(); k++) {
            auto it = bin_iterators_[k];

            const std::vector<size_t>& bin_ind(it->second);
            if (bin_ind.size() < min_points_in_bin) continue;

            scale = 1.0f/bin_ind.size();

            point.setZero();
            for (size_t i = 0; i < bin_ind.size(); i++) {
                point += input_cloud_.points[bin_ind[i]];
            }

            points.emplace_back(scale*point);

            if (do_normals) {
                normal.setZero();
                Eigen::Vector3f ref_dir = input_cloud_.normals[bin_ind[0]];
                size_t pos = 0, neg = 0;
                for (size_t i = 0; i < bin_ind.size(); i++) {
                    const Eigen::Vector3f& curr_normal = input_cloud_.normals[bin_ind[i]];
                    if (ref_dir.dot(curr_normal) < 0.0f) {
                        normal -= curr_normal;
                        neg++;
                    } else {
                        normal += curr_normal;
                        pos++;
                    }
                }
                if (neg > pos) normal *= -1.0f;

                normals.emplace_back(normal.normalized());
            }

            if (do_colors) {
                color.setZero();
                for (size_t i = 0; i < bin_ind.size(); i++) {
                    color += input_cloud_.colors[bin_ind[i]];
                }

                colors.emplace_back(scale*color);
            }

//#pragma omp critical
//        {
//            points.emplace_back(scale*point);
//            if (do_normals) normals.emplace_back(normal.normalized());
//            if (do_colors) colors.emplace_back(scale*color);
//        }

        }

        return PointCloud(points, normals, colors);
    }
}
