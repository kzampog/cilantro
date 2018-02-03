#pragma once

#include <cilantro/kd_tree.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class MeanShift {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        MeanShift(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points)
                : data_map_(points),
                  kd_tree_ptr_(new KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2>(points)),
                  kd_tree_owned_(true),
                  iteration_count_(0)
        {}

        MeanShift(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> &kd_tree)
                : data_map_(points),
                  kd_tree_ptr_(&kd_tree),
                  kd_tree_owned_(false),
                  iteration_count_(0)
        {}

        ~MeanShift() {
            if (kd_tree_owned_) delete kd_tree_ptr_;
        }

        MeanShift& cluster(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &seeds, ScalarT kernel_radius, size_t max_iter, ScalarT cluster_tol, ScalarT convergence_tol = std::numeric_limits<ScalarT>::epsilon()) {
            shifted_seeds_ = seeds;

            // Shift points
            ScalarT radius_sq = kernel_radius*kernel_radius;
            ScalarT conv_tol_sq = convergence_tol*convergence_tol;
            iteration_count_ = 0;

            Eigen::Matrix<bool,Eigen::Dynamic,1> has_converged = Eigen::Matrix<bool,Eigen::Dynamic,1>::Constant(seeds.cols(), 1, false);
            Vector<ScalarT,EigenDim> point_tmp(shifted_seeds_.rows(), 1);
            ScalarT scale;
            bool all_converged;
            std::vector<size_t> neighbors;
            std::vector<ScalarT> distances;

            while (iteration_count_ < max_iter) {
                all_converged = true;
#pragma omp parallel for shared (has_converged, all_converged) private (point_tmp, scale, neighbors, distances)
                for (size_t i = 0; i < shifted_seeds_.cols(); i++) {
                    if (has_converged[i]) continue;
                    kd_tree_ptr_->radiusSearch(shifted_seeds_.col(i), radius_sq, neighbors, distances);
                    point_tmp.setZero();
                    for (size_t j = 0; j < neighbors.size(); j++) {
                        point_tmp += data_map_.col(neighbors[j]);
                    }
                    scale = 1.0/neighbors.size();
                    point_tmp *= scale;
                    if ((shifted_seeds_.col(i) - point_tmp).squaredNorm() < conv_tol_sq) {
                        has_converged[i] = true;
                    } else {
                        all_converged = false;
                    }
                    shifted_seeds_.col(i) = point_tmp;
                }

                iteration_count_++;

                if (all_converged) break;
            }

            // Cluster
            ScalarT cluster_tol_sq = cluster_tol*cluster_tol;

            for (size_t i = 0; i < shifted_seeds_.cols(); i++) {
                size_t c;
                for (c = 0; c < cluster_point_indices_.size(); c++) {
                    if ((shifted_seeds_.col(i) - shifted_seeds_.col(cluster_point_indices_[c][0])).squaredNorm() < cluster_tol_sq) break;
                }

                if (c == cluster_point_indices_.size()) {
                    cluster_point_indices_.emplace_back(1, i);
                } else {
                    cluster_point_indices_[c].emplace_back(i);
                }
            }

            cluster_index_map_.resize(shifted_seeds_.cols());
            cluster_modes_.resize(data_map_.rows(), cluster_point_indices_.size());
            for (size_t i = 0; i < cluster_point_indices_.size(); i++) {
                cluster_modes_.col(i).setZero();
                for (size_t j = 0; j < cluster_point_indices_[i].size(); j++) {
                    cluster_modes_.col(i) += shifted_seeds_.col(cluster_point_indices_[i][j]);
                    cluster_index_map_[cluster_point_indices_[i][j]] = i;
                }
                scale = 1.0/cluster_point_indices_[i].size();
                cluster_modes_.col(i) *= scale;
            }

            return *this;
        }

        inline MeanShift& cluster(ScalarT kernel_radius, size_t max_iter, ScalarT cluster_tol, ScalarT convergence_tol = std::numeric_limits<ScalarT>::epsilon()) {
            return cluster(data_map_, kernel_radius, max_iter, cluster_tol, convergence_tol);
        }

        inline const VectorSet<ScalarT,EigenDim>& getShiftedSeeds() const { return shifted_seeds_; }

        inline const VectorSet<ScalarT,EigenDim>& getClusterModes() const { return cluster_modes_; }

        inline const std::vector<std::vector<size_t>>& getClusterPointIndices() const { return cluster_point_indices_; }

        inline const std::vector<size_t>& getClusterIndexMap() const { return cluster_index_map_; }

        inline size_t getNumberOfClusters() const { return cluster_point_indices_.size(); }

        inline size_t getPerformedIterationsCount() const { return iteration_count_; }

    private:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> data_map_;
        const KDTree<ScalarT,EigenDim,KDTreeDistanceAdaptors::L2> *kd_tree_ptr_;
        bool kd_tree_owned_;
        size_t iteration_count_;

        VectorSet<ScalarT,EigenDim> shifted_seeds_;
        VectorSet<ScalarT,EigenDim> cluster_modes_;
        std::vector<std::vector<size_t>> cluster_point_indices_;
        std::vector<size_t> cluster_index_map_;
    };

    typedef MeanShift<float,2> MeanShift2D;
    typedef MeanShift<float,3> MeanShift3D;
}
