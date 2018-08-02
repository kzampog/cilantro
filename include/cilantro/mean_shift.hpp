#pragma once

#include <cilantro/kd_tree.hpp>
#include <cilantro/common_pair_evaluators.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    class MeanShift {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum { Dimension = EigenDim };

        MeanShift(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, size_t max_leaf_size = 10)
                : data_map_(points),
                  kd_tree_ptr_(new KDTree<ScalarT,EigenDim,DistAdaptor>(points, max_leaf_size)),
                  kd_tree_owned_(true),
                  iteration_count_(0)
        {}

        MeanShift(const KDTree<ScalarT,EigenDim,DistAdaptor> &kd_tree)
                : data_map_(kd_tree.getPointsMatrixMap()),
                  kd_tree_ptr_(&kd_tree),
                  kd_tree_owned_(false),
                  iteration_count_(0)
        {}

        ~MeanShift() {
            if (kd_tree_owned_) delete kd_tree_ptr_;
        }

        template <class KernelEvaluatorT = UnityWeightEvaluator<ScalarT,ScalarT>>
        MeanShift& cluster(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &seeds,
                           ScalarT kernel_radius,
                           size_t max_iter,
                           ScalarT cluster_tol,
                           ScalarT convergence_tol = std::numeric_limits<ScalarT>::epsilon(),
                           const KernelEvaluatorT &evaluator = KernelEvaluatorT())
        {
            shifted_seeds_ = seeds;

            // Shift points
            const ScalarT radius_sq = kernel_radius*kernel_radius;
            const ScalarT conv_tol_sq = convergence_tol*convergence_tol;
            iteration_count_ = 0;

            std::vector<char> has_converged(shifted_seeds_.cols(), 0);
            bool all_converged;
            NeighborSet<ScalarT> nn;
            Vector<ScalarT,EigenDim> point_tmp;

            while (iteration_count_ < max_iter) {
                all_converged = true;
#pragma omp parallel for shared (has_converged, all_converged) private (nn, point_tmp)
                for (size_t i = 0; i < shifted_seeds_.cols(); i++) {
                    if (has_converged[i]) continue;
                    kd_tree_ptr_->radiusSearch(shifted_seeds_.col(i), radius_sq, nn);
                    point_tmp.setZero(shifted_seeds_.rows(), 1);
                    ScalarT total_weight = (ScalarT)0;
                    for (size_t j = 0; j < nn.size(); j++) {
                        const ScalarT weight = evaluator.template operator()<Eigen::Ref<const Vector<ScalarT,EigenDim>>>(shifted_seeds_.col(i), data_map_.col(nn[j].index), nn[j].value);
                        point_tmp.noalias() += weight*data_map_.col(nn[j].index);
                        total_weight += weight;
                    }
                    point_tmp *= (ScalarT)(1.0)/total_weight;
                    if ((shifted_seeds_.col(i) - point_tmp).squaredNorm() < conv_tol_sq) {
                        has_converged[i] = 1;
                    } else {
                        all_converged = false;
                    }
                    shifted_seeds_.col(i) = point_tmp;
                }

                iteration_count_++;

                if (all_converged) break;
            }

            // Cluster
            const ScalarT cluster_tol_sq = cluster_tol*cluster_tol;

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
                cluster_modes_.col(i) *= (ScalarT)(1.0)/cluster_point_indices_[i].size();
            }

            return *this;
        }

        template <class KernelEvaluatorT = UnityWeightEvaluator<ScalarT,ScalarT>>
        inline MeanShift& cluster(ScalarT kernel_radius,
                                  size_t max_iter,
                                  ScalarT cluster_tol,
                                  ScalarT convergence_tol = std::numeric_limits<ScalarT>::epsilon(),
                                  const KernelEvaluatorT &evaluator = KernelEvaluatorT())
        {
            return cluster<KernelEvaluatorT>(data_map_, kernel_radius, max_iter, cluster_tol, convergence_tol, evaluator);
        }

        inline const VectorSet<ScalarT,EigenDim>& getShiftedSeeds() const { return shifted_seeds_; }

        inline const VectorSet<ScalarT,EigenDim>& getClusterModes() const { return cluster_modes_; }

        inline const std::vector<std::vector<size_t>>& getClusterPointIndices() const { return cluster_point_indices_; }

        inline const std::vector<size_t>& getClusterIndexMap() const { return cluster_index_map_; }

        inline size_t getNumberOfClusters() const { return cluster_point_indices_.size(); }

        inline size_t getNumberOfPerformedIterations() const { return iteration_count_; }

    private:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> data_map_;
        const KDTree<ScalarT,EigenDim,DistAdaptor> *kd_tree_ptr_;
        bool kd_tree_owned_;
        size_t iteration_count_;

        VectorSet<ScalarT,EigenDim> shifted_seeds_;
        VectorSet<ScalarT,EigenDim> cluster_modes_;
        std::vector<std::vector<size_t>> cluster_point_indices_;
        std::vector<size_t> cluster_index_map_;
    };

    typedef MeanShift<float,2,KDTreeDistanceAdaptors::L2> MeanShift2f;
    typedef MeanShift<double,2,KDTreeDistanceAdaptors::L2> MeanShift2d;
    typedef MeanShift<float,3,KDTreeDistanceAdaptors::L2> MeanShift3f;
    typedef MeanShift<double,3,KDTreeDistanceAdaptors::L2> MeanShift3d;
    typedef MeanShift<float,Eigen::Dynamic,KDTreeDistanceAdaptors::L2> MeanShiftXf;
    typedef MeanShift<double,Eigen::Dynamic,KDTreeDistanceAdaptors::L2> MeanShiftXd;
}
