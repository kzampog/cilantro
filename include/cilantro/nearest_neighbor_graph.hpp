#pragma once

#include <Eigen/Sparse>
#include <cilantro/kd_tree.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    class NearestNeighborGraph {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        NearestNeighborGraph() {}

        // Query set is the same as the indexed set
        NearestNeighborGraph(const KDTree<ScalarT,EigenDim,DistAdaptor> &points_tree,
                             const NeighborhoodSpecification<ScalarT> &nh,
                             bool remove_self_from_nn = false)
                : nh_(nh)
        {
            compute_(points_tree, points_tree.getPointsMatrixMap(), remove_self_from_nn);
        }

        // Query set is the same as the indexed set
        NearestNeighborGraph(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                             const NeighborhoodSpecification<ScalarT> &nh,
                             bool remove_self_from_nn = false)
                : nh_(nh)
        {
            compute_(KDTree<ScalarT,EigenDim,DistAdaptor>(points), points, remove_self_from_nn);
        }

        // Query and indexed sets are different
        NearestNeighborGraph(const KDTree<ScalarT,EigenDim,DistAdaptor> &indexed_points_tree,
                             const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_points,
                             const NeighborhoodSpecification<ScalarT> &nh)
                : nh_(nh)
        {
            compute_(indexed_points_tree, query_points, false);
        }

        // Query and indexed sets are different
        NearestNeighborGraph(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &indexed_points,
                             const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_points,
                             const NeighborhoodSpecification<ScalarT> &nh)
                : nh_(nh)
        {
            compute_(KDTree<ScalarT,EigenDim,DistAdaptor>(indexed_points), query_points, false);
        }

        ~NearestNeighborGraph() {}

        inline const std::vector<std::vector<size_t>>& getAdjacencyList() const { return neighbor_indices_; }

        inline const std::vector<std::vector<ScalarT>>& getNeighborDistances() const { return neighbor_distances_; }

        inline const NeighborhoodSpecification<ScalarT>& getNeighborhoodSpecification() const { return nh_; }

        std::vector<size_t> getNodeDegrees() const {
            std::vector<size_t> deg(neighbor_indices_.size());
            for (size_t i = 0; i < deg.size(); i++) {
                deg[i] = neighbor_indices_[i].size();
            }
            return deg;
        }

        size_t getMaxNodeDegree() const {
            size_t max = 0;
            for (size_t i = 0; i < neighbor_indices_.size(); i++) {
                if (max < neighbor_indices_[i].size()) max = neighbor_indices_[i].size();
            }
            return max;
        }

        size_t getSumOfNodeDegrees() const {
            size_t sum = 0;
            for (size_t i = 0; i < neighbor_indices_.size(); i++) {
                sum += neighbor_indices_[i].size();
            }
            return sum;
        }

        template <class PairEvaluatorT, typename ValueT = decltype(std::declval<PairEvaluatorT>().getValue((size_t)0,(size_t)0,(ScalarT)0))>
        std::vector<std::vector<ValueT>> getFunctionValueList(const PairEvaluatorT &evaluator) const {
            std::vector<std::vector<ValueT>> f_values(neighbor_indices_.size());
            for (size_t i = 0; i < f_values.size(); i++) {
                f_values[i].resize(neighbor_indices_[i].size());
                for (size_t j = 0; j < f_values[i].size(); j++) {
                    f_values[i][j] = evaluator.getValue(i, neighbor_indices_[i][j], neighbor_distances_[i][j]);
                }
            }
            return f_values;
        }

        template <typename ValueT = bool>
        inline Eigen::Matrix<ValueT,Eigen::Dynamic,Eigen::Dynamic> getDenseAdjacencyMatrix(bool force_symmetry = false) const {
            return getFunctionValueDenseMatrix<AdjacencyEvaluator_<ValueT>,ValueT>(AdjacencyEvaluator_<ValueT>(), force_symmetry);
        }

        template <typename ValueT = bool>
        inline Eigen::SparseMatrix<ValueT> getSparseAdjacencyMatrix(bool force_symmetry = false) const {
            return getFunctionValueSparseMatrix<AdjacencyEvaluator_<ValueT>,ValueT>(AdjacencyEvaluator_<ValueT>(), force_symmetry);
        }

        template <typename ValueT = ScalarT>
        inline Eigen::Matrix<ValueT,Eigen::Dynamic,Eigen::Dynamic> getDenseDistanceMatrix(bool force_symmetry = false) const {
            return getFunctionValueDenseMatrix<DistanceEvaluator_<ValueT>,ValueT>(DistanceEvaluator_<ValueT>(), force_symmetry);
        }

        template <typename ValueT = ScalarT>
        inline Eigen::SparseMatrix<ValueT> getSparseDistanceMatrix(bool force_symmetry = false) const {
            return getFunctionValueSparseMatrix<DistanceEvaluator_<ValueT>,ValueT>(DistanceEvaluator_<ValueT>(), force_symmetry);
        }

        template <class PairEvaluatorT, typename ValueT = decltype(std::declval<PairEvaluatorT>().getValue((size_t)0,(size_t)0,(ScalarT)0))>
        Eigen::Matrix<ValueT,Eigen::Dynamic,Eigen::Dynamic> getFunctionValueDenseMatrix(const PairEvaluatorT &evaluator, bool force_symmetry = false) const {
            Eigen::Matrix<ValueT,Eigen::Dynamic,Eigen::Dynamic> mat(Eigen::Matrix<ValueT,Eigen::Dynamic,Eigen::Dynamic>::Zero(neighbor_indices_.size(),neighbor_indices_.size()));
            if (force_symmetry) {
                for (size_t i = 0; i < neighbor_indices_.size(); i++) {
                    for (size_t j = 0; j < neighbor_indices_[i].size(); j++) {
                        ValueT val = evaluator.getValue(i,neighbor_indices_[i][j],neighbor_distances_[i][j]);
                        mat(neighbor_indices_[i][j],i) = val;
                        mat(i,neighbor_indices_[i][j]) = val;
                    }
                }
            } else {
                for (size_t i = 0; i < neighbor_indices_.size(); i++) {
                    for (size_t j = 0; j < neighbor_indices_[i].size(); j++) {
                        mat(neighbor_indices_[i][j],i) = evaluator.getValue(i,neighbor_indices_[i][j],neighbor_distances_[i][j]);
                    }
                }
            }
            return mat;
        }

        template <class PairEvaluatorT, typename ValueT = decltype(std::declval<PairEvaluatorT>().getValue((size_t)0,(size_t)0,(ScalarT)0))>
        Eigen::SparseMatrix<ValueT> getFunctionValueSparseMatrix(const PairEvaluatorT &evaluator, bool force_symmetry = false) const {
            std::vector<Eigen::Triplet<ValueT>> triplet_list;
            if (force_symmetry) {
                triplet_list.reserve(2*getSumOfNodeDegrees());
                for (size_t i = 0; i < neighbor_indices_.size(); i++) {
                    for (size_t j = 0; j < neighbor_indices_[i].size(); j++) {
                        ValueT val = evaluator.getValue(i,neighbor_indices_[i][j],neighbor_distances_[i][j]);
                        triplet_list.emplace_back(neighbor_indices_[i][j], i, val);
                        triplet_list.emplace_back(i, neighbor_indices_[i][j], val);
                    }
                }
            } else {
                triplet_list.reserve(getSumOfNodeDegrees());
                for (size_t i = 0; i < neighbor_indices_.size(); i++) {
                    for (size_t j = 0; j < neighbor_indices_[i].size(); j++) {
                        triplet_list.emplace_back(neighbor_indices_[i][j], i, evaluator.getValue(i,neighbor_indices_[i][j],neighbor_distances_[i][j]));
                    }
                }
            }

            Eigen::SparseMatrix<ValueT> mat(neighbor_indices_.size(),neighbor_indices_.size());
            mat.setFromTriplets(triplet_list.begin(), triplet_list.end());

            return mat;
        }

    private:
        std::vector<std::vector<size_t>> neighbor_indices_;
        std::vector<std::vector<ScalarT>> neighbor_distances_;
        NeighborhoodSpecification<ScalarT> nh_;

        inline void compute_(const KDTree<ScalarT,EigenDim,DistAdaptor> &ref_tree,
                             const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_points,
                             bool remove_self_from_nn)
        {
            switch (nh_.type) {
                case NeighborhoodType::KNN:
                    compute_<NeighborhoodType::KNN>(ref_tree, query_points, remove_self_from_nn);
                    break;
                case NeighborhoodType::RADIUS:
                    compute_<NeighborhoodType::RADIUS>(ref_tree, query_points, remove_self_from_nn);
                    break;
                case NeighborhoodType::KNN_IN_RADIUS:
                    compute_<NeighborhoodType::KNN_IN_RADIUS>(ref_tree, query_points, remove_self_from_nn);
                    break;
            }
        }

        template <NeighborhoodType NT>
        inline void compute_(const KDTree<ScalarT,EigenDim,DistAdaptor> &ref_tree,
                      const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_points,
                      bool remove_self_from_nn)
        {
            neighbor_indices_.resize(query_points.cols());
            neighbor_distances_.resize(query_points.cols());
            std::vector<size_t> neighbors;
            std::vector<ScalarT> distances;
            neighbors.reserve(query_points.cols());
            distances.reserve(query_points.cols());

            if (remove_self_from_nn) {
                nh_.maxNumberOfNeighbors++;
#pragma omp parallel for private (neighbors, distances)
                for (size_t i = 0; i < query_points.cols(); i++) {
                    ref_tree.template search<NT>(query_points.col(i), nh_, neighbors, distances);
                    if (!neighbors.empty()) {
                        neighbor_indices_[i] = std::vector<size_t>(neighbors.begin()+1, neighbors.end());
                        neighbor_distances_[i] = std::vector<ScalarT>(distances.begin()+1, distances.end());
                    }
                }
                nh_.maxNumberOfNeighbors--;
            } else {
#pragma omp parallel for private (neighbors, distances)
                for (size_t i = 0; i < query_points.cols(); i++) {
                    ref_tree.template search<NT>(query_points.col(i), nh_, neighbors, distances);
                    neighbor_indices_[i] = neighbors;
                    neighbor_distances_[i] = distances;
                }
            }
        }

        // Dummy function evaluators for neighboring point pairs
        template <typename ValueT>
        struct AdjacencyEvaluator_ {
            inline ValueT getValue(size_t pt_ind, size_t nn_ind, ScalarT dist) const {
                return 1;
            }
        };

        template <typename ValueT>
        struct DistanceEvaluator_ {
            inline ValueT getValue(size_t pt_ind, size_t nn_ind, ScalarT dist) const {
                return dist;
            }
        };
    };

    typedef NearestNeighborGraph<float,2> NearestNeighborGraph2D;
    typedef NearestNeighborGraph<float,3> NearestNeighborGraph3D;
}
