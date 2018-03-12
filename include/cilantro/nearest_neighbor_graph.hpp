#pragma once

#include <Eigen/Sparse>
#include <cilantro/kd_tree.hpp>

namespace cilantro {
    // Dummy function evaluators for neighboring point pairs
    template <typename ScalarT, typename ValueT>
    struct AdjacencyEvaluator {
        inline ValueT getValue(size_t pt_ind, size_t nn_ind, ScalarT dist) const {
            return 1;
        }
    };

    template <typename ScalarT, typename ValueT>
    struct DistanceEvaluator {
        inline ValueT getValue(size_t pt_ind, size_t nn_ind, ScalarT dist) const {
            return dist;
        }
    };

    template <typename T>
    std::vector<size_t> getNNGraphNodeDegrees(const std::vector<std::vector<T>> &adj_list,
                                              bool remove_self = true)
    {
        std::vector<size_t> deg(adj_list.size());
        if (remove_self) {
            for (size_t i = 0; i < deg.size(); i++) {
                deg[i] = adj_list[i].size() - 1;
            }
        } else {
            for (size_t i = 0; i < deg.size(); i++) {
                deg[i] = adj_list[i].size();
            }
        }
        return deg;
    }

    template <typename T>
    size_t getNNGraphMaxNodeDegree(const std::vector<std::vector<T>> &adj_list,
                                   bool remove_self = true)
    {
        size_t max = 0;
        for (size_t i = 0; i < adj_list.size(); i++) {
            if (max < adj_list[i].size()) max = adj_list[i].size();
        }
        return (remove_self) ? max - 1 : max;
    }

    template <typename T>
    size_t getNNGraphSumOfNodeDegrees(const std::vector<std::vector<T>> &adj_list,
                                      bool remove_self = true)
    {
        size_t sum = 0;
        for (size_t i = 0; i < adj_list.size(); i++) {
            sum += adj_list[i].size();
        }
        return (remove_self) ? sum - adj_list.size() : sum;
    }

    template <typename ScalarT, class PairEvaluatorT, typename ValueT = decltype(std::declval<PairEvaluatorT>().getValue((size_t)0,(size_t)0,(ScalarT)0))>
    std::vector<std::vector<ValueT>> getNNGraphFunctionValueList(const std::vector<std::vector<NearestNeighborSearchResult<ScalarT>>> &adj_list,
                                                                 const PairEvaluatorT &evaluator = PairEvaluatorT())
    {
        std::vector<std::vector<ValueT>> f_values(adj_list.size());
#pragma omp parallel for shared (f_values)
        for (size_t i = 0; i < f_values.size(); i++) {
            f_values[i].resize(adj_list[i].size());
            for (size_t j = 0; j < f_values[i].size(); j++) {
                f_values[i][j] = evaluator.getValue(i, adj_list[i][j].index, adj_list[i][j].distance);
            }
        }
        return f_values;
    }

    template <typename ScalarT, class PairEvaluatorT, typename ValueT = decltype(std::declval<PairEvaluatorT>().getValue((size_t)0,(size_t)0,(ScalarT)0))>
    Eigen::Matrix<ValueT,Eigen::Dynamic,Eigen::Dynamic> getNNGraphFunctionValueDenseMatrix(const std::vector<std::vector<NearestNeighborSearchResult<ScalarT>>> &adj_list,
                                                                                           const PairEvaluatorT &evaluator = PairEvaluatorT(),
                                                                                           bool force_symmetry = false)
    {
        Eigen::Matrix<ValueT,Eigen::Dynamic,Eigen::Dynamic> mat(Eigen::Matrix<ValueT,Eigen::Dynamic,Eigen::Dynamic>::Zero(adj_list.size(),adj_list.size()));
        if (force_symmetry) {
            for (size_t i = 0; i < adj_list.size(); i++) {
                for (size_t j = 0; j < adj_list[i].size(); j++) {
                    ValueT val = evaluator.getValue(i, adj_list[i][j].index, adj_list[i][j].distance);
                    mat(adj_list[i][j].index, i) = val;
                    mat(i, adj_list[i][j].index) = val;
                }
            }
        } else {
            for (size_t i = 0; i < adj_list.size(); i++) {
                for (size_t j = 0; j < adj_list[i].size(); j++) {
                    mat(adj_list[i][j].index, i) = evaluator.getValue(i, adj_list[i][j].index, adj_list[i][j].distance);
                }
            }
        }
        return mat;
    }

    template <typename ScalarT, class PairEvaluatorT, typename ValueT = decltype(std::declval<PairEvaluatorT>().getValue((size_t)0,(size_t)0,(ScalarT)0))>
    Eigen::SparseMatrix<ValueT> getNNGraphFunctionValueSparseMatrix(const std::vector<std::vector<NearestNeighborSearchResult<ScalarT>>> &adj_list,
                                                                    const PairEvaluatorT &evaluator = PairEvaluatorT(),
                                                                    bool force_symmetry = false)
    {
        std::vector<Eigen::Triplet<ValueT>> triplet_list;
        if (force_symmetry) {
            triplet_list.reserve(2*getNNGraphSumOfNodeDegrees(adj_list, false));
            for (size_t i = 0; i < adj_list.size(); i++) {
                for (size_t j = 0; j < adj_list[i].size(); j++) {
                    ValueT val = evaluator.getValue(i, adj_list[i][j].index, adj_list[i][j].distance);
                    triplet_list.emplace_back(adj_list[i][j].index, i, val);
                    triplet_list.emplace_back(i, adj_list[i][j].index, val);
                }
            }
        } else {
            triplet_list.reserve(getNNGraphSumOfNodeDegrees(adj_list, false));
            for (size_t i = 0; i < adj_list.size(); i++) {
                for (size_t j = 0; j < adj_list[i].size(); j++) {
                    triplet_list.emplace_back(adj_list[i][j].index, i, evaluator.getValue(i, adj_list[i][j].index, adj_list[i][j].distance));
                }
            }
        }

        Eigen::SparseMatrix<ValueT> mat(adj_list.size(),adj_list.size());
        mat.setFromTriplets(triplet_list.begin(), triplet_list.end());

        return mat;
    }

    template <typename ScalarT, typename ValueT = bool>
    inline Eigen::Matrix<ValueT,Eigen::Dynamic,Eigen::Dynamic> getNNGraphDenseAdjacencyMatrix(const std::vector<std::vector<NearestNeighborSearchResult<ScalarT>>> &adj_list,
                                                                                              bool force_symmetry = false)
    {
        return getNNGraphFunctionValueDenseMatrix<ScalarT,AdjacencyEvaluator<ScalarT,ValueT>,ValueT>(adj_list, AdjacencyEvaluator<ScalarT,ValueT>(), force_symmetry);
    }

    template <typename ScalarT, typename ValueT = bool>
    inline Eigen::SparseMatrix<ValueT> getNNGraphSparseAdjacencyMatrix(const std::vector<std::vector<NearestNeighborSearchResult<ScalarT>>> &adj_list,
                                                                       bool force_symmetry = false)
    {
        return getNNGraphFunctionValueSparseMatrix<ScalarT,AdjacencyEvaluator<ScalarT,ValueT>,ValueT>(adj_list, AdjacencyEvaluator<ScalarT,ValueT>(), force_symmetry);
    }

    template <typename ScalarT, typename ValueT = ScalarT>
    inline Eigen::Matrix<ValueT,Eigen::Dynamic,Eigen::Dynamic> getNNGraphDenseDistanceMatrix(const std::vector<std::vector<NearestNeighborSearchResult<ScalarT>>> &adj_list,
                                                                                             bool force_symmetry = false)
    {
        return getNNGraphFunctionValueDenseMatrix<ScalarT,DistanceEvaluator<ScalarT,ValueT>,ValueT>(adj_list, DistanceEvaluator<ScalarT,ValueT>(), force_symmetry);
    }

    template <typename ScalarT, typename ValueT = ScalarT>
    inline Eigen::SparseMatrix<ValueT> getNNGraphSparseDistanceMatrix(const std::vector<std::vector<NearestNeighborSearchResult<ScalarT>>> &adj_list,
                                                                      bool force_symmetry = false)
    {
        return getNNGraphFunctionValueSparseMatrix<ScalarT,DistanceEvaluator<ScalarT,ValueT>,ValueT>(adj_list, DistanceEvaluator<ScalarT,ValueT>(), force_symmetry);
    }

//    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
//    class NearestNeighborGraph {
//    public:
//        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//
//        NearestNeighborGraph() {}
//
//        NearestNeighborGraph(const std::vector<std::vector<size_t>> &neighbor_indices,
//                             const std::vector<std::vector<ScalarT>> &neighbor_distances)
//                : neighbor_indices_(neighbor_indices), neighbor_distances_(neighbor_distances)
//        {}
//
//        NearestNeighborGraph(const KDTree<ScalarT,EigenDim,DistAdaptor> &points_tree,
//                             const NeighborhoodSpecification<ScalarT> &nh)
//        {
//            points_tree.search(points_tree.getPointsMatrixMap(), nh, neighbor_indices_, neighbor_distances_);
//        }
//
//        NearestNeighborGraph(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
//                             const NeighborhoodSpecification<ScalarT> &nh,
//                             size_t max_leaf_size = 10)
//        {
//            KDTree<ScalarT,EigenDim,DistAdaptor>(points, max_leaf_size).search(points, nh, neighbor_indices_, neighbor_distances_);
//        }
//
//        ~NearestNeighborGraph() {}
//
//        inline const std::vector<std::vector<size_t>>& getAdjacencyList() const { return neighbor_indices_; }
//
//        inline const std::vector<std::vector<ScalarT>>& getNeighborDistances() const { return neighbor_distances_; }
//
//        std::vector<size_t> getNodeDegrees(bool remove_self = true) const {
//            std::vector<size_t> deg(neighbor_indices_.size());
//            if (remove_self) {
//                for (size_t i = 0; i < deg.size(); i++) {
//                    deg[i] = neighbor_indices_[i].size() - 1;
//                }
//            } else {
//                for (size_t i = 0; i < deg.size(); i++) {
//                    deg[i] = neighbor_indices_[i].size();
//                }
//            }
//            return deg;
//        }
//
//        size_t getMaxNodeDegree(bool remove_self = true) const {
//            size_t max = 0;
//            for (size_t i = 0; i < neighbor_indices_.size(); i++) {
//                if (max < neighbor_indices_[i].size()) max = neighbor_indices_[i].size();
//            }
//            return (remove_self) ? max - 1 : max;
//        }
//
//        size_t getSumOfNodeDegrees(bool remove_self = true) const {
//            size_t sum = 0;
//            for (size_t i = 0; i < neighbor_indices_.size(); i++) {
//                sum += neighbor_indices_[i].size();
//            }
//            return (remove_self) ? sum - neighbor_indices_.size() : sum;
//        }
//
//        template <class PairEvaluatorT, typename ValueT = decltype(std::declval<PairEvaluatorT>().getValue((size_t)0,(size_t)0,(ScalarT)0))>
//        std::vector<std::vector<ValueT>> getFunctionValueList(const PairEvaluatorT &evaluator) const {
//            std::vector<std::vector<ValueT>> f_values(neighbor_indices_.size());
//            for (size_t i = 0; i < f_values.size(); i++) {
//                f_values[i].resize(neighbor_indices_[i].size());
//                for (size_t j = 0; j < f_values[i].size(); j++) {
//                    f_values[i][j] = evaluator.getValue(i, neighbor_indices_[i][j], neighbor_distances_[i][j]);
//                }
//            }
//            return f_values;
//        }
//
//        template <typename ValueT = bool>
//        inline Eigen::Matrix<ValueT,Eigen::Dynamic,Eigen::Dynamic> getDenseAdjacencyMatrix(bool force_symmetry = false) const {
//            return getFunctionValueDenseMatrix<AdjacencyEvaluator_<ValueT>,ValueT>(AdjacencyEvaluator_<ValueT>(), force_symmetry);
//        }
//
//        template <typename ValueT = bool>
//        inline Eigen::SparseMatrix<ValueT> getSparseAdjacencyMatrix(bool force_symmetry = false) const {
//            return getFunctionValueSparseMatrix<AdjacencyEvaluator_<ValueT>,ValueT>(AdjacencyEvaluator_<ValueT>(), force_symmetry);
//        }
//
//        template <typename ValueT = ScalarT>
//        inline Eigen::Matrix<ValueT,Eigen::Dynamic,Eigen::Dynamic> getDenseDistanceMatrix(bool force_symmetry = false) const {
//            return getFunctionValueDenseMatrix<DistanceEvaluator_<ValueT>,ValueT>(DistanceEvaluator_<ValueT>(), force_symmetry);
//        }
//
//        template <typename ValueT = ScalarT>
//        inline Eigen::SparseMatrix<ValueT> getSparseDistanceMatrix(bool force_symmetry = false) const {
//            return getFunctionValueSparseMatrix<DistanceEvaluator_<ValueT>,ValueT>(DistanceEvaluator_<ValueT>(), force_symmetry);
//        }
//
//        template <class PairEvaluatorT, typename ValueT = decltype(std::declval<PairEvaluatorT>().getValue((size_t)0,(size_t)0,(ScalarT)0))>
//        Eigen::Matrix<ValueT,Eigen::Dynamic,Eigen::Dynamic> getFunctionValueDenseMatrix(const PairEvaluatorT &evaluator, bool force_symmetry = false) const {
//            Eigen::Matrix<ValueT,Eigen::Dynamic,Eigen::Dynamic> mat(Eigen::Matrix<ValueT,Eigen::Dynamic,Eigen::Dynamic>::Zero(neighbor_indices_.size(),neighbor_indices_.size()));
//            if (force_symmetry) {
//                for (size_t i = 0; i < neighbor_indices_.size(); i++) {
//                    for (size_t j = 0; j < neighbor_indices_[i].size(); j++) {
//                        ValueT val = evaluator.getValue(i,neighbor_indices_[i][j],neighbor_distances_[i][j]);
//                        mat(neighbor_indices_[i][j],i) = val;
//                        mat(i,neighbor_indices_[i][j]) = val;
//                    }
//                }
//            } else {
//                for (size_t i = 0; i < neighbor_indices_.size(); i++) {
//                    for (size_t j = 0; j < neighbor_indices_[i].size(); j++) {
//                        mat(neighbor_indices_[i][j],i) = evaluator.getValue(i,neighbor_indices_[i][j],neighbor_distances_[i][j]);
//                    }
//                }
//            }
//            return mat;
//        }
//
//        template <class PairEvaluatorT, typename ValueT = decltype(std::declval<PairEvaluatorT>().getValue((size_t)0,(size_t)0,(ScalarT)0))>
//        Eigen::SparseMatrix<ValueT> getFunctionValueSparseMatrix(const PairEvaluatorT &evaluator, bool force_symmetry = false) const {
//            std::vector<Eigen::Triplet<ValueT>> triplet_list;
//            if (force_symmetry) {
//                triplet_list.reserve(2*getSumOfNodeDegrees());
//                for (size_t i = 0; i < neighbor_indices_.size(); i++) {
//                    for (size_t j = 0; j < neighbor_indices_[i].size(); j++) {
//                        ValueT val = evaluator.getValue(i,neighbor_indices_[i][j],neighbor_distances_[i][j]);
//                        triplet_list.emplace_back(neighbor_indices_[i][j], i, val);
//                        triplet_list.emplace_back(i, neighbor_indices_[i][j], val);
//                    }
//                }
//            } else {
//                triplet_list.reserve(getSumOfNodeDegrees());
//                for (size_t i = 0; i < neighbor_indices_.size(); i++) {
//                    for (size_t j = 0; j < neighbor_indices_[i].size(); j++) {
//                        triplet_list.emplace_back(neighbor_indices_[i][j], i, evaluator.getValue(i,neighbor_indices_[i][j],neighbor_distances_[i][j]));
//                    }
//                }
//            }
//
//            Eigen::SparseMatrix<ValueT> mat(neighbor_indices_.size(),neighbor_indices_.size());
//            mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
//
//            return mat;
//        }
//
//    private:
//        std::vector<std::vector<size_t>> neighbor_indices_;
//        std::vector<std::vector<ScalarT>> neighbor_distances_;
//
//        // Dummy function evaluators for neighboring point pairs
//        template <typename ValueT>
//        struct AdjacencyEvaluator_ {
//            inline ValueT getValue(size_t pt_ind, size_t nn_ind, ScalarT dist) const {
//                return 1;
//            }
//        };
//
//        template <typename ValueT>
//        struct DistanceEvaluator_ {
//            inline ValueT getValue(size_t pt_ind, size_t nn_ind, ScalarT dist) const {
//                return dist;
//            }
//        };
//    };
//
//    typedef NearestNeighborGraph<float,2> NearestNeighborGraph2D;
//    typedef NearestNeighborGraph<float,3> NearestNeighborGraph3D;
}
