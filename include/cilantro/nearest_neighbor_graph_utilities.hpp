#pragma once

#include <Eigen/Sparse>
#include <cilantro/nearest_neighbors.hpp>
#include <cilantro/common_pair_evaluators.hpp>

namespace cilantro {
    template <typename T>
    std::vector<size_t> getNNGraphNodeDegrees(const std::vector<std::vector<T>> &adj_list,
                                              bool remove_self = true)
    {
        std::vector<size_t> deg(adj_list.size());
        if (remove_self) {
#pragma omp parallel for
            for (size_t i = 0; i < deg.size(); i++) {
                deg[i] = adj_list[i].size() - 1;
            }
        } else {
#pragma omp parallel for
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
#pragma omp parallel for reduction (max: max)
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
#pragma omp parallel for reduction (+: sum)
        for (size_t i = 0; i < adj_list.size(); i++) {
            sum += adj_list[i].size();
        }
        return (remove_self) ? sum - adj_list.size() : sum;
    }

    template <typename ScalarT, class PairEvaluatorT, typename ValueT = typename PairEvaluatorT::OutputScalar>
    std::vector<std::vector<ValueT>> getNNGraphFunctionValueList(const std::vector<std::vector<Neighbor<ScalarT>>> &adj_list,
                                                                 const PairEvaluatorT &evaluator = PairEvaluatorT())
    {
        std::vector<std::vector<ValueT>> f_values(adj_list.size());
#pragma omp parallel for shared (f_values)
        for (size_t i = 0; i < f_values.size(); i++) {
            f_values[i].resize(adj_list[i].size());
            for (size_t j = 0; j < f_values[i].size(); j++) {
                f_values[i][j] = evaluator(i, adj_list[i][j].index, adj_list[i][j].value);
            }
        }
        return f_values;
    }

    template <typename ScalarT, class PairEvaluatorT, typename ValueT = typename PairEvaluatorT::OutputScalar>
    Eigen::Matrix<ValueT,Eigen::Dynamic,Eigen::Dynamic> getNNGraphFunctionValueDenseMatrix(const std::vector<std::vector<Neighbor<ScalarT>>> &adj_list,
                                                                                           const PairEvaluatorT &evaluator = PairEvaluatorT(),
                                                                                           bool force_symmetry = false)
    {
        Eigen::Matrix<ValueT,Eigen::Dynamic,Eigen::Dynamic> mat(Eigen::Matrix<ValueT,Eigen::Dynamic,Eigen::Dynamic>::Zero(adj_list.size(), adj_list.size()));
        if (force_symmetry) {
            for (size_t i = 0; i < adj_list.size(); i++) {
                for (size_t j = 0; j < adj_list[i].size(); j++) {
                    ValueT val = evaluator(i, adj_list[i][j].index, adj_list[i][j].value);
                    mat(adj_list[i][j].index, i) = val;
                    mat(i, adj_list[i][j].index) = val;
                }
            }
        } else {
            for (size_t i = 0; i < adj_list.size(); i++) {
                for (size_t j = 0; j < adj_list[i].size(); j++) {
                    mat(adj_list[i][j].index, i) = evaluator(i, adj_list[i][j].index, adj_list[i][j].value);
                }
            }
        }
        return mat;
    }

    template <typename ScalarT, class PairEvaluatorT, typename ValueT = typename PairEvaluatorT::OutputScalar>
    Eigen::SparseMatrix<ValueT> getNNGraphFunctionValueSparseMatrix(const std::vector<std::vector<Neighbor<ScalarT>>> &adj_list,
                                                                    const PairEvaluatorT &evaluator = PairEvaluatorT(),
                                                                    bool force_symmetry = false)
    {
        std::vector<Eigen::Triplet<ValueT>> triplet_list;
        if (force_symmetry) {
            triplet_list.reserve(2*getNNGraphSumOfNodeDegrees(adj_list, false));
            for (size_t i = 0; i < adj_list.size(); i++) {
                for (size_t j = 0; j < adj_list[i].size(); j++) {
                    ValueT val = evaluator(i, adj_list[i][j].index, adj_list[i][j].value);
                    triplet_list.emplace_back(adj_list[i][j].index, i, val);
                    triplet_list.emplace_back(i, adj_list[i][j].index, val);
                }
            }
        } else {
            triplet_list.reserve(getNNGraphSumOfNodeDegrees(adj_list, false));
            for (size_t i = 0; i < adj_list.size(); i++) {
                for (size_t j = 0; j < adj_list[i].size(); j++) {
                    triplet_list.emplace_back(adj_list[i][j].index, i, evaluator(i, adj_list[i][j].index, adj_list[i][j].value));
                }
            }
        }

        Eigen::SparseMatrix<ValueT> mat(adj_list.size(), adj_list.size());
        mat.setFromTriplets(triplet_list.begin(), triplet_list.end());

        return mat;
    }

    template <typename ScalarT, typename ValueT = bool>
    inline Eigen::Matrix<ValueT,Eigen::Dynamic,Eigen::Dynamic> getNNGraphDenseAdjacencyMatrix(const std::vector<std::vector<Neighbor<ScalarT>>> &adj_list,
                                                                                              bool force_symmetry = false)
    {
        return getNNGraphFunctionValueDenseMatrix<ScalarT,AdjacencyEvaluator<ScalarT,ValueT>,ValueT>(adj_list, AdjacencyEvaluator<ScalarT,ValueT>(), force_symmetry);
    }

    template <typename ScalarT, typename ValueT = bool>
    inline Eigen::SparseMatrix<ValueT> getNNGraphSparseAdjacencyMatrix(const std::vector<std::vector<Neighbor<ScalarT>>> &adj_list,
                                                                       bool force_symmetry = false)
    {
        return getNNGraphFunctionValueSparseMatrix<ScalarT,AdjacencyEvaluator<ScalarT,ValueT>,ValueT>(adj_list, AdjacencyEvaluator<ScalarT,ValueT>(), force_symmetry);
    }

    template <typename ScalarT, typename ValueT = ScalarT>
    inline Eigen::Matrix<ValueT,Eigen::Dynamic,Eigen::Dynamic> getNNGraphDenseDistanceMatrix(const std::vector<std::vector<Neighbor<ScalarT>>> &adj_list,
                                                                                             bool force_symmetry = false)
    {
        return getNNGraphFunctionValueDenseMatrix<ScalarT,DistanceEvaluator<ScalarT,ValueT>,ValueT>(adj_list, DistanceEvaluator<ScalarT,ValueT>(), force_symmetry);
    }

    template <typename ScalarT, typename ValueT = ScalarT>
    inline Eigen::SparseMatrix<ValueT> getNNGraphSparseDistanceMatrix(const std::vector<std::vector<Neighbor<ScalarT>>> &adj_list,
                                                                      bool force_symmetry = false)
    {
        return getNNGraphFunctionValueSparseMatrix<ScalarT,DistanceEvaluator<ScalarT,ValueT>,ValueT>(adj_list, DistanceEvaluator<ScalarT,ValueT>(), force_symmetry);
    }
}
