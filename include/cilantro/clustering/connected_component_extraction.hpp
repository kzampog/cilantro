#pragma once

#include <set>
#include <cilantro/core/kd_tree.hpp>
#include <cilantro/core/common_pair_evaluators.hpp>
#include <cilantro/clustering/clustering_base.hpp>

namespace cilantro {
    template <class T>
    struct SizeLessComparator {
        inline bool operator()(const T& obj1, const T& obj2) const { return obj1.size() < obj2.size(); }
    };

    template <class T>
    struct SizeGreaterComparator {
        inline bool operator()(const T& obj1, const T& obj2) const { return obj1.size() > obj2.size(); }
    };

    // Given neighbors and seeds
    template <typename ScalarT, typename IndexT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    void extractConnectedComponents(const NeighborhoodSet<ScalarT,IndexT> &neighbors,
                                    const std::vector<IndexT> &seeds_ind,
                                    std::vector<std::vector<IndexT>> &segment_to_point_map,
                                    const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                    size_t min_segment_size = 1,
                                    size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        const size_t unassigned = std::numeric_limits<size_t>::max();
        std::vector<size_t> current_label(neighbors.size(), unassigned);

        std::vector<size_t> frontier_set;
        frontier_set.reserve(neighbors.size());

        std::vector<std::set<size_t>> seeds_to_merge_with(seeds_ind.size());
        std::vector<char> seed_active(seeds_ind.size(), 0);

#pragma omp parallel for shared (seeds_ind, current_label, seed_active, seeds_to_merge_with) private (frontier_set)
        for (size_t i = 0; i < seeds_ind.size(); i++) {
            if (current_label[seeds_ind[i]] != unassigned) continue;

            seeds_to_merge_with[i].insert(i);

            frontier_set.clear();
            frontier_set.emplace_back(seeds_ind[i]);

            current_label[seeds_ind[i]] = i;
            seed_active[i] = 1;

            while (!frontier_set.empty()) {
                const size_t curr_seed = frontier_set.back();
                frontier_set.pop_back();

                const Neighborhood<ScalarT,IndexT>& nn(neighbors[curr_seed]);
                for (size_t j = 1; j < nn.size(); j++) {
                    const size_t curr_lbl = current_label[nn[j].index];
                    if (curr_lbl == i || evaluator(curr_seed, nn[j].index, nn[j].value)) {
                        if (curr_lbl == unassigned) {
                            frontier_set.emplace_back(nn[j].index);
                            current_label[nn[j].index] = i;
                        } else {
                            if (curr_lbl != i) seeds_to_merge_with[i].insert(curr_lbl);
                        }
                    }
                }
            }
        }

        for (size_t i = 0; i < seeds_to_merge_with.size(); i++) {
            for (auto it = seeds_to_merge_with[i].begin(); it != seeds_to_merge_with[i].end(); ++it) {
                seeds_to_merge_with[*it].insert(i);
            }
        }

        std::vector<size_t> seed_repr(seeds_ind.size(), unassigned);
        size_t seed_cluster_num = 0;
        for (size_t i = 0; i < seeds_to_merge_with.size(); i++) {
            if (seed_active[i] == 0 || seed_repr[i] != unassigned) continue;

            frontier_set.clear();
            frontier_set.emplace_back(i);
            seed_repr[i] = seed_cluster_num;

            while (!frontier_set.empty()) {
                const size_t curr_seed = frontier_set.back();
                frontier_set.pop_back();
                for (auto it = seeds_to_merge_with[curr_seed].begin(); it != seeds_to_merge_with[curr_seed].end(); ++it) {
                    if (seed_active[i] == 1 && seed_repr[*it] == unassigned) {
                        frontier_set.emplace_back(*it);
                        seed_repr[*it] = seed_cluster_num;
                    }
                }
            }

            seed_cluster_num++;
        }

        std::vector<std::vector<IndexT>> segment_to_point_map_tmp(seed_cluster_num);
        for (size_t i = 0; i < current_label.size(); i++) {
            if (current_label[i] == unassigned) continue;
            const auto ind = seed_repr[current_label[i]];
            if (segment_to_point_map_tmp[ind].size() <= max_segment_size) {
                segment_to_point_map_tmp[ind].emplace_back(i);
            }
        }

        segment_to_point_map.clear();
        for (size_t i = 0; i < segment_to_point_map_tmp.size(); i++) {
            if (segment_to_point_map_tmp[i].size() >= min_segment_size && segment_to_point_map_tmp[i].size() <= max_segment_size) {
                segment_to_point_map.emplace_back(std::move(segment_to_point_map_tmp[i]));
            }
        }

        std::sort(segment_to_point_map.begin(), segment_to_point_map.end(), SizeGreaterComparator<std::vector<IndexT>>());
    }

    // Given neighbors and seeds
    template <typename ScalarT, typename IndexT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    inline std::vector<std::vector<IndexT>> extractConnectedComponents(const NeighborhoodSet<ScalarT,IndexT> &neighbors,
                                                                       const std::vector<IndexT> &seeds_ind,
                                                                       const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                                       size_t min_segment_size = 1,
                                                                       size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        std::vector<std::vector<IndexT>> segment_to_point_map;
        extractConnectedComponents<ScalarT,IndexT,PointSimilarityEvaluator>(neighbors, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
        return segment_to_point_map;
    }

    // Given neighbors, all seeds
    template <typename ScalarT, typename IndexT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    void extractConnectedComponents(const NeighborhoodSet<ScalarT,IndexT> &neighbors,
                                    std::vector<std::vector<IndexT>> &segment_to_point_map,
                                    const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                    size_t min_segment_size = 1,
                                    size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        std::vector<IndexT> seeds_ind(neighbors.size());
        for (size_t i = 0; i < seeds_ind.size(); i++) seeds_ind[i] = static_cast<IndexT>(i);
        extractConnectedComponents<ScalarT,IndexT,PointSimilarityEvaluator>(neighbors, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
    }

    // Given neighbors, all seeds
    template <typename ScalarT, typename IndexT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    std::vector<std::vector<IndexT>> extractConnectedComponents(const NeighborhoodSet<ScalarT,IndexT> &neighbors,
                                                                const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                                size_t min_segment_size = 1,
                                                                size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        std::vector<IndexT> seeds_ind(neighbors.size());
        for (size_t i = 0; i < seeds_ind.size(); i++) seeds_ind[i] = static_cast<IndexT>(i);
        std::vector<std::vector<IndexT>> segment_to_point_map;
        extractConnectedComponents<ScalarT,IndexT,PointSimilarityEvaluator>(neighbors, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
        return segment_to_point_map;
    }

    // Given search tree and seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, typename IndexT, class NeighborhoodSpecT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    void extractConnectedComponents(const KDTree<ScalarT,EigenDim,DistAdaptor,IndexT> &tree,
                                    const NeighborhoodSpecT &nh,
                                    const std::vector<IndexT> &seeds_ind,
                                    std::vector<std::vector<IndexT>> &segment_to_point_map,
                                    const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                    size_t min_segment_size = 1,
                                    size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        const ConstVectorSetMatrixMap<ScalarT,EigenDim>& points(tree.getPointsMatrixMap());

        const size_t unassigned = std::numeric_limits<size_t>::max();
        std::vector<size_t> current_label(points.cols(), unassigned);

        std::vector<size_t> frontier_set;
        frontier_set.reserve(points.cols());

        std::vector<std::set<size_t>> seeds_to_merge_with(seeds_ind.size());
        std::vector<char> seed_active(seeds_ind.size(), 0);

        Neighborhood<ScalarT,IndexT> nn;

#pragma omp parallel for shared (seeds_ind, current_label, seed_active, seeds_to_merge_with) private (nn, frontier_set)
        for (size_t i = 0; i < seeds_ind.size(); i++) {
            if (current_label[seeds_ind[i]] != unassigned) continue;

            seeds_to_merge_with[i].insert(i);

            frontier_set.clear();
            frontier_set.emplace_back(seeds_ind[i]);

            current_label[seeds_ind[i]] = i;
            seed_active[i] = 1;

            while (!frontier_set.empty()) {
                const size_t curr_seed = frontier_set.back();
                frontier_set.pop_back();

                tree.search(points.col(curr_seed), nh, nn);
                for (size_t j = 1; j < nn.size(); j++) {
                    const size_t curr_lbl = current_label[nn[j].index];
                    if (curr_lbl == i || evaluator(curr_seed, nn[j].index, nn[j].value)) {
                        if (curr_lbl == unassigned) {
                            frontier_set.emplace_back(nn[j].index);
                            current_label[nn[j].index] = i;
                        } else {
                            if (curr_lbl != i) seeds_to_merge_with[i].insert(curr_lbl);
                        }
                    }
                }
            }
        }

        for (size_t i = 0; i < seeds_to_merge_with.size(); i++) {
            for (auto it = seeds_to_merge_with[i].begin(); it != seeds_to_merge_with[i].end(); ++it) {
                seeds_to_merge_with[*it].insert(i);
            }
        }

        std::vector<size_t> seed_repr(seeds_ind.size(), unassigned);
        size_t seed_cluster_num = 0;
        for (size_t i = 0; i < seeds_to_merge_with.size(); i++) {
            if (seed_active[i] == 0 || seed_repr[i] != unassigned) continue;

            frontier_set.clear();
            frontier_set.emplace_back(i);
            seed_repr[i] = seed_cluster_num;

            while (!frontier_set.empty()) {
                const size_t curr_seed = frontier_set.back();
                frontier_set.pop_back();
                for (auto it = seeds_to_merge_with[curr_seed].begin(); it != seeds_to_merge_with[curr_seed].end(); ++it) {
                    if (seed_active[i] == 1 && seed_repr[*it] == unassigned) {
                        frontier_set.emplace_back(*it);
                        seed_repr[*it] = seed_cluster_num;
                    }
                }
            }

            seed_cluster_num++;
        }

        std::vector<std::vector<IndexT>> segment_to_point_map_tmp(seed_cluster_num);
        for (size_t i = 0; i < current_label.size(); i++) {
            if (current_label[i] == unassigned) continue;
            const auto ind = seed_repr[current_label[i]];
            if (segment_to_point_map_tmp[ind].size() <= max_segment_size) {
                segment_to_point_map_tmp[ind].emplace_back(i);
            }
        }

        segment_to_point_map.clear();
        for (size_t i = 0; i < segment_to_point_map_tmp.size(); i++) {
            if (segment_to_point_map_tmp[i].size() >= min_segment_size && segment_to_point_map_tmp[i].size() <= max_segment_size) {
                segment_to_point_map.emplace_back(std::move(segment_to_point_map_tmp[i]));
            }
        }

        std::sort(segment_to_point_map.begin(), segment_to_point_map.end(), SizeGreaterComparator<std::vector<IndexT>>());
    }

    // Given search tree and seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, typename IndexT, class NeighborhoodSpecT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    inline std::vector<std::vector<IndexT>> extractConnectedComponents(const KDTree<ScalarT,EigenDim,DistAdaptor,IndexT> &tree,
                                                                       const NeighborhoodSpecT &nh,
                                                                       const std::vector<IndexT> &seeds_ind,
                                                                       const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                                       size_t min_segment_size = 1,
                                                                       size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        std::vector<std::vector<IndexT>> segment_to_point_map;
        extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,IndexT,NeighborhoodSpecT,PointSimilarityEvaluator>(tree, nh, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
        return segment_to_point_map;
    }

    // Given search tree, all seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, typename IndexT, class NeighborhoodSpecT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    void extractConnectedComponents(const KDTree<ScalarT,EigenDim,DistAdaptor,IndexT> &tree,
                                    const NeighborhoodSpecT &nh,
                                    std::vector<std::vector<IndexT>> &segment_to_point_map,
                                    const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                    size_t min_segment_size = 1,
                                    size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        std::vector<IndexT> seeds_ind(tree.getPointsMatrixMap().cols());
        for (size_t i = 0; i < seeds_ind.size(); i++) seeds_ind[i] = static_cast<IndexT>(i);
        extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,IndexT,NeighborhoodSpecT,PointSimilarityEvaluator>(tree, nh, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
    }

    // Given search tree, all seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, typename IndexT, class NeighborhoodSpecT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    std::vector<std::vector<IndexT>> extractConnectedComponents(const KDTree<ScalarT,EigenDim,DistAdaptor,IndexT> &tree,
                                                                const NeighborhoodSpecT &nh,
                                                                const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                                size_t min_segment_size = 1,
                                                                size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        std::vector<IndexT> seeds_ind(tree.getPointsMatrixMap().cols());
        for (size_t i = 0; i < seeds_ind.size(); i++) seeds_ind[i] = static_cast<IndexT>(i);
        std::vector<std::vector<IndexT>> segment_to_point_map;
        extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,IndexT,NeighborhoodSpecT,PointSimilarityEvaluator>(tree, nh, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
        return segment_to_point_map;
    }

    // Given points and seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, typename IndexT, class NeighborhoodSpecT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    inline void extractConnectedComponents(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                           const NeighborhoodSpecT &nh,
                                           const std::vector<IndexT> &seeds_ind,
                                           std::vector<std::vector<IndexT>> &segment_to_point_map,
                                           const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                           size_t min_segment_size = 1,
                                           size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,IndexT,NeighborhoodSpecT,PointSimilarityEvaluator>(KDTree<ScalarT,EigenDim,DistAdaptor,IndexT>(points), nh, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
    }

    // Given points and seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, typename IndexT, class NeighborhoodSpecT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    inline std::vector<std::vector<IndexT>> extractConnectedComponents(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                                                       const NeighborhoodSpecT &nh,
                                                                       const std::vector<IndexT> &seeds_ind,
                                                                       const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                                       size_t min_segment_size = 1,
                                                                       size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        return extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,IndexT,NeighborhoodSpecT,PointSimilarityEvaluator>(KDTree<ScalarT,EigenDim,DistAdaptor,IndexT>(points), nh, seeds_ind, evaluator, min_segment_size, max_segment_size);
    }

    // Given points, all seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, typename IndexT, class NeighborhoodSpecT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    inline void extractConnectedComponents(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                           const NeighborhoodSpecT &nh,
                                           std::vector<std::vector<IndexT>> &segment_to_point_map,
                                           const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                           size_t min_segment_size = 1,
                                           size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,IndexT,NeighborhoodSpecT,PointSimilarityEvaluator>(KDTree<ScalarT,EigenDim,DistAdaptor,IndexT>(points), nh, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
    }

    // Given points, all seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, typename IndexT, class NeighborhoodSpecT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    inline std::vector<std::vector<IndexT>> extractConnectedComponents(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                                                       const NeighborhoodSpecT &nh,
                                                                       const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                                       size_t min_segment_size = 1,
                                                                       size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        return extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,IndexT,NeighborhoodSpecT,PointSimilarityEvaluator>(KDTree<ScalarT,EigenDim,DistAdaptor,IndexT>(points), nh, evaluator, min_segment_size, max_segment_size);
    }

    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, typename PointIndexT = size_t, typename ClusterIndexT = size_t>
    class ConnectedComponentExtraction : public ClusteringBase<ConnectedComponentExtraction<ScalarT,EigenDim,DistAdaptor>,PointIndexT,ClusterIndexT> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum { Dimension = EigenDim };

        typedef KDTree<ScalarT,EigenDim,DistAdaptor,PointIndexT> SearchTree;

        ConnectedComponentExtraction(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, size_t max_leaf_size = 10)
                : points_(points),
                  kd_tree_ptr_(new SearchTree(points, max_leaf_size)),
                  kd_tree_owned_(true)
        {}

        ConnectedComponentExtraction(const SearchTree &kd_tree)
                : points_(kd_tree.getPointsMatrixMap()),
                  kd_tree_ptr_(&kd_tree),
                  kd_tree_owned_(false)
        {}

        ~ConnectedComponentExtraction() {
            if (kd_tree_owned_) delete kd_tree_ptr_;
        }

        template <class NeighborhoodSpecT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
        inline ConnectedComponentExtraction& segment(const NeighborhoodSpecT &nh,
                                                     const std::vector<PointIndexT> &seeds_ind,
                                                     const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                     size_t min_segment_size = 1,
                                                     size_t max_segment_size = std::numeric_limits<size_t>::max())
        {
            extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,PointIndexT,NeighborhoodSpecT,PointSimilarityEvaluator>(*kd_tree_ptr_, nh, seeds_ind, this->cluster_to_point_indices_map_, evaluator, min_segment_size, max_segment_size);
            this->point_to_cluster_index_map_ = cilantro::getPointToClusterIndexMap<ClusterIndexT,PointIndexT>(this->cluster_to_point_indices_map_, points_.cols());
            return *this;
        }

        template <class NeighborhoodSpecT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
        inline ConnectedComponentExtraction& segment(const NeighborhoodSpecT &nh,
                                                     const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                     size_t min_segment_size = 1,
                                                     size_t max_segment_size = std::numeric_limits<size_t>::max())
        {
            extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,PointIndexT,NeighborhoodSpecT,PointSimilarityEvaluator>(*kd_tree_ptr_, nh, this->cluster_to_point_indices_map_, evaluator, min_segment_size, max_segment_size);
            this->point_to_cluster_index_map_ = cilantro::getPointToClusterIndexMap<ClusterIndexT,PointIndexT>(this->cluster_to_point_indices_map_, points_.cols());
            return *this;
        }

    protected:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> points_;
        const SearchTree *kd_tree_ptr_;
        bool kd_tree_owned_;
    };

    template <template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, typename PointIndexT = size_t, typename ClusterIndexT = size_t>
    using ConnectedComponentExtraction2f = ConnectedComponentExtraction<float,2,DistAdaptor,PointIndexT,ClusterIndexT>;

    template <template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, typename PointIndexT = size_t, typename ClusterIndexT = size_t>
    using ConnectedComponentExtraction2d = ConnectedComponentExtraction<double,2,DistAdaptor,PointIndexT,ClusterIndexT>;

    template <template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, typename PointIndexT = size_t, typename ClusterIndexT = size_t>
    using ConnectedComponentExtraction3f = ConnectedComponentExtraction<float,3,DistAdaptor,PointIndexT,ClusterIndexT>;

    template <template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, typename PointIndexT = size_t, typename ClusterIndexT = size_t>
    using ConnectedComponentExtraction3d = ConnectedComponentExtraction<double,3,DistAdaptor,PointIndexT,ClusterIndexT>;

    template <template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, typename PointIndexT = size_t, typename ClusterIndexT = size_t>
    using ConnectedComponentExtractionXf = ConnectedComponentExtraction<float,Eigen::Dynamic,DistAdaptor,PointIndexT,ClusterIndexT>;

    template <template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, typename PointIndexT = size_t, typename ClusterIndexT = size_t>
    using ConnectedComponentExtractionXd = ConnectedComponentExtraction<double,Eigen::Dynamic,DistAdaptor,PointIndexT,ClusterIndexT>;
}
