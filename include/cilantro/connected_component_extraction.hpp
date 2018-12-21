#pragma once

#include <set>
#include <cilantro/kd_tree.hpp>
#include <cilantro/common_pair_evaluators.hpp>
#include <cilantro/clustering_base.hpp>

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
    template <typename ScalarT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    void extractConnectedComponents(const NeighborhoodSet<ScalarT> &neighbors,
                                    const std::vector<size_t> &seeds_ind,
                                    std::vector<std::vector<size_t>> &segment_to_point_map,
                                    const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                    size_t min_segment_size = 1,
                                    size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        const size_t unassigned = std::numeric_limits<size_t>::max();
        std::vector<size_t> current_label(neighbors.size(), unassigned);

        std::vector<size_t> frontier_set;
        frontier_set.reserve(neighbors.size());

//        std::vector<std::set<size_t>> ind_per_seed(seeds_ind.size());
        std::vector<std::vector<size_t>> ind_per_seed(seeds_ind.size());
        std::vector<std::set<size_t>> seeds_to_merge_with(seeds_ind.size());

#pragma omp parallel for shared (seeds_ind, current_label, ind_per_seed, seeds_to_merge_with) private (frontier_set)
        for (size_t i = 0; i < seeds_ind.size(); i++) {
            if (current_label[seeds_ind[i]] != unassigned) continue;

            seeds_to_merge_with[i].insert(i);

            frontier_set.clear();
            frontier_set.emplace_back(seeds_ind[i]);

            current_label[seeds_ind[i]] = i;

            while (!frontier_set.empty()) {
                const size_t curr_seed = frontier_set.back();
                frontier_set.pop_back();

//                ind_per_seed[i].insert(curr_seed);
                ind_per_seed[i].emplace_back(curr_seed);

                const Neighborhood<ScalarT>& nn(neighbors[curr_seed]);
                for (size_t j = 1; j < nn.size(); j++) {
                    const size_t& curr_lbl = current_label[nn[j].index];
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
                if (*it > i) seeds_to_merge_with[*it].insert(i);
            }
        }

        segment_to_point_map.clear();
        for (size_t i = seeds_to_merge_with.size() - 1; i != static_cast<size_t>(-1); i--) {
            if (seeds_to_merge_with[i].empty()) continue;
            size_t min_seed_ind = *seeds_to_merge_with[i].begin();
            if (min_seed_ind < i) {
                for (auto it = seeds_to_merge_with[i].begin(); it != seeds_to_merge_with[i].end(); ++it) {
                    if (*it < i) seeds_to_merge_with[*it].insert(seeds_to_merge_with[i].begin(), seeds_to_merge_with[i].end());
                }
//                seeds_to_merge_with[i].clear();
            } else {
                std::set<size_t> curr_cc_ind;
                for (auto it = seeds_to_merge_with[i].begin(); it != seeds_to_merge_with[i].end(); ++it) {
                    curr_cc_ind.insert(ind_per_seed[*it].begin(), ind_per_seed[*it].end());
                }
                if (curr_cc_ind.size() >= min_segment_size && curr_cc_ind.size() <= max_segment_size) {
                    segment_to_point_map.emplace_back(curr_cc_ind.begin(), curr_cc_ind.end());
                }
            }
        }

        std::sort(segment_to_point_map.begin(), segment_to_point_map.end(), SizeGreaterComparator<std::vector<size_t>>());
    }

    // Given neighbors and seeds
    template <typename ScalarT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    inline std::vector<std::vector<size_t>> extractConnectedComponents(const NeighborhoodSet<ScalarT> &neighbors,
                                                                       const std::vector<size_t> &seeds_ind,
                                                                       const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                                       size_t min_segment_size = 1,
                                                                       size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        std::vector<std::vector<size_t>> segment_to_point_map;
        extractConnectedComponents<ScalarT,PointSimilarityEvaluator>(neighbors, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
        return segment_to_point_map;
    }

    // Given neighbors, all seeds
    template <typename ScalarT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    void extractConnectedComponents(const NeighborhoodSet<ScalarT> &neighbors,
                                    std::vector<std::vector<size_t>> &segment_to_point_map,
                                    const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                    size_t min_segment_size = 1,
                                    size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        std::vector<size_t> seeds_ind(neighbors.size());
        for (size_t i = 0; i < seeds_ind.size(); i++) seeds_ind[i] = i;
        extractConnectedComponents<ScalarT,PointSimilarityEvaluator>(neighbors, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
    }

    // Given neighbors, all seeds
    template <typename ScalarT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    std::vector<std::vector<size_t>> extractConnectedComponents(const NeighborhoodSet<ScalarT> &neighbors,
                                                                const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                                size_t min_segment_size = 1,
                                                                size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        std::vector<size_t> seeds_ind(neighbors.size());
        for (size_t i = 0; i < seeds_ind.size(); i++) seeds_ind[i] = i;
        std::vector<std::vector<size_t>> segment_to_point_map;
        extractConnectedComponents<ScalarT,PointSimilarityEvaluator>(neighbors, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
        return segment_to_point_map;
    }

    // Given search tree and seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, class NeighborhoodSpecT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    void extractConnectedComponents(const KDTree<ScalarT,EigenDim,DistAdaptor> &tree,
                                    const NeighborhoodSpecT &nh,
                                    const std::vector<size_t> &seeds_ind,
                                    std::vector<std::vector<size_t>> &segment_to_point_map,
                                    const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                    size_t min_segment_size = 1,
                                    size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        const ConstVectorSetMatrixMap<ScalarT,EigenDim>& points(tree.getPointsMatrixMap());

        const size_t unassigned = std::numeric_limits<size_t>::max();
        std::vector<size_t> current_label(points.cols(), unassigned);

        std::vector<size_t> frontier_set;
        frontier_set.reserve(points.cols());

//        std::vector<std::set<size_t>> ind_per_seed(seeds_ind.size());
        std::vector<std::vector<size_t>> ind_per_seed(seeds_ind.size());
        std::vector<std::set<size_t>> seeds_to_merge_with(seeds_ind.size());

        Neighborhood<ScalarT> nn;

#pragma omp parallel for shared (seeds_ind, current_label, ind_per_seed, seeds_to_merge_with) private (nn, frontier_set)
        for (size_t i = 0; i < seeds_ind.size(); i++) {
            if (current_label[seeds_ind[i]] != unassigned) continue;

            seeds_to_merge_with[i].insert(i);

            frontier_set.clear();
            frontier_set.emplace_back(seeds_ind[i]);

            current_label[seeds_ind[i]] = i;

            while (!frontier_set.empty()) {
                const size_t curr_seed = frontier_set.back();
                frontier_set.pop_back();

//                ind_per_seed[i].insert(curr_seed);
                ind_per_seed[i].emplace_back(curr_seed);

                tree.template search<NeighborhoodSpecT>(points.col(curr_seed), nh, nn);
                for (size_t j = 1; j < nn.size(); j++) {
                    const size_t& curr_lbl = current_label[nn[j].index];
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
                if (*it > i) seeds_to_merge_with[*it].insert(i);
            }
        }

        segment_to_point_map.clear();
        for (size_t i = seeds_to_merge_with.size() - 1; i != static_cast<size_t>(-1); i--) {
            if (seeds_to_merge_with[i].empty()) continue;
            size_t min_seed_ind = *seeds_to_merge_with[i].begin();
            if (min_seed_ind < i) {
                for (auto it = seeds_to_merge_with[i].begin(); it != seeds_to_merge_with[i].end(); ++it) {
                    if (*it < i) seeds_to_merge_with[*it].insert(seeds_to_merge_with[i].begin(), seeds_to_merge_with[i].end());
                }
//                seeds_to_merge_with[i].clear();
            } else {
                std::set<size_t> curr_cc_ind;
                for (auto it = seeds_to_merge_with[i].begin(); it != seeds_to_merge_with[i].end(); ++it) {
                    curr_cc_ind.insert(ind_per_seed[*it].begin(), ind_per_seed[*it].end());
                }
                if (curr_cc_ind.size() >= min_segment_size && curr_cc_ind.size() <= max_segment_size) {
                    segment_to_point_map.emplace_back(curr_cc_ind.begin(), curr_cc_ind.end());
                }
            }
        }

        std::sort(segment_to_point_map.begin(), segment_to_point_map.end(), SizeGreaterComparator<std::vector<size_t>>());
    }

    // Given search tree and seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, class NeighborhoodSpecT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    inline std::vector<std::vector<size_t>> extractConnectedComponents(const KDTree<ScalarT,EigenDim,DistAdaptor> &tree,
                                                                       const NeighborhoodSpecT &nh,
                                                                       const std::vector<size_t> &seeds_ind,
                                                                       const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                                       size_t min_segment_size = 1,
                                                                       size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        std::vector<std::vector<size_t>> segment_to_point_map;
        extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,NeighborhoodSpecT,PointSimilarityEvaluator>(tree, nh, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
        return segment_to_point_map;
    }

    // Given search tree, all seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, class NeighborhoodSpecT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    void extractConnectedComponents(const KDTree<ScalarT,EigenDim,DistAdaptor> &tree,
                                    const NeighborhoodSpecT &nh,
                                    std::vector<std::vector<size_t>> &segment_to_point_map,
                                    const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                    size_t min_segment_size = 1,
                                    size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        std::vector<size_t> seeds_ind(tree.getPointsMatrixMap().cols());
        for (size_t i = 0; i < seeds_ind.size(); i++) seeds_ind[i] = i;
        extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,NeighborhoodSpecT,PointSimilarityEvaluator>(tree, nh, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
    }

    // Given search tree, all seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, class NeighborhoodSpecT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    std::vector<std::vector<size_t>> extractConnectedComponents(const KDTree<ScalarT,EigenDim,DistAdaptor> &tree,
                                                                const NeighborhoodSpecT &nh,
                                                                const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                                size_t min_segment_size = 1,
                                                                size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        std::vector<size_t> seeds_ind(tree.getPointsMatrixMap().cols());
        for (size_t i = 0; i < seeds_ind.size(); i++) seeds_ind[i] = i;
        std::vector<std::vector<size_t>> segment_to_point_map;
        extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,NeighborhoodSpecT,PointSimilarityEvaluator>(tree, nh, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
        return segment_to_point_map;
    }

    // Given points and seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, class NeighborhoodSpecT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    inline void extractConnectedComponents(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                           const NeighborhoodSpecT &nh,
                                           const std::vector<size_t> &seeds_ind,
                                           std::vector<std::vector<size_t>> &segment_to_point_map,
                                           const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                           size_t min_segment_size = 1,
                                           size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,NeighborhoodSpecT,PointSimilarityEvaluator>(KDTree<ScalarT,EigenDim,DistAdaptor>(points), nh, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
    }

    // Given points and seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, class NeighborhoodSpecT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    inline std::vector<std::vector<size_t>> extractConnectedComponents(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                                                       const NeighborhoodSpecT &nh,
                                                                       const std::vector<size_t> &seeds_ind,
                                                                       const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                                       size_t min_segment_size = 1,
                                                                       size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        return extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,NeighborhoodSpecT,PointSimilarityEvaluator>(KDTree<ScalarT,EigenDim,DistAdaptor>(points), nh, seeds_ind, evaluator, min_segment_size, max_segment_size);
    }

    // Given points, all seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, class NeighborhoodSpecT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    inline void extractConnectedComponents(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                           const NeighborhoodSpecT &nh,
                                           std::vector<std::vector<size_t>> &segment_to_point_map,
                                           const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                           size_t min_segment_size = 1,
                                           size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,NeighborhoodSpecT,PointSimilarityEvaluator>(KDTree<ScalarT,EigenDim,DistAdaptor>(points), nh, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
    }

    // Given points, all seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, class NeighborhoodSpecT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    inline std::vector<std::vector<size_t>> extractConnectedComponents(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                                                       const NeighborhoodSpecT &nh,
                                                                       const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                                       size_t min_segment_size = 1,
                                                                       size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        return extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,NeighborhoodSpecT,PointSimilarityEvaluator>(KDTree<ScalarT,EigenDim,DistAdaptor>(points), nh, evaluator, min_segment_size, max_segment_size);
    }

    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    class ConnectedComponentExtraction : public ClusteringBase<ConnectedComponentExtraction<ScalarT,EigenDim,DistAdaptor>> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum { Dimension = EigenDim };

        typedef KDTree<ScalarT,EigenDim,DistAdaptor> SearchTree;

        ConnectedComponentExtraction(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, size_t max_leaf_size = 10)
                : points_(points),
                  kd_tree_ptr_(new KDTree<ScalarT,EigenDim,DistAdaptor>(points, max_leaf_size)),
                  kd_tree_owned_(true)
        {}

        ConnectedComponentExtraction(const KDTree<ScalarT,EigenDim,DistAdaptor> &kd_tree)
                : points_(kd_tree.getPointsMatrixMap()),
                  kd_tree_ptr_(&kd_tree),
                  kd_tree_owned_(false)
        {}

        ~ConnectedComponentExtraction() {
            if (kd_tree_owned_) delete kd_tree_ptr_;
        }

        template <class NeighborhoodSpecT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
        inline ConnectedComponentExtraction& segment(const NeighborhoodSpecT &nh,
                                                     const std::vector<size_t> &seeds_ind,
                                                     const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                     size_t min_segment_size = 1,
                                                     size_t max_segment_size = std::numeric_limits<size_t>::max())
        {
            extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,NeighborhoodSpecT,PointSimilarityEvaluator>(*kd_tree_ptr_, nh, seeds_ind, this->clusterToPointIndicesMap, evaluator, min_segment_size, max_segment_size);
            this->pointToClusterIndexMap = cilantro::getPointToClusterIndexMap(this->clusterToPointIndicesMap, points_.cols());
            return *this;
        }

        template <class NeighborhoodSpecT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
        inline ConnectedComponentExtraction& segment(const NeighborhoodSpecT &nh,
                                                     const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                     size_t min_segment_size = 1,
                                                     size_t max_segment_size = std::numeric_limits<size_t>::max())
        {
            extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,NeighborhoodSpecT,PointSimilarityEvaluator>(*kd_tree_ptr_, nh, this->clusterToPointIndicesMap, evaluator, min_segment_size, max_segment_size);
            this->pointToClusterIndexMap = cilantro::getPointToClusterIndexMap(this->clusterToPointIndicesMap, points_.cols());
            return *this;
        }

    protected:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> points_;
        const KDTree<ScalarT,EigenDim,DistAdaptor> *kd_tree_ptr_;
        bool kd_tree_owned_;
    };

    template <template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    using ConnectedComponentExtraction2f = ConnectedComponentExtraction<float,2,DistAdaptor>;

    template <template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    using ConnectedComponentExtraction2d = ConnectedComponentExtraction<double,2,DistAdaptor>;

    template <template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    using ConnectedComponentExtraction3f = ConnectedComponentExtraction<float,3,DistAdaptor>;

    template <template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    using ConnectedComponentExtraction3d = ConnectedComponentExtraction<double,3,DistAdaptor>;

    template <template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    using ConnectedComponentExtractionXf = ConnectedComponentExtraction<float,Eigen::Dynamic,DistAdaptor>;

    template <template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    using ConnectedComponentExtractionXd = ConnectedComponentExtraction<double,Eigen::Dynamic,DistAdaptor>;
}
