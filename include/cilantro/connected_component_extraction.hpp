#pragma once

#include <cilantro/kd_tree.hpp>
#include <cilantro/common_pair_evaluators.hpp>
#include <set>

namespace cilantro {
    template <class T>
    struct SizeLessComparator {
        inline bool operator()(const T& obj1, const T& obj2) const { return obj1.size() < obj2.size(); }
    };

    template <class T>
    struct SizeGreaterComparator {
        inline bool operator()(const T& obj1, const T& obj2) const { return obj1.size() > obj2.size(); }
    };

    std::vector<size_t> getPointToSegmentIndexMap(const std::vector<std::vector<size_t>> &segment_to_point,
                                                  size_t num_points)
    {
        std::vector<size_t> point_to_segment(num_points, segment_to_point.size());
        for (size_t i = 0; i < segment_to_point.size(); i++) {
            for (size_t j = 0; j < segment_to_point[i].size(); j++) {
                point_to_segment[segment_to_point[i][j]] = i;
            }
        }
        return point_to_segment;
    }

    std::vector<std::vector<size_t>> getSegmentToPointIndicesMap(const std::vector<size_t> &point_to_segment,
                                                               size_t num_segments)
    {
        std::vector<std::vector<size_t>> segment_to_point(num_segments);
        for (size_t i = 0; i < point_to_segment.size(); i++) {
            segment_to_point[point_to_segment[i]].emplace_back(i);
        }
        return segment_to_point;
    }

    std::vector<size_t> getUnlabeledPointIndices(const std::vector<std::vector<size_t>> &segment_to_point_map,
                                                 const std::vector<size_t> &point_to_segment_map)
    {
        const size_t no_label = segment_to_point_map.size();
        std::vector<size_t> res;
        res.reserve(point_to_segment_map.size());
        for (size_t i = 0; i < point_to_segment_map.size(); i++) {
            if (point_to_segment_map[i] == no_label) res.emplace_back(i);
        }
        return res;
    }

    // Given neighbors and seeds
    template <typename ScalarT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    void extractConnectedComponents(const std::vector<NeighborSet<ScalarT>> &neighbors,
                                    const std::vector<size_t> &seeds_ind,
                                    std::vector<std::vector<size_t>> &segment_to_point_map,
                                    const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                    size_t min_segment_size = 0,
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

                const NeighborSet<ScalarT>& nn(neighbors[curr_seed]);
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
    inline std::vector<std::vector<size_t>> extractConnectedComponents(const std::vector<NeighborSet<ScalarT>> &neighbors,
                                                                       const std::vector<size_t> &seeds_ind,
                                                                       const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                                       size_t min_segment_size = 0,
                                                                       size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        std::vector<std::vector<size_t>> segment_to_point_map;
        extractConnectedComponents<ScalarT,PointSimilarityEvaluator>(neighbors, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
        return segment_to_point_map;
    }

    // Given neighbors, all seeds
    template <typename ScalarT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    void extractConnectedComponents(const std::vector<NeighborSet<ScalarT>> &neighbors,
                                    std::vector<std::vector<size_t>> &segment_to_point_map,
                                    const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                    size_t min_segment_size = 0,
                                    size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        std::vector<size_t> seeds_ind(neighbors.size());
        for (size_t i = 0; i < seeds_ind.size(); i++) seeds_ind[i] = i;
        extractConnectedComponents<ScalarT,PointSimilarityEvaluator>(neighbors, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
    }

    // Given neighbors, all seeds
    template <typename ScalarT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    std::vector<std::vector<size_t>> extractConnectedComponents(const std::vector<NeighborSet<ScalarT>> &neighbors,
                                                                const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                                size_t min_segment_size = 0,
                                                                size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        std::vector<size_t> seeds_ind(neighbors.size());
        for (size_t i = 0; i < seeds_ind.size(); i++) seeds_ind[i] = i;
        std::vector<std::vector<size_t>> segment_to_point_map;
        extractConnectedComponents<ScalarT,PointSimilarityEvaluator>(neighbors, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
        return segment_to_point_map;
    }

    // Given search tree and seeds (NeighborhoodType given as template parameter)
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, NeighborhoodType NT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    void extractConnectedComponents(const KDTree<ScalarT,EigenDim,DistAdaptor> &tree,
                                    const NeighborhoodSpecification<ScalarT> &nh,
                                    const std::vector<size_t> &seeds_ind,
                                    std::vector<std::vector<size_t>> &segment_to_point_map,
                                    const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                    size_t min_segment_size = 0,
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

        NeighborSet<ScalarT> nn;

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

                tree.template search<NT>(points.col(curr_seed), nh, nn);
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

    // Given search tree and seeds (NeighborhoodType given as template parameter)
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, NeighborhoodType NT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    inline std::vector<std::vector<size_t>> extractConnectedComponents(const KDTree<ScalarT,EigenDim,DistAdaptor> &tree,
                                                                       const NeighborhoodSpecification<ScalarT> &nh,
                                                                       const std::vector<size_t> &seeds_ind,
                                                                       const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                                       size_t min_segment_size = 0,
                                                                       size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        std::vector<std::vector<size_t>> segment_to_point_map;
        extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,NT,PointSimilarityEvaluator>(tree, nh, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
        return segment_to_point_map;
    }

    // Given search tree, all seeds (NeighborhoodType given as template parameter)
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, NeighborhoodType NT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    void extractConnectedComponents(const KDTree<ScalarT,EigenDim,DistAdaptor> &tree,
                                    const NeighborhoodSpecification<ScalarT> &nh,
                                    std::vector<std::vector<size_t>> &segment_to_point_map,
                                    const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                    size_t min_segment_size = 0,
                                    size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        std::vector<size_t> seeds_ind(tree.getPointsMatrixMap().cols());
        for (size_t i = 0; i < seeds_ind.size(); i++) seeds_ind[i] = i;
        extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,NT,PointSimilarityEvaluator>(tree, nh, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
    }

    // Given search tree, all seeds (NeighborhoodType given as template parameter)
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, NeighborhoodType NT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    std::vector<std::vector<size_t>> extractConnectedComponents(const KDTree<ScalarT,EigenDim,DistAdaptor> &tree,
                                                                const NeighborhoodSpecification<ScalarT> &nh,
                                                                const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                                size_t min_segment_size = 0,
                                                                size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        std::vector<size_t> seeds_ind(tree.getPointsMatrixMap().cols());
        for (size_t i = 0; i < seeds_ind.size(); i++) seeds_ind[i] = i;
        std::vector<std::vector<size_t>> segment_to_point_map;
        extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,NT,PointSimilarityEvaluator>(tree, nh, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
        return segment_to_point_map;
    }

    // Given search tree and seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    void extractConnectedComponents(const KDTree<ScalarT,EigenDim,DistAdaptor> &tree,
                                    const NeighborhoodSpecification<ScalarT> &nh,
                                    const std::vector<size_t> &seeds_ind,
                                    std::vector<std::vector<size_t>> &segment_to_point_map,
                                    const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                    size_t min_segment_size = 0,
                                    size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        switch (nh.type) {
            case NeighborhoodType::KNN:
                extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,NeighborhoodType::KNN,PointSimilarityEvaluator>(tree, nh, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
                break;
            case NeighborhoodType::RADIUS:
                extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,NeighborhoodType::RADIUS,PointSimilarityEvaluator>(tree, nh, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
                break;
            case NeighborhoodType::KNN_IN_RADIUS:
                extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,NeighborhoodType::KNN_IN_RADIUS,PointSimilarityEvaluator>(tree, nh, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
                break;
        }
    }

    // Given search tree and seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    inline std::vector<std::vector<size_t>> extractConnectedComponents(const KDTree<ScalarT,EigenDim,DistAdaptor> &tree,
                                                                       const NeighborhoodSpecification<ScalarT> &nh,
                                                                       const std::vector<size_t> &seeds_ind,
                                                                       const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                                       size_t min_segment_size = 0,
                                                                       size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        std::vector<std::vector<size_t>> segment_to_point_map;
        extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,PointSimilarityEvaluator>(tree, nh, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
        return segment_to_point_map;
    }

    // Given search tree, all seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    void extractConnectedComponents(const KDTree<ScalarT,EigenDim,DistAdaptor> &tree,
                                    const NeighborhoodSpecification<ScalarT> &nh,
                                    std::vector<std::vector<size_t>> &segment_to_point_map,
                                    const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                    size_t min_segment_size = 0,
                                    size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        std::vector<size_t> seeds_ind(tree.getPointsMatrixMap().cols());
        for (size_t i = 0; i < seeds_ind.size(); i++) seeds_ind[i] = i;
        extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,PointSimilarityEvaluator>(tree, nh, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
    }

    // Given search tree, all seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    std::vector<std::vector<size_t>> extractConnectedComponents(const KDTree<ScalarT,EigenDim,DistAdaptor> &tree,
                                                                const NeighborhoodSpecification<ScalarT> &nh,
                                                                const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                                size_t min_segment_size = 0,
                                                                size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        std::vector<size_t> seeds_ind(tree.getPointsMatrixMap().cols());
        for (size_t i = 0; i < seeds_ind.size(); i++) seeds_ind[i] = i;
        std::vector<std::vector<size_t>> segment_to_point_map;
        extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,PointSimilarityEvaluator>(tree, nh, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
        return segment_to_point_map;
    }

    // Given points and seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    inline void extractConnectedComponents(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                           const NeighborhoodSpecification<ScalarT> &nh,
                                           const std::vector<size_t> &seeds_ind,
                                           std::vector<std::vector<size_t>> &segment_to_point_map,
                                           const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                           size_t min_segment_size = 0,
                                           size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,PointSimilarityEvaluator>(KDTree<ScalarT,EigenDim,DistAdaptor>(points), nh, seeds_ind, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
    }

    // Given points and seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    inline std::vector<std::vector<size_t>> extractConnectedComponents(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                                                       const NeighborhoodSpecification<ScalarT> &nh,
                                                                       const std::vector<size_t> &seeds_ind,
                                                                       const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                                       size_t min_segment_size = 0,
                                                                       size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        return extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,PointSimilarityEvaluator>(KDTree<ScalarT,EigenDim,DistAdaptor>(points), nh, seeds_ind, evaluator, min_segment_size, max_segment_size);
    }

    // Given points, all seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    inline void extractConnectedComponents(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                           const NeighborhoodSpecification<ScalarT> &nh,
                                           std::vector<std::vector<size_t>> &segment_to_point_map,
                                           const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                           size_t min_segment_size = 0,
                                           size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,PointSimilarityEvaluator>(KDTree<ScalarT,EigenDim,DistAdaptor>(points), nh, segment_to_point_map, evaluator, min_segment_size, max_segment_size);
    }

    // Given points, all seeds
    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
    inline std::vector<std::vector<size_t>> extractConnectedComponents(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                                                       const NeighborhoodSpecification<ScalarT> &nh,
                                                                       const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                                       size_t min_segment_size = 0,
                                                                       size_t max_segment_size = std::numeric_limits<size_t>::max())
    {
        return extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,PointSimilarityEvaluator>(KDTree<ScalarT,EigenDim,DistAdaptor>(points), nh, evaluator, min_segment_size, max_segment_size);
    }

    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    class ConnectedComponentExtraction {
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

        template <class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
        inline ConnectedComponentExtraction& segment(const std::vector<size_t> &seeds_ind,
                                                     const NeighborhoodSpecification<ScalarT> &nh,
                                                     const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                     size_t min_segment_size = 0,
                                                     size_t max_segment_size = std::numeric_limits<size_t>::max())
        {
            extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,PointSimilarityEvaluator>(*kd_tree_ptr_, nh, seeds_ind, segment_to_point_map_, evaluator, min_segment_size, max_segment_size);
            point_to_segment_map_ = cilantro::getPointToSegmentIndexMap(segment_to_point_map_, points_.cols());
            return *this;
        }

        template <class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
        inline ConnectedComponentExtraction& segment(const NeighborhoodSpecification<ScalarT> &nh,
                                                     const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                     size_t min_segment_size = 0,
                                                     size_t max_segment_size = std::numeric_limits<size_t>::max())
        {
            extractConnectedComponents<ScalarT,EigenDim,DistAdaptor,PointSimilarityEvaluator>(*kd_tree_ptr_, nh, segment_to_point_map_, evaluator, min_segment_size, max_segment_size);
            point_to_segment_map_ = cilantro::getPointToSegmentIndexMap(segment_to_point_map_, points_.cols());
            return *this;
        }

        inline const std::vector<std::vector<size_t>>& getSegmentToPointIndicesMap() const { return segment_to_point_map_; }

        inline const std::vector<size_t>& getPointToSegmentIndexMap() const { return point_to_segment_map_; }

        inline std::vector<size_t> getUnlabeledPointIndices() const {
            return getUnlabeledPointIndices(segment_to_point_map_, point_to_segment_map_);
        }

        inline size_t getNumberOfExtractedComponents() const { return segment_to_point_map_.size(); }

    protected:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> points_;
        const KDTree<ScalarT,EigenDim,DistAdaptor> *kd_tree_ptr_;
        bool kd_tree_owned_;
        std::vector<std::vector<size_t>> segment_to_point_map_;
        std::vector<size_t> point_to_segment_map_;
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
