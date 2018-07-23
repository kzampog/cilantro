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

    class ConnectedComponentSegmentation {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Given neighbors and seeds
        template <typename ScalarT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
        ConnectedComponentSegmentation& segment(const std::vector<NeighborSet<ScalarT>> &neighbors,
                                                const std::vector<size_t> &seeds_ind,
                                                const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                size_t min_segment_size = 0,
                                                size_t max_segment_size = std::numeric_limits<size_t>::max())
        {
            segment_given_neighbors_<ScalarT,PointSimilarityEvaluator>(neighbors, seeds_ind, evaluator, min_segment_size, max_segment_size);
            return *this;
        }

        // Given neighbors, all seeds
        template <typename ScalarT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
        ConnectedComponentSegmentation& segment(const std::vector<NeighborSet<ScalarT>> &neighbors,
                                                const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                size_t min_segment_size = 0,
                                                size_t max_segment_size = std::numeric_limits<size_t>::max())
        {
            std::vector<size_t> seeds_ind(neighbors.size());
            for (size_t i = 0; i < seeds_ind.size(); i++) seeds_ind[i] = i;
            segment_given_neighbors_<ScalarT,PointSimilarityEvaluator>(neighbors, seeds_ind, evaluator, min_segment_size, max_segment_size);
            return *this;
        }

        // Given search tree and seeds
        template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
        ConnectedComponentSegmentation& segment(const KDTree<ScalarT,EigenDim,DistAdaptor> &tree,
                                                const NeighborhoodSpecification<ScalarT> &nh,
                                                const std::vector<size_t> &seeds_ind,
                                                const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                size_t min_segment_size = 0,
                                                size_t max_segment_size = std::numeric_limits<size_t>::max())
        {
            switch (nh.type) {
                case NeighborhoodType::KNN:
                    segment_given_search_tree_<ScalarT,EigenDim,DistAdaptor,NeighborhoodType::KNN,PointSimilarityEvaluator>(tree, nh, seeds_ind, evaluator, min_segment_size, max_segment_size);
                    break;
                case NeighborhoodType::RADIUS:
                    segment_given_search_tree_<ScalarT,EigenDim,DistAdaptor,NeighborhoodType::RADIUS,PointSimilarityEvaluator>(tree, nh, seeds_ind, evaluator, min_segment_size, max_segment_size);
                    break;
                case NeighborhoodType::KNN_IN_RADIUS:
                    segment_given_search_tree_<ScalarT,EigenDim,DistAdaptor,NeighborhoodType::KNN_IN_RADIUS,PointSimilarityEvaluator>(tree, nh, seeds_ind, evaluator, min_segment_size, max_segment_size);
                    break;
            }
            return *this;
        }

        // Given search tree, all seeds
        template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
        ConnectedComponentSegmentation& segment(const KDTree<ScalarT,EigenDim,DistAdaptor> &tree,
                                                const NeighborhoodSpecification<ScalarT> &nh,
                                                const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                size_t min_segment_size = 0,
                                                size_t max_segment_size = std::numeric_limits<size_t>::max())
        {
            std::vector<size_t> seeds_ind(tree.getPointsMatrixMap().cols());
            for (size_t i = 0; i < seeds_ind.size(); i++) seeds_ind[i] = i;
            switch (nh.type) {
                case NeighborhoodType::KNN:
                    segment_given_search_tree_<ScalarT,EigenDim,DistAdaptor,NeighborhoodType::KNN,PointSimilarityEvaluator>(tree, nh, seeds_ind, evaluator, min_segment_size, max_segment_size);
                    break;
                case NeighborhoodType::RADIUS:
                    segment_given_search_tree_<ScalarT,EigenDim,DistAdaptor,NeighborhoodType::RADIUS,PointSimilarityEvaluator>(tree, nh, seeds_ind, evaluator, min_segment_size, max_segment_size);
                    break;
                case NeighborhoodType::KNN_IN_RADIUS:
                    segment_given_search_tree_<ScalarT,EigenDim,DistAdaptor,NeighborhoodType::KNN_IN_RADIUS,PointSimilarityEvaluator>(tree, nh, seeds_ind, evaluator, min_segment_size, max_segment_size);
                    break;
            }
            return *this;
        }

        // Given points and seeds
        template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
        ConnectedComponentSegmentation& segment(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                                const NeighborhoodSpecification<ScalarT> &nh,
                                                const std::vector<size_t> &seeds_ind,
                                                const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                size_t min_segment_size = 0,
                                                size_t max_segment_size = std::numeric_limits<size_t>::max())
        {
            switch (nh.type) {
                case NeighborhoodType::KNN:
                    segment_given_search_tree_<ScalarT,EigenDim,DistAdaptor,NeighborhoodType::KNN,PointSimilarityEvaluator>(KDTree<ScalarT,EigenDim,DistAdaptor>(points), nh, seeds_ind, evaluator, min_segment_size, max_segment_size);
                    break;
                case NeighborhoodType::RADIUS:
                    segment_given_search_tree_<ScalarT,EigenDim,DistAdaptor,NeighborhoodType::RADIUS,PointSimilarityEvaluator>(KDTree<ScalarT,EigenDim,DistAdaptor>(points), nh, seeds_ind, evaluator, min_segment_size, max_segment_size);
                    break;
                case NeighborhoodType::KNN_IN_RADIUS:
                    segment_given_search_tree_<ScalarT,EigenDim,DistAdaptor,NeighborhoodType::KNN_IN_RADIUS,PointSimilarityEvaluator>(KDTree<ScalarT,EigenDim,DistAdaptor>(points), nh, seeds_ind, evaluator, min_segment_size, max_segment_size);
                    break;
            }
            return *this;
        }

        // Given points, all seeds
        template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
        ConnectedComponentSegmentation& segment(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                                const NeighborhoodSpecification<ScalarT> &nh,
                                                const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
                                                size_t min_segment_size = 0,
                                                size_t max_segment_size = std::numeric_limits<size_t>::max())
        {
            std::vector<size_t> seeds_ind(points.cols());
            for (size_t i = 0; i < seeds_ind.size(); i++) seeds_ind[i] = i;
            switch (nh.type) {
                case NeighborhoodType::KNN:
                    segment_given_search_tree_<ScalarT,EigenDim,DistAdaptor,NeighborhoodType::KNN,PointSimilarityEvaluator>(KDTree<ScalarT,EigenDim,DistAdaptor>(points), nh, seeds_ind, evaluator, min_segment_size, max_segment_size);
                    break;
                case NeighborhoodType::RADIUS:
                    segment_given_search_tree_<ScalarT,EigenDim,DistAdaptor,NeighborhoodType::RADIUS,PointSimilarityEvaluator>(KDTree<ScalarT,EigenDim,DistAdaptor>(points), nh, seeds_ind, evaluator, min_segment_size, max_segment_size);
                    break;
                case NeighborhoodType::KNN_IN_RADIUS:
                    segment_given_search_tree_<ScalarT,EigenDim,DistAdaptor,NeighborhoodType::KNN_IN_RADIUS,PointSimilarityEvaluator>(KDTree<ScalarT,EigenDim,DistAdaptor>(points), nh, seeds_ind, evaluator, min_segment_size, max_segment_size);
                    break;
            }
            return *this;
        }

//        // Given search tree and seeds (NeighborhoodType given as template parameter)
//        template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, NeighborhoodType NT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
//        ConnectedComponentSegmentation& segment(const KDTree<ScalarT,EigenDim,DistAdaptor> &tree,
//                                                const NeighborhoodSpecification<ScalarT> &nh,
//                                                const std::vector<size_t> &seeds_ind,
//                                                const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
//                                                size_t min_segment_size = 0,
//                                                size_t max_segment_size = std::numeric_limits<size_t>::max())
//        {
//            segment_given_search_tree_<ScalarT,EigenDim,DistAdaptor,NT,PointSimilarityEvaluator>(tree, nh, seeds_ind, evaluator, min_segment_size, max_segment_size);
//            return *this;
//        }
//
//        // Given search tree, all seeds (NeighborhoodType given as template parameter)
//        template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, NeighborhoodType NT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
//        ConnectedComponentSegmentation& segment(const KDTree<ScalarT,EigenDim,DistAdaptor> &tree,
//                                                const NeighborhoodSpecification<ScalarT> &nh,
//                                                const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
//                                                size_t min_segment_size = 0,
//                                                size_t max_segment_size = std::numeric_limits<size_t>::max())
//        {
//            std::vector<size_t> seeds_ind(tree.getPointsMatrixMap().cols());
//            for (size_t i = 0; i < seeds_ind.size(); i++) seeds_ind[i] = i;
//            segment_given_search_tree_<ScalarT,EigenDim,DistAdaptor,NT,PointSimilarityEvaluator>(tree, nh, seeds_ind, evaluator, min_segment_size, max_segment_size);
//            return *this;
//        }
//
//        // Given points and seeds (NeighborhoodType given as template parameter)
//        template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, NeighborhoodType NT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
//        ConnectedComponentSegmentation& segment(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
//                                                const NeighborhoodSpecification<ScalarT> &nh,
//                                                const std::vector<size_t> &seeds_ind,
//                                                const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
//                                                size_t min_segment_size = 0,
//                                                size_t max_segment_size = std::numeric_limits<size_t>::max())
//        {
//            segment_given_search_tree_<ScalarT,EigenDim,DistAdaptor,NT,PointSimilarityEvaluator>(KDTree<ScalarT,EigenDim,DistAdaptor>(points), nh, seeds_ind, evaluator, min_segment_size, max_segment_size);
//            return *this;
//        }
//
//        // Given points, all seeds (NeighborhoodType given as template parameter)
//        template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, NeighborhoodType NT, class PointSimilarityEvaluator = AlwaysTrueEvaluator<ScalarT>>
//        ConnectedComponentSegmentation& segment(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
//                                                const NeighborhoodSpecification<ScalarT> &nh,
//                                                const PointSimilarityEvaluator &evaluator = PointSimilarityEvaluator(),
//                                                size_t min_segment_size = 0,
//                                                size_t max_segment_size = std::numeric_limits<size_t>::max())
//        {
//            std::vector<size_t> seeds_ind(points.cols());
//            for (size_t i = 0; i < seeds_ind.size(); i++) seeds_ind[i] = i;
//            segment_given_search_tree_<ScalarT,EigenDim,DistAdaptor,NT,PointSimilarityEvaluator>(KDTree<ScalarT,EigenDim,DistAdaptor>(points), nh, seeds_ind, evaluator, min_segment_size, max_segment_size);
//            return *this;
//        }

        inline const std::vector<std::vector<size_t>>& getComponentPointIndices() const { return component_point_indices_; }

        inline const std::vector<size_t>& getComponentIndexMap() const { return segment_index_map_; }

        std::vector<size_t> getUnlabeledPointIndices() const {
            const size_t no_label = component_point_indices_.size();
            std::vector<size_t> res;
            res.reserve(segment_index_map_.size());
            for (size_t i = 0; i < segment_index_map_.size(); i++) {
                if (segment_index_map_[i] == no_label) res.emplace_back(i);
            }
            return res;
        }

        inline size_t getNumberOfSegments() const { return component_point_indices_.size(); }

    private:
        std::vector<std::vector<size_t>> component_point_indices_;
        std::vector<size_t> segment_index_map_;

        template <typename ScalarT, class PointSimilarityEvaluator>
        void segment_given_neighbors_(const std::vector<NeighborSet<ScalarT>> &neighbors,
                                      const std::vector<size_t> &seeds_ind,
                                      const PointSimilarityEvaluator &evaluator,
                                      size_t min_segment_size,
                                      size_t max_segment_size)
        {
            const size_t unassigned = std::numeric_limits<size_t>::max();
            std::vector<size_t> current_label(neighbors.size(), unassigned);

            std::vector<size_t> frontier_set;
            frontier_set.reserve(neighbors.size());

//            std::vector<std::set<size_t>> ind_per_seed(seeds_ind.size());
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

//                    ind_per_seed[i].insert(curr_seed);
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

            component_point_indices_.clear();
            for (size_t i = seeds_to_merge_with.size() - 1; i != static_cast<size_t>(-1); i--) {
                if (seeds_to_merge_with[i].empty()) continue;
                size_t min_seed_ind = *seeds_to_merge_with[i].begin();
                if (min_seed_ind < i) {
                    for (auto it = seeds_to_merge_with[i].begin(); it != seeds_to_merge_with[i].end(); ++it) {
                        if (*it < i) seeds_to_merge_with[*it].insert(seeds_to_merge_with[i].begin(), seeds_to_merge_with[i].end());
                    }
//                    seeds_to_merge_with[i].clear();
                } else {
                    std::set<size_t> curr_cc_ind;
                    for (auto it = seeds_to_merge_with[i].begin(); it != seeds_to_merge_with[i].end(); ++it) {
                        curr_cc_ind.insert(ind_per_seed[*it].begin(), ind_per_seed[*it].end());
                    }
                    if (curr_cc_ind.size() >= min_segment_size && curr_cc_ind.size() <= max_segment_size) {
                        component_point_indices_.emplace_back(curr_cc_ind.begin(), curr_cc_ind.end());
                    }
                }
            }

            std::sort(component_point_indices_.begin(), component_point_indices_.end(), SizeGreaterComparator<std::vector<size_t>>());

            segment_index_map_ = std::vector<size_t>(neighbors.size(), component_point_indices_.size());
            for (size_t i = 0; i < component_point_indices_.size(); i++) {
                for (size_t j = 0; j < component_point_indices_[i].size(); j++) {
                    segment_index_map_[component_point_indices_[i][j]] = i;
                }
            }
        }

        template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor, NeighborhoodType NT, class PointSimilarityEvaluator>
        void segment_given_search_tree_(const KDTree<ScalarT,EigenDim,DistAdaptor> &tree,
                                        const NeighborhoodSpecification<ScalarT> &nh,
                                        const std::vector<size_t> &seeds_ind,
                                        const PointSimilarityEvaluator &evaluator,
                                        size_t min_segment_size,
                                        size_t max_segment_size)
        {
            const ConstVectorSetMatrixMap<ScalarT,EigenDim>& points(tree.getPointsMatrixMap());

            const size_t unassigned = std::numeric_limits<size_t>::max();
            std::vector<size_t> current_label(points.cols(), unassigned);

            std::vector<size_t> frontier_set;
            frontier_set.reserve(points.cols());

//            std::vector<std::set<size_t>> ind_per_seed(seeds_ind.size());
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

//                    ind_per_seed[i].insert(curr_seed);
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

            component_point_indices_.clear();
            for (size_t i = seeds_to_merge_with.size() - 1; i != static_cast<size_t>(-1); i--) {
                if (seeds_to_merge_with[i].empty()) continue;
                size_t min_seed_ind = *seeds_to_merge_with[i].begin();
                if (min_seed_ind < i) {
                    for (auto it = seeds_to_merge_with[i].begin(); it != seeds_to_merge_with[i].end(); ++it) {
                        if (*it < i) seeds_to_merge_with[*it].insert(seeds_to_merge_with[i].begin(), seeds_to_merge_with[i].end());
                    }
//                    seeds_to_merge_with[i].clear();
                } else {
                    std::set<size_t> curr_cc_ind;
                    for (auto it = seeds_to_merge_with[i].begin(); it != seeds_to_merge_with[i].end(); ++it) {
                        curr_cc_ind.insert(ind_per_seed[*it].begin(), ind_per_seed[*it].end());
                    }
                    if (curr_cc_ind.size() >= min_segment_size && curr_cc_ind.size() <= max_segment_size) {
                        component_point_indices_.emplace_back(curr_cc_ind.begin(), curr_cc_ind.end());
                    }
                }
            }

            std::sort(component_point_indices_.begin(), component_point_indices_.end(), SizeGreaterComparator<std::vector<size_t>>());

            segment_index_map_ = std::vector<size_t>(points.cols(), component_point_indices_.size());
            for (size_t i = 0; i < component_point_indices_.size(); i++) {
                for (size_t j = 0; j < component_point_indices_[i].size(); j++) {
                    segment_index_map_[component_point_indices_[i][j]] = i;
                }
            }
        }
    };
}
