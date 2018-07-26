#pragma once

#include <cilantro/3rd_party/nanoflann/nanoflann.hpp>
#include <cilantro/data_containers.hpp>
#include <cilantro/nearest_neighbors.hpp>

namespace cilantro {
    struct KDTreeDataAdaptors {
        // Eigen Map to nanoflann adaptor class
        template <class ScalarT, ptrdiff_t EigenDim>
        struct EigenMap {
            typedef ScalarT coord_t;

            // A const ref to the data set origin
            const Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>>& obj;

            // The constructor that sets the data set source
            EigenMap(const Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>> &obj_) : obj(obj_) {}

            // CRTP helper method
            inline const Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>>& derived() const { return obj; }

            // Must return the number of data points
            inline size_t kdtree_get_point_count() const { return obj.cols(); }

            // Returns the dim'th component of the idx'th point in the class
            inline coord_t kdtree_get_pt(const size_t idx, int dim) const { return obj(dim,idx); }

            // Optional bounding-box computation: return false to default to a standard bbox computation loop.
            //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
            //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
            template <class BBOX>
            bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }
        };
    };

    struct KDTreeDistanceAdaptors {
        template <class DataAdaptor>
        using L1 = nanoflann::L1_Adaptor<typename DataAdaptor::coord_t, DataAdaptor, typename DataAdaptor::coord_t>;

        template <class DataAdaptor>
        using L2 = nanoflann::L2_Adaptor<typename DataAdaptor::coord_t, DataAdaptor, typename DataAdaptor::coord_t>;

        template <class DataAdaptor>
        using L2Simple = nanoflann::L2_Simple_Adaptor<typename DataAdaptor::coord_t, DataAdaptor, typename DataAdaptor::coord_t>;

        template <class DataAdaptor>
        using SO2 = nanoflann::SO2_Adaptor<typename DataAdaptor::coord_t, DataAdaptor, typename DataAdaptor::coord_t>;

        template <class DataAdaptor>
        using SO3 = nanoflann::SO3_Adaptor<typename DataAdaptor::coord_t, DataAdaptor, typename DataAdaptor::coord_t>;
    };

    template <typename ScalarT>
    class KNNSearchResultAdaptor {
    public:
        KNNSearchResultAdaptor(NeighborSet<ScalarT> &results, size_t k)
                : results_(results), k_(k), count_(0)
        {
            results_.resize(k_);
            results_[k_-1].value = std::numeric_limits<ScalarT>::max();
        }

        inline size_t size() const { return count_; }

        inline bool full() const { return count_ == k_; }

        inline bool addPoint(ScalarT dist, size_t index) {
            size_t i;
            for (i = count_; i > 0; --i) {
                if (results_[i-1].value > dist) {
                    if (i < k_) {
                        results_[i].index = results_[i-1].index;
                        results_[i].value = results_[i-1].value;
                    }
                } else {
                    break;
                }
            }
            if (i < k_) {
                results_[i].index = index;
                results_[i].value = dist;
            }
            if (count_ < k_) count_++;

            return true;
        }

        inline ScalarT worstDist() const { return results_[k_-1].value; }

    private:
        NeighborSet<ScalarT>& results_;
        const size_t k_;
        size_t count_;
    };


    template <typename ScalarT>
    class RadiusSearchResultAdaptor {
    public:
        RadiusSearchResultAdaptor(NeighborSet<ScalarT> &results, ScalarT radius)
                : results_(results), radius_(radius)
        {
            results_.clear();
        }

        inline size_t size() const { return results_.size(); }

        inline bool full() const { return true; }

        inline bool addPoint(ScalarT dist, size_t index) {
            if (dist < radius_) results_.emplace_back(index, dist);
            return true;
        }

        inline ScalarT worstDist() const { return radius_; }

    private:
        NeighborSet<ScalarT>& results_;
        const ScalarT radius_;
    };

    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2>
    class KDTree {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum { Dimension = EigenDim };

        KDTree(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &data, size_t max_leaf_size = 10)
                : data_map_(data),
                  mat_to_kd_(data_map_),
                  kd_tree_(data.rows(), mat_to_kd_, nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_size))
        {
            params_.sorted = true;
            kd_tree_.buildIndex();
        }

        ~KDTree() {}

        inline const ConstVectorSetMatrixMap<ScalarT,EigenDim>& getPointsMatrixMap() const { return data_map_; }

        inline bool isEmpty() const { return data_map_.cols() > 0; }

        inline void nearestNeighborSearch(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                          Neighbor<ScalarT> &result) const
        {
            kd_tree_.knnSearch(query_pt.data(), 1, &result.index, &result.value);
        }

        void nearestNeighborSearch(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                   NeighborSet<ScalarT> &results) const
        {
            results.resize(query_pts.cols());
#pragma omp parallel for shared (results)
            for (size_t i = 0; i < query_pts.cols(); i++) {
                nearestNeighborSearch(query_pts.col(i), results[i]);
            }
        }

        inline void kNNSearch(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                              size_t k,
                              NeighborSet<ScalarT> &results) const
        {
            KNNSearchResultAdaptor<ScalarT> sra(results, k);
            kd_tree_.findNeighbors(sra, query_pt.data(), params_);
            results.resize(sra.size());
        }

        void kNNSearch(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                       size_t k,
                       std::vector<NeighborSet<ScalarT>> &results) const
        {
            results.resize(query_pts.cols());
#pragma omp parallel for shared (results)
            for (size_t i = 0; i < query_pts.cols(); i++) {
                kNNSearch(query_pts.col(i), k, results[i]);
            }
        }

        inline void radiusSearch(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                 ScalarT radius,
                                 NeighborSet<ScalarT> &results) const
        {
            RadiusSearchResultAdaptor<ScalarT> sra(results, radius);
            kd_tree_.findNeighbors(sra, query_pt.data(), params_);
            std::sort(results.begin(), results.end(), typename Neighbor<ScalarT>::ValueLessComparator());
        }

        void radiusSearch(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                          ScalarT radius,
                          std::vector<NeighborSet<ScalarT>> &results) const
        {
            results.resize(query_pts.cols());
#pragma omp parallel for shared (results)
            for (size_t i = 0; i < query_pts.cols(); i++) {
                radiusSearch(query_pts.col(i), radius, results[i]);
            }
        }

        inline void kNNInRadiusSearch(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                      size_t k,
                                      ScalarT radius,
                                      NeighborSet<ScalarT> &results) const
        {
            this->kNNSearch(query_pt, k, results);
            size_t ind = results.size() - 1;
            while (ind != static_cast<size_t>(-1) && results[ind].value >= radius) ind--;
            results.resize(ind+1);
//            KDTree::radiusSearch(query_pt, radius, neighbors, distances);
//            if (neighbors.size() > k) {
//                neighbors.resize(k);
//                distances.resize(k);
//            }
        }

        void kNNInRadiusSearch(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                               size_t k,
                               ScalarT radius,
                               std::vector<NeighborSet<ScalarT>> &results) const
        {
            results.resize(query_pts.cols());
#pragma omp parallel for shared (results)
            for (size_t i = 0; i < query_pts.cols(); i++) {
                kNNInRadiusSearch(query_pts.col(i), k, radius, results[i]);
            }
        }

        template <NeighborhoodType NT>
        inline typename std::enable_if<NT == NeighborhoodType::KNN, void>::type search(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                                                                       const NeighborhoodSpecification<ScalarT> &nh,
                                                                                       NeighborSet<ScalarT> &results) const
        {
            kNNSearch(query_pt, nh.maxNumberOfNeighbors, results);
        }

        template <NeighborhoodType NT>
        inline typename std::enable_if<NT == NeighborhoodType::KNN, void>::type search(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                                                                       const NeighborhoodSpecification<ScalarT> &nh,
                                                                                       std::vector<NeighborSet<ScalarT>> &results) const
        {
            kNNSearch(query_pts, nh.maxNumberOfNeighbors, results);
        }

        template <NeighborhoodType NT>
        inline typename std::enable_if<NT == NeighborhoodType::RADIUS, void>::type search(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                                                                          const NeighborhoodSpecification<ScalarT> &nh,
                                                                                          NeighborSet<ScalarT> &results) const
        {
            radiusSearch(query_pt, nh.radius, results);
        }

        template <NeighborhoodType NT>
        inline typename std::enable_if<NT == NeighborhoodType::RADIUS, void>::type search(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                                                                          const NeighborhoodSpecification<ScalarT> &nh,
                                                                                          std::vector<NeighborSet<ScalarT>> &results) const
        {
            radiusSearch(query_pts, nh.radius, results);
        }

        template <NeighborhoodType NT>
        inline typename std::enable_if<NT == NeighborhoodType::KNN_IN_RADIUS, void>::type search(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                                                                                 const NeighborhoodSpecification<ScalarT> &nh,
                                                                                                 NeighborSet<ScalarT> &results) const
        {
            kNNInRadiusSearch(query_pt, nh.maxNumberOfNeighbors, nh.radius, results);
        }

        template <NeighborhoodType NT>
        inline typename std::enable_if<NT == NeighborhoodType::KNN_IN_RADIUS, void>::type search(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                                                                                 const NeighborhoodSpecification<ScalarT> &nh,
                                                                                                 std::vector<NeighborSet<ScalarT>> &results) const
        {
            kNNInRadiusSearch(query_pts, nh.maxNumberOfNeighbors, nh.radius, results);
        }

        inline void search(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                           const NeighborhoodSpecification<ScalarT> &nh,
                           NeighborSet<ScalarT> &results) const
        {
            switch (nh.type) {
                case NeighborhoodType::KNN:
                    kNNSearch(query_pt, nh.maxNumberOfNeighbors, results);
                    break;
                case NeighborhoodType::RADIUS:
                    radiusSearch(query_pt, nh.radius, results);
                    break;
                case NeighborhoodType::KNN_IN_RADIUS:
                    kNNInRadiusSearch(query_pt, nh.maxNumberOfNeighbors, nh.radius, results);
                    break;
            }
        }

        inline void search(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                           const NeighborhoodSpecification<ScalarT> &nh,
                           std::vector<NeighborSet<ScalarT>> &results) const
        {
            switch (nh.type) {
                case NeighborhoodType::KNN:
                    kNNSearch(query_pts, nh.maxNumberOfNeighbors, results);
                    break;
                case NeighborhoodType::RADIUS:
                    radiusSearch(query_pts, nh.radius, results);
                    break;
                case NeighborhoodType::KNN_IN_RADIUS:
                    kNNInRadiusSearch(query_pts, nh.maxNumberOfNeighbors, nh.radius, results);
                    break;
            }
        }

    private:
        typedef nanoflann::KDTreeSingleIndexAdaptor<DistAdaptor<KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim>>, KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim>, EigenDim> TreeType_;

        ConstVectorSetMatrixMap<ScalarT,EigenDim> data_map_;
        const KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim> mat_to_kd_;
        TreeType_ kd_tree_;
        nanoflann::SearchParams params_;
    };

    typedef KDTree<float,2,KDTreeDistanceAdaptors::L2> KDTree2f;
    typedef KDTree<double,2,KDTreeDistanceAdaptors::L2> KDTree2d;
    typedef KDTree<float,3,KDTreeDistanceAdaptors::L2> KDTree3f;
    typedef KDTree<double,3,KDTreeDistanceAdaptors::L2> KDTree3d;
    typedef KDTree<float,Eigen::Dynamic,KDTreeDistanceAdaptors::L2> KDTreeXf;
    typedef KDTree<double,Eigen::Dynamic,KDTreeDistanceAdaptors::L2> KDTreeXd;
}
