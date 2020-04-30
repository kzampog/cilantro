#pragma once

#include <cilantro/3rd_party/nanoflann/nanoflann.hpp>
#include <cilantro/core/data_containers.hpp>
#include <cilantro/core/nearest_neighbors.hpp>

namespace cilantro {
    namespace KDTreeDataAdaptors {
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
    }

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

    template <typename ScalarT, typename IndexT = size_t, typename CountT = size_t>
    class KNNSearchResultAdaptor {
    public:
        KNNSearchResultAdaptor(Neighborhood<ScalarT,IndexT> &results, CountT k, ScalarT max_radius = std::numeric_limits<ScalarT>::max())
                : results_(results), k_(k), count_(0)
        {
            results_.resize(k_);
            results_[k_-1].value = max_radius;
        }

        inline CountT size() const { return count_; }

        inline bool full() const { return count_ == k_; }

        inline bool addPoint(ScalarT dist, IndexT index) {
            CountT i;
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
        Neighborhood<ScalarT,IndexT>& results_;
        const CountT k_;
        CountT count_;
    };


    template <typename ScalarT, typename IndexT = size_t, typename CountT = size_t>
    class RadiusSearchResultAdaptor {
    public:
        RadiusSearchResultAdaptor(Neighborhood<ScalarT,IndexT> &results, ScalarT radius)
                : results_(results), radius_(radius)
        {
            results_.clear();
        }

        inline CountT size() const { return results_.size(); }

        inline bool full() const { return true; }

        inline bool addPoint(ScalarT dist, IndexT index) {
            // dist < worstDist() is guaranteed when addPoint is called.
            results_.emplace_back(index, dist);
            return true;
        }

        inline ScalarT worstDist() const { return radius_; }

    private:
        Neighborhood<ScalarT,IndexT>& results_;
        const ScalarT radius_;
    };

    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor = KDTreeDistanceAdaptors::L2, typename IndexT = size_t>
    class KDTree {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;
        typedef IndexT Index;

        typedef Neighbor<ScalarT,IndexT> NeighborResult;
        typedef Neighborhood<ScalarT,IndexT> NeighborhoodResult;
        typedef NeighborSet<ScalarT,IndexT> NeighborSetResult;
        typedef NeighborhoodSet<ScalarT,IndexT> NeighborhoodSetResult;

        enum { Dimension = EigenDim };

        typedef nanoflann::KDTreeSingleIndexAdaptor<DistAdaptor<KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim>>,KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim>,EigenDim,IndexT> InternalTree;

        KDTree(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &data, size_t max_leaf_size = 10)
                : data_map_(data),
                  data_adaptor_(data_map_),
                  kd_tree_(data.rows(), data_adaptor_, nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_size))
        {
            params_.sorted = true;
            kd_tree_.buildIndex();
        }

        ~KDTree() {}

        inline const ConstVectorSetMatrixMap<ScalarT,EigenDim>& getPointsMatrixMap() const { return data_map_; }

        inline bool isEmpty() const { return data_map_.cols() == 0; }

        inline const InternalTree& nanoflannTree() const { return kd_tree_; }

        // Do not call if tree is empty!
        inline const KDTree& nearestNeighborSearch(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                                   NeighborResult &result) const
        {
            kd_tree_.knnSearch(query_pt.data(), 1, &result.index, &result.value);
            return *this;
        }

        // Do not call if tree is empty!
        inline NeighborResult nearestNeighborSearch(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt) const
        {
            NeighborResult result;
            kd_tree_.knnSearch(query_pt.data(), 1, &result.index, &result.value);
            return result;
        }

        // Do not call if tree is empty!
        const KDTree& nearestNeighborSearch(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                            NeighborhoodResult &results) const
        {
            results.resize(query_pts.cols());
#pragma omp parallel for shared (results)
            for (size_t i = 0; i < query_pts.cols(); i++) {
                nearestNeighborSearch(query_pts.col(i), results[i]);
            }
            return *this;
        }

        // Do not call if tree is empty!
        // Unlike the other batch searches, this one returns a flat vector of Neighbor
        inline NeighborSet<ScalarT> nearestNeighborSearch(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts) const {
            NeighborSet<ScalarT> results;
            nearestNeighborSearch(query_pts, results);
            return results;
        }

        template <typename CountT = size_t>
        inline const KDTree& kNNSearch(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                       CountT k,
                                       NeighborhoodResult &results) const
        {
            KNNSearchResultAdaptor<ScalarT,IndexT,CountT> sra(results, k);
            kd_tree_.findNeighbors(sra, query_pt.data(), params_);
            results.resize(sra.size());
            return *this;
        }

        template <typename CountT = size_t>
        inline NeighborhoodResult kNNSearch(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                            CountT k) const
        {
            NeighborhoodResult results;
            kNNSearch(query_pt, k, results);
            return results;
        }

        template <typename CountT = size_t>
        const KDTree& kNNSearch(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                CountT k,
                                NeighborhoodSetResult &results) const
        {
            results.resize(query_pts.cols());
#pragma omp parallel for shared (results)
            for (size_t i = 0; i < query_pts.cols(); i++) {
                kNNSearch(query_pts.col(i), k, results[i]);
            }
            return *this;
        }

        template <typename CountT = size_t>
        inline NeighborhoodSetResult kNNSearch(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                               CountT k) const
        {
            NeighborhoodSetResult results;
            kNNSearch(query_pts, k, results);
            return results;
        }

        inline const KDTree& radiusSearch(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                          ScalarT radius,
                                          NeighborhoodResult &results) const
        {
            RadiusSearchResultAdaptor<ScalarT,IndexT,size_t> sra(results, radius);
            kd_tree_.findNeighbors(sra, query_pt.data(), params_);
            std::sort(results.begin(), results.end(), typename NeighborResult::ValueLessComparator());
            return *this;
        }

        inline NeighborhoodResult radiusSearch(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                               ScalarT radius) const
        {
            NeighborhoodResult results;
            radiusSearch(query_pt, radius, results);
            return results;
        }

        const KDTree& radiusSearch(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                   ScalarT radius,
                                   NeighborhoodSetResult &results) const
        {
            results.resize(query_pts.cols());
#pragma omp parallel for shared (results)
            for (size_t i = 0; i < query_pts.cols(); i++) {
                radiusSearch(query_pts.col(i), radius, results[i]);
            }
            return *this;
        }

        inline NeighborhoodSetResult radiusSearch(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                                  ScalarT radius) const
        {
            NeighborhoodSetResult results;
            radiusSearch(query_pts, radius, results);
            return results;
        }

        template <typename CountT = size_t>
        inline const KDTree& kNNInRadiusSearch(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                               CountT k,
                                               ScalarT radius,
                                               NeighborhoodResult &results) const
        {
            KNNSearchResultAdaptor<ScalarT,IndexT,CountT> sra(results, k, radius);
            kd_tree_.findNeighbors(sra, query_pt.data(), params_);
            results.resize(sra.size());
            return *this;
        }

        template <typename CountT = size_t>
        inline NeighborhoodResult kNNInRadiusSearch(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                                    CountT k,
                                                    ScalarT radius) const
        {
            NeighborhoodResult results;
            kNNInRadiusSearch(query_pt, k, radius, results);
            return results;
        }

        template <typename CountT = size_t>
        const KDTree& kNNInRadiusSearch(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                        CountT k,
                                        ScalarT radius,
                                        NeighborhoodSetResult &results) const
        {
            results.resize(query_pts.cols());
#pragma omp parallel for shared (results)
            for (size_t i = 0; i < query_pts.cols(); i++) {
                kNNInRadiusSearch(query_pts.col(i), k, radius, results[i]);
            }
            return *this;
        }

        template <typename CountT = size_t>
        inline NeighborhoodSetResult kNNInRadiusSearch(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                                       CountT k,
                                                       ScalarT radius) const
        {
            NeighborhoodSetResult results;
            kNNInRadiusSearch(query_pts, k, radius, results);
            return results;
        }

        template <typename CountT = size_t>
        inline const KDTree& search(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                    const KNNNeighborhoodSpecification<CountT> &nh,
                                    NeighborhoodResult &results) const
        {
            kNNSearch(query_pt, nh.maxNumberOfNeighbors, results);
            return *this;
        }

        template <typename CountT = size_t>
        inline const KDTree& search(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                    const KNNNeighborhoodSpecification<CountT> &nh,
                                    NeighborhoodSetResult &results) const
        {
            kNNSearch(query_pts, nh.maxNumberOfNeighbors, results);
            return *this;
        }

        inline const KDTree& search(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                    const RadiusNeighborhoodSpecification<ScalarT> &nh,
                                    NeighborhoodResult &results) const
        {
            radiusSearch(query_pt, nh.radius, results);
            return *this;
        }

        inline const KDTree& search(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                    const RadiusNeighborhoodSpecification<ScalarT> &nh,
                                    NeighborhoodSetResult &results) const
        {
            radiusSearch(query_pts, nh.radius, results);
            return *this;
        }

        template <typename CountT = size_t>
        inline const KDTree& search(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                    const KNNInRadiusNeighborhoodSpecification<ScalarT,CountT> &nh,
                                    NeighborhoodResult &results) const
        {
            kNNInRadiusSearch(query_pt, nh.maxNumberOfNeighbors, nh.radius, results);
            return *this;
        }

        template <typename CountT = size_t>
        inline const KDTree& search(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                    const KNNInRadiusNeighborhoodSpecification<ScalarT,CountT> &nh,
                                    NeighborhoodSetResult &results) const
        {
            kNNInRadiusSearch(query_pts, nh.maxNumberOfNeighbors, nh.radius, results);
            return *this;
        }

        template <typename PointT, typename NeighborhoodSpecT>
        inline typename std::enable_if<PointT::ColsAtCompileTime == 1,NeighborhoodResult>::type
        search(const PointT &query_pt,
               const NeighborhoodSpecT &nh) const
        {
            NeighborhoodResult res;
            search(query_pt, nh, res);
            return res;
        }

        template <typename PointsT, typename NeighborhoodSpecT>
        inline typename std::enable_if<PointsT::ColsAtCompileTime == Eigen::Dynamic,NeighborhoodSetResult>::type
        search(const PointsT &query_pts,
               const NeighborhoodSpecT &nh) const
        {
            NeighborhoodSetResult res;
            search(query_pts, nh, res);
            return res;
        }

    private:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> data_map_;
        const KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim> data_adaptor_;
        InternalTree kd_tree_;
        nanoflann::SearchParams params_;
    };

    typedef KDTree<float,2,KDTreeDistanceAdaptors::L2> KDTree2f;
    typedef KDTree<double,2,KDTreeDistanceAdaptors::L2> KDTree2d;
    typedef KDTree<float,3,KDTreeDistanceAdaptors::L2> KDTree3f;
    typedef KDTree<double,3,KDTreeDistanceAdaptors::L2> KDTree3d;
    typedef KDTree<float,Eigen::Dynamic,KDTreeDistanceAdaptors::L2> KDTreeXf;
    typedef KDTree<double,Eigen::Dynamic,KDTreeDistanceAdaptors::L2> KDTreeXd;
}
