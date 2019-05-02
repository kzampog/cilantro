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
        KNNSearchResultAdaptor(Neighborhood<ScalarT> &results, size_t k, ScalarT max_radius = std::numeric_limits<ScalarT>::max())
                : results_(results), k_(k), count_(0)
        {
            results_.resize(k_);
            results_[k_-1].value = max_radius;
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
        Neighborhood<ScalarT>& results_;
        const size_t k_;
        size_t count_;
    };


    template <typename ScalarT>
    class RadiusSearchResultAdaptor {
    public:
        RadiusSearchResultAdaptor(Neighborhood<ScalarT> &results, ScalarT radius)
                : results_(results), radius_(radius)
        {
            results_.clear();
        }

        inline size_t size() const { return results_.size(); }

        inline bool full() const { return true; }

        inline bool addPoint(ScalarT dist, size_t index) {
            // dist < worstDist() is guaranteed when addPoint is called.
            results_.emplace_back(index, dist);
            return true;
        }

        inline ScalarT worstDist() const { return radius_; }

    private:
        Neighborhood<ScalarT>& results_;
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
                  data_adaptor_(data_map_),
                  kd_tree_(data.rows(), data_adaptor_, nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_size))
        {
            params_.sorted = true;
            kd_tree_.buildIndex();
        }

        ~KDTree() {}

        inline const ConstVectorSetMatrixMap<ScalarT,EigenDim>& getPointsMatrixMap() const { return data_map_; }

        inline bool isEmpty() const { return data_map_.cols() > 0; }

        // Do not call if tree is empty!
        inline const KDTree& nearestNeighborSearch(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                                   Neighbor<ScalarT> &result) const
        {
            kd_tree_.knnSearch(query_pt.data(), 1, &result.index, &result.value);
            return *this;
        }

        // Do not call if tree is empty!
        inline Neighbor<ScalarT> nearestNeighborSearch(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt) const
        {
            Neighbor<ScalarT> result;
            kd_tree_.knnSearch(query_pt.data(), 1, &result.index, &result.value);
            return result;
        }

        // Do not call if tree is empty!
        const KDTree& nearestNeighborSearch(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                            Neighborhood<ScalarT> &results) const
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

        inline const KDTree& kNNSearch(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                       size_t k,
                                       Neighborhood<ScalarT> &results) const
        {
            KNNSearchResultAdaptor<ScalarT> sra(results, k);
            kd_tree_.findNeighbors(sra, query_pt.data(), params_);
            results.resize(sra.size());
            return *this;
        }

        inline Neighborhood<ScalarT> kNNSearch(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                               size_t k) const
        {
            Neighborhood<ScalarT> results;
            kNNSearch(query_pt, k, results);
            return results;
        }

        const KDTree& kNNSearch(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                size_t k,
                                NeighborhoodSet<ScalarT> &results) const
        {
            results.resize(query_pts.cols());
#pragma omp parallel for shared (results)
            for (size_t i = 0; i < query_pts.cols(); i++) {
                kNNSearch(query_pts.col(i), k, results[i]);
            }
            return *this;
        }

        inline NeighborhoodSet<ScalarT> kNNSearch(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                                  size_t k) const
        {
            NeighborhoodSet<ScalarT> results;
            kNNSearch(query_pts, k, results);
            return results;
        }

        inline const KDTree& radiusSearch(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                          ScalarT radius,
                                          Neighborhood<ScalarT> &results) const
        {
            RadiusSearchResultAdaptor<ScalarT> sra(results, radius);
            kd_tree_.findNeighbors(sra, query_pt.data(), params_);
            std::sort(results.begin(), results.end(), typename Neighbor<ScalarT>::ValueLessComparator());
            return *this;
        }

        inline Neighborhood<ScalarT> radiusSearch(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                                  ScalarT radius) const
        {
            Neighborhood<ScalarT> results;
            radiusSearch(query_pt, radius, results);
            return results;
        }

        const KDTree& radiusSearch(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                   ScalarT radius,
                                   NeighborhoodSet<ScalarT> &results) const
        {
            results.resize(query_pts.cols());
#pragma omp parallel for shared (results)
            for (size_t i = 0; i < query_pts.cols(); i++) {
                radiusSearch(query_pts.col(i), radius, results[i]);
            }
            return *this;
        }

        inline NeighborhoodSet<ScalarT> radiusSearch(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                                     ScalarT radius) const
        {
            NeighborhoodSet<ScalarT> results;
            radiusSearch(query_pts, radius, results);
            return results;
        }

        inline const KDTree& kNNInRadiusSearch(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                               size_t k,
                                               ScalarT radius,
                                               Neighborhood<ScalarT> &results) const
        {
            KNNSearchResultAdaptor<ScalarT> sra(results, k, radius);
            kd_tree_.findNeighbors(sra, query_pt.data(), params_);
            results.resize(sra.size());
            return *this;
        }

        inline Neighborhood<ScalarT> kNNInRadiusSearch(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
                                                       size_t k,
                                                       ScalarT radius) const
        {
            Neighborhood<ScalarT> results;
            kNNInRadiusSearch(query_pt, k, radius, results);
            return results;
        }

        const KDTree& kNNInRadiusSearch(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                        size_t k,
                                        ScalarT radius,
                                        NeighborhoodSet<ScalarT> &results) const
        {
            results.resize(query_pts.cols());
#pragma omp parallel for shared (results)
            for (size_t i = 0; i < query_pts.cols(); i++) {
                kNNInRadiusSearch(query_pts.col(i), k, radius, results[i]);
            }
            return *this;
        }

        inline NeighborhoodSet<ScalarT> kNNInRadiusSearch(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
                                                          size_t k,
                                                          ScalarT radius) const
        {
            NeighborhoodSet<ScalarT> results;
            kNNInRadiusSearch(query_pts, k, radius, results);
            return results;
        }

        template <typename NeighborhoodSpecT>
        inline typename std::enable_if<std::is_same<NeighborhoodSpecT,KNNNeighborhoodSpecification>::value,const KDTree&>::type
        search(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
               const NeighborhoodSpecT &nh,
               Neighborhood<ScalarT> &results) const
        {
            kNNSearch(query_pt, nh.maxNumberOfNeighbors, results);
            return *this;
        }

        template <typename NeighborhoodSpecT>
        inline typename std::enable_if<std::is_same<NeighborhoodSpecT,KNNNeighborhoodSpecification>::value,const KDTree&>::type
        search(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
               const NeighborhoodSpecT &nh,
               NeighborhoodSet<ScalarT> &results) const
        {
            kNNSearch(query_pts, nh.maxNumberOfNeighbors, results);
            return *this;
        }

        template <typename NeighborhoodSpecT>
        inline typename std::enable_if<std::is_same<NeighborhoodSpecT,RadiusNeighborhoodSpecification<ScalarT>>::value,const KDTree&>::type
        search(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
               const NeighborhoodSpecT &nh,
               Neighborhood<ScalarT> &results) const
        {
            radiusSearch(query_pt, nh.radius, results);
            return *this;
        }

        template <typename NeighborhoodSpecT>
        inline typename std::enable_if<std::is_same<NeighborhoodSpecT,RadiusNeighborhoodSpecification<ScalarT>>::value,const KDTree&>::type
        search(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
               const NeighborhoodSpecT &nh,
               NeighborhoodSet<ScalarT> &results) const
        {
            radiusSearch(query_pts, nh.radius, results);
            return *this;
        }

        template <typename NeighborhoodSpecT>
        inline typename std::enable_if<std::is_same<NeighborhoodSpecT,KNNInRadiusNeighborhoodSpecification<ScalarT>>::value,const KDTree&>::type
        search(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &query_pt,
               const NeighborhoodSpecT &nh,
               Neighborhood<ScalarT> &results) const
        {
            kNNInRadiusSearch(query_pt, nh.maxNumberOfNeighbors, nh.radius, results);
            return *this;
        }

        template <typename NeighborhoodSpecT>
        inline typename std::enable_if<std::is_same<NeighborhoodSpecT,KNNInRadiusNeighborhoodSpecification<ScalarT>>::value,const KDTree&>::type
        search(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &query_pts,
               const NeighborhoodSpecT &nh,
               NeighborhoodSet<ScalarT> &results) const
        {
            kNNInRadiusSearch(query_pts, nh.maxNumberOfNeighbors, nh.radius, results);
            return *this;
        }

        template <typename PointT, typename NeighborhoodSpecT>
        inline typename std::enable_if<PointT::ColsAtCompileTime == 1,Neighborhood<ScalarT>>::type
        search(const PointT &query_pt,
               const NeighborhoodSpecT &nh) const
        {
            Neighborhood<ScalarT> res;
            search(query_pt, nh, res);
            return res;
        }

        template <typename PointsT, typename NeighborhoodSpecT>
        inline typename std::enable_if<PointsT::ColsAtCompileTime == Eigen::Dynamic,NeighborhoodSet<ScalarT>>::type
        search(const PointsT &query_pts,
               const NeighborhoodSpecT &nh) const
        {
            NeighborhoodSet<ScalarT> res;
            search(query_pts, nh, res);
            return res;
        }

    private:
        typedef nanoflann::KDTreeSingleIndexAdaptor<DistAdaptor<KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim>>, KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim>, EigenDim> TreeType_;

        ConstVectorSetMatrixMap<ScalarT,EigenDim> data_map_;
        const KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim> data_adaptor_;
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
