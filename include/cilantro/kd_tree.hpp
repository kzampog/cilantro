#pragma once

#include <cilantro/3rd_party/nanoflann/nanoflann.hpp>
#include <cilantro/data_matrix_map.hpp>

namespace cilantro {
    struct KDTreeDataAdaptors {
        // Eigen Map to nanoflann adaptor class
        template <class ScalarT, ptrdiff_t EigenDim>
        struct EigenMap {
            typedef ScalarT coord_t;

            // A const ref to the data set origin
            const Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> >& obj;

            /// The constructor that sets the data set source
            EigenMap(const Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> > &obj_) : obj(obj_) {}

            /// CRTP helper method
            inline const Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> >& derived() const { return obj; }

            // Must return the number of data points
            inline size_t kdtree_get_point_count() const { return obj.cols(); }

            // Returns the dim'th component of the idx'th point in the class:
            // Since this is inlined and the "dim" argument is typically an immediate value, the
            //  "if/else's" are actually solved at compile time.
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

    template <typename ScalarT, ptrdiff_t EigenDim, template <class> class DistAdaptor>
    class KDTree {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        enum struct NeighborhoodType {KNN, RADIUS, KNN_IN_RADIUS};
        struct Neighborhood {
            inline Neighborhood() : type(NeighborhoodType::KNN), maxNumberOfNeighbors(1) {}
            inline Neighborhood(size_t knn, ScalarT radius) : type(NeighborhoodType::KNN_IN_RADIUS), maxNumberOfNeighbors(knn), radius(radius) {}
            inline Neighborhood(NeighborhoodType type, size_t knn, ScalarT radius) : type(type), maxNumberOfNeighbors(knn), radius(radius) {}

            NeighborhoodType type;
            size_t maxNumberOfNeighbors;
            ScalarT radius;
        };

        KDTree(const ConstDataMatrixMap<ScalarT,EigenDim> &data, size_t max_leaf_size = 10)
                : data_map_(data.data(), EigenDim, data.cols()),
                  mat_to_kd_(data_map_),
                  kd_tree_(EigenDim, mat_to_kd_, nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_size))
        {
            params_.sorted = true;
            kd_tree_.buildIndex();
        }

        ~KDTree() {}

        void nearestNeighborSearch(const Eigen::Matrix<ScalarT,EigenDim,1> &query_pt, size_t &neighbor, ScalarT &distance) const {
            kd_tree_.knnSearch(query_pt.data(), 1, &neighbor, &distance);
        }

        void kNNSearch(const Eigen::Matrix<ScalarT,EigenDim,1> &query_pt, size_t k, std::vector<size_t> &neighbors, std::vector<ScalarT> &distances) const {
            neighbors.resize(k);
            distances.resize(k);
            size_t num_results = kd_tree_.knnSearch(query_pt.data(), k, neighbors.data(), distances.data());
            neighbors.resize(num_results);
            distances.resize(num_results);
        }

        void radiusSearch(const Eigen::Matrix<ScalarT,EigenDim,1> &query_pt, ScalarT radius, std::vector<size_t> &neighbors, std::vector<ScalarT> &distances) const {
            std::vector<std::pair<size_t,ScalarT> > matches;
            matches.reserve(data_map_.cols());
            size_t num_results = kd_tree_.radiusSearch(query_pt.data(), radius, matches, params_);
            neighbors.resize(num_results);
            distances.resize(num_results);
            for (size_t i = 0; i < num_results; i++) {
                neighbors[i] = matches[i].first;
                distances[i] = matches[i].second;
            }
        }

        void kNNInRadiusSearch(const Eigen::Matrix<ScalarT,EigenDim,1> &query_pt, size_t k, ScalarT radius, std::vector<size_t> &neighbors, std::vector<ScalarT> &distances) const {
            KDTree::kNNSearch(query_pt, k, neighbors, distances);
            size_t ind = neighbors.size() - 1;
            while (ind >= 0 && distances[ind] >= radius) ind--;
            neighbors.resize(ind+1);
            distances.resize(ind+1);
//            KDTree::radiusSearch(query_pt, radius, neighbors, distances);
//            if (neighbors.size() > k) {
//                neighbors.resize(k);
//                distances.resize(k);
//            }
        }

        inline void search(const Eigen::Matrix<ScalarT,EigenDim,1> &query_pt, std::vector<size_t> &neighbors, std::vector<ScalarT> &distances, const Neighborhood &nh) const {
            switch (nh.type) {
                case NeighborhoodType::KNN:
                    kNNSearch(query_pt, nh.maxNumberOfNeighbors, neighbors, distances);
                    break;
                case NeighborhoodType::RADIUS:
                    radiusSearch(query_pt, nh.radius, neighbors, distances);
                    break;
                case NeighborhoodType::KNN_IN_RADIUS:
                    kNNInRadiusSearch(query_pt, nh.maxNumberOfNeighbors, nh.radius, neighbors, distances);
                    break;
            }
        }

    private:
        typedef nanoflann::KDTreeSingleIndexAdaptor<DistAdaptor<KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim> >, KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim>, EigenDim> TreeType_;

        Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> > data_map_;
        const KDTreeDataAdaptors::EigenMap<ScalarT,EigenDim> mat_to_kd_;
        TreeType_ kd_tree_;
        nanoflann::SearchParams params_;
    };

    typedef KDTree<float,2,KDTreeDistanceAdaptors::L2> KDTree2D;
    typedef KDTree<float,3,KDTreeDistanceAdaptors::L2> KDTree3D;
}
