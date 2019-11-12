#pragma once

#include <Eigen/Dense>
#include <cilantro/registration/correspondence_search_combined_metric_adaptor.hpp>

namespace cilantro {
    template <class PointToPointCorrespondenceSearchT, class PointToPlaneCorrespondenceSearchT>
    class CorrespondenceSearchCombinedMetricCombiner {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef PointToPointCorrespondenceSearchT PointToPointCorrespondenceSearch;

        typedef PointToPlaneCorrespondenceSearchT PointToPlaneCorrespondenceSearch;

        typedef typename CorrespondenceSearchCombinedMetricAdaptor<PointToPointCorrespondenceSearchT>::PointToPointCorrespondenceScalar PointToPointCorrespondenceScalar;

        typedef typename CorrespondenceSearchCombinedMetricAdaptor<PointToPointCorrespondenceSearchT>::PointToPointCorrespondenceSearchResult PointToPointCorrespondenceSearchResult;

        typedef typename CorrespondenceSearchCombinedMetricAdaptor<PointToPlaneCorrespondenceSearchT>::PointToPlaneCorrespondenceScalar PointToPlaneCorrespondenceScalar;

        typedef typename CorrespondenceSearchCombinedMetricAdaptor<PointToPlaneCorrespondenceSearchT>::PointToPlaneCorrespondenceSearchResult PointToPlaneCorrespondenceSearchResult;

        inline CorrespondenceSearchCombinedMetricCombiner(PointToPointCorrespondenceSearch& point_to_point_corr_search,
                                                          PointToPlaneCorrespondenceSearch& point_to_plane_corr_search)
                : point_to_point_corr_search_(point_to_point_corr_search),
                  point_to_plane_corr_search_(point_to_plane_corr_search)
        {}

        inline CorrespondenceSearchCombinedMetricCombiner& findCorrespondences() {
            if (std::is_same<PointToPointCorrespondenceSearchT,PointToPlaneCorrespondenceSearchT>::value &&
                &point_to_point_corr_search_ == (PointToPointCorrespondenceSearchT *)(&point_to_plane_corr_search_))
            {
                point_to_point_corr_search_.findCorrespondences();
            } else {
                point_to_point_corr_search_.findCorrespondences();
                point_to_plane_corr_search_.findCorrespondences();
            }
            return *this;
        }

        template <class TransformT>
        inline CorrespondenceSearchCombinedMetricCombiner& findCorrespondences(const TransformT& tform) {
            if (std::is_same<PointToPointCorrespondenceSearchT,PointToPlaneCorrespondenceSearchT>::value &&
                &point_to_point_corr_search_ == (PointToPointCorrespondenceSearchT *)(&point_to_plane_corr_search_))
            {
                point_to_point_corr_search_.findCorrespondences(tform);
            } else {
                point_to_point_corr_search_.findCorrespondences(tform);
                point_to_plane_corr_search_.findCorrespondences(tform);
            }
            return *this;
        }

        inline const PointToPointCorrespondenceSearchResult& getPointToPointCorrespondences() const {
            return CorrespondenceSearchCombinedMetricAdaptor<PointToPointCorrespondenceSearchT>(point_to_point_corr_search_).getPointToPointCorrespondences();
        }

        inline const PointToPlaneCorrespondenceSearchResult& getPointToPlaneCorrespondences() const {
            return CorrespondenceSearchCombinedMetricAdaptor<PointToPlaneCorrespondenceSearchT>(point_to_plane_corr_search_).getPointToPlaneCorrespondences();
        }

        inline PointToPointCorrespondenceSearch& pointToPointCorrespondenceSearchEngine() {
            return point_to_point_corr_search_;
        }

        inline PointToPlaneCorrespondenceSearch& pointToPlaneCorrespondenceSearchEngine() {
            return point_to_plane_corr_search_;
        }

    private:
        PointToPointCorrespondenceSearch& point_to_point_corr_search_;
        PointToPlaneCorrespondenceSearch& point_to_plane_corr_search_;
    };
}
