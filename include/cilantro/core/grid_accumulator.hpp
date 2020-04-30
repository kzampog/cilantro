#pragma once

#include <map>
#include <cilantro/config.hpp>
#include <cilantro/core/data_containers.hpp>

namespace cilantro {
    namespace internal {
        template <typename ScalarT, ptrdiff_t EigenDim, ptrdiff_t EigenCoeff>
        struct EigenVectorComparatorHelper {
            enum { coeff = EigenDim - EigenCoeff };

            template <typename VectorT1, typename VectorT2>
            static inline bool result(const VectorT1 &p1, const VectorT2 &p2) {
                if (p1[coeff] < p2[coeff]) return true;
                if (p2[coeff] < p1[coeff]) return false;
                return EigenVectorComparatorHelper<ScalarT,EigenDim,EigenCoeff-1>::result(p1, p2);
            }
        };

        template <typename ScalarT, ptrdiff_t EigenDim>
        struct EigenVectorComparatorHelper<ScalarT, EigenDim, 1> {
            enum { coeff = EigenDim - 1 };

            template <typename VectorT1, typename VectorT2>
            static inline bool result(const VectorT1 &p1, const VectorT2 &p2) {
                return p1[coeff] < p2[coeff];
            }
        };
    }

    template <typename ScalarT, ptrdiff_t EigenDim>
    struct EigenVectorComparator {
        template <typename VectorT1, typename VectorT2>
        inline bool operator()(const VectorT1 &p1, const VectorT2 &p2) const {
            return internal::EigenVectorComparatorHelper<ScalarT,EigenDim,EigenDim>::result(p1, p2);
        }
    };

    template <typename ScalarT>
    struct EigenVectorComparator<ScalarT, Eigen::Dynamic> {
        template <typename VectorT1, typename VectorT2>
        inline bool operator()(const VectorT1 &p1, const VectorT2 &p2) const {
            for (size_t i = 0; i < p1.rows() - 1; i++) {
                if (p1[i] < p2[i]) return true;
                if (p2[i] < p1[i]) return false;
            }
            return p1[p1.rows()-1] < p2[p1.rows()-1];
        }
    };

    template <typename ScalarT, ptrdiff_t EigenDim, class AccumulatorProxyT, typename GridPointScalarT = ptrdiff_t>
    class GridAccumulator {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum { Dimension = EigenDim };

        typedef typename AccumulatorProxyT::Accumulator Accumulator;

        typedef AccumulatorProxyT AccumulatorProxy;

        typedef GridPointScalarT GridPointScalar;

        typedef Eigen::Matrix<GridPointScalarT,EigenDim,1> GridPoint;

        typedef typename std::conditional<(EigenDim != Eigen::Dynamic && sizeof(GridPoint) % 16 == 0) || (Accumulator::EigenAlign > 0),
                std::map<GridPoint,Accumulator,EigenVectorComparator<typename GridPoint::Scalar,EigenDim>,Eigen::aligned_allocator<std::pair<const GridPoint,Accumulator>>>,
                std::map<GridPoint,Accumulator,EigenVectorComparator<typename GridPoint::Scalar,EigenDim>>>::type GridBinMap;

        typedef typename GridBinMap::iterator GridBinMapIterator;

        typedef typename GridBinMap::const_iterator GridBinMapConstIterator;

        GridAccumulator(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &data,
                        const Eigen::Ref<const Vector<ScalarT,EigenDim>> &bin_size,
                        const AccumulatorProxy &accum_proxy,
                        bool parallel = true)
                : data_map_(data),
                  bin_size_(bin_size),
                  bin_size_inv_(bin_size_.cwiseInverse())
        {
            build_index_(accum_proxy, parallel);
        }

        GridAccumulator(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &data,
                        ScalarT bin_size,
                        const AccumulatorProxy &accum_proxy,
                        bool parallel = true)
                : data_map_(data),
                  bin_size_(Vector<ScalarT,EigenDim>::Constant(data_map_.rows(), 1, bin_size)),
                  bin_size_inv_(bin_size_.cwiseInverse())
        {
            build_index_(accum_proxy, parallel);
        }

        ~GridAccumulator() {}

        inline const ConstVectorSetMatrixMap<ScalarT,EigenDim>& getPointsMatrixMap() const { return data_map_; }

        inline const Vector<ScalarT,EigenDim>& getBinSize() const { return bin_size_; }

        inline const GridBinMap& getOccupiedBinMap() const { return grid_lookup_table_; }

        inline const std::vector<GridBinMapIterator>& getOccupiedBinIterators() const { return bin_iterators_; }

        inline const GridBinMapConstIterator findContainingGridBin(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &point) const {
            return grid_lookup_table_.find(getPointGridCoordinates(point));
        }

        inline const GridBinMapConstIterator findContainingGridBin(size_t ind) const {
            return getPointBinNeighbors(data_map_.col(ind));
        }

        inline GridPoint getPointGridCoordinates(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &point) const {
            // return point.cwiseProduct(bin_size_inv_).array().floor().template cast<typename GridPoint::Scalar>();
            GridPoint grid_coords(data_map_.rows());
            for (size_t i = 0; i < data_map_.rows(); i++) {
                grid_coords[i] = std::floor(point[i]*bin_size_inv_[i]);
            }
            return grid_coords;
        }

        inline GridPoint getPointGridCoordinates(size_t point_ind) const {
            return getPointGridCoordinates(data_map_.col(point_ind));
        }

        inline Vector<ScalarT,EigenDim> getBinCornerCoordinates(const Eigen::Ref<const GridPoint> &grid_point) const {
            Vector<ScalarT,EigenDim> point(data_map_.rows());
            for (size_t i = 0; i < data_map_.rows(); i++) {
                point[i] = grid_point[i]*bin_size_[i];
            }
            return point;
        }

    protected:
        ConstVectorSetMatrixMap<ScalarT,EigenDim> data_map_;
        Vector<ScalarT,EigenDim> bin_size_;
        Vector<ScalarT,EigenDim> bin_size_inv_;

        GridBinMap grid_lookup_table_;
        std::vector<GridBinMapIterator> bin_iterators_;

        inline void build_index_(const AccumulatorProxy &accum_proxy, bool parallel) {
            if (data_map_.cols() == 0) return;

            if (parallel) {
#pragma omp parallel
                {
                    GridBinMap lookup_priv;

#pragma omp for nowait
                    for (size_t i = 0; i < data_map_.cols(); i++) {
                        GridPoint grid_coords = getPointGridCoordinates(data_map_.col(i));

                        auto lb = lookup_priv.lower_bound(grid_coords);
                        if (lb != lookup_priv.end() && !(lookup_priv.key_comp()(grid_coords, lb->first))) {
                            accum_proxy.addToAccumulator(lb->second, i);
                        } else {
                            lookup_priv.emplace_hint(lb, std::move(grid_coords), accum_proxy.buildAccumulator(i));
                        }
                    }

#pragma omp critical
                    {
                        for (auto it = lookup_priv.begin(); it != lookup_priv.end(); ++it) {
                            auto lb = grid_lookup_table_.lower_bound(it->first);
                            if (lb != grid_lookup_table_.end() && !(grid_lookup_table_.key_comp()(it->first, lb->first))) {
                                lb->second.mergeWith(it->second);
                            } else {
                                grid_lookup_table_.emplace_hint(lb, std::move(it->first), std::move(it->second));
                            }
                        }
                    }
                }

                bin_iterators_.resize(grid_lookup_table_.size());
                auto it = grid_lookup_table_.begin();
                for (size_t i = 0; i < bin_iterators_.size(); i++) {
                    bin_iterators_[i] = it++;
                }
            } else {
                for (size_t i = 0; i < data_map_.cols(); i++) {
                    GridPoint grid_coords = getPointGridCoordinates(data_map_.col(i));

                    auto lb = grid_lookup_table_.lower_bound(grid_coords);
                    if (lb != grid_lookup_table_.end() && !(grid_lookup_table_.key_comp()(grid_coords, lb->first))) {
                        accum_proxy.addToAccumulator(lb->second, i);
                    } else {
                        bin_iterators_.emplace_back(grid_lookup_table_.emplace_hint(lb, std::move(grid_coords), accum_proxy.buildAccumulator(i)));
                    }
                }
            }
        }
    };
}
