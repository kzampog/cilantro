#pragma once

#include <map>
#include <cilantro/data_matrix_map.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim, ptrdiff_t EigenCoeff>
    struct EigenVectorComparatorHelper {
        static bool result(const Eigen::Matrix<ScalarT,EigenDim,1> &p1, const Eigen::Matrix<ScalarT,EigenDim,1> &p2) {
            enum { coeff = EigenDim - EigenCoeff };
            if (p1[coeff] < p2[coeff]) return true;
            if (p2[coeff] < p1[coeff]) return false;
            return EigenVectorComparatorHelper<ScalarT,EigenDim,EigenCoeff-1>::result (p1, p2);
//                if (p1[coeff] < p2[coeff]) {
//                    return true;
//                } else {
//                    if (p2[coeff] < p1[coeff]) {
//                        return false;
//                    } else {
//                        return EigenVectorComparatorHelper_<ScalarT,EigenDim,EigenCoeff-1>::result (p1, p2);
//                    }
//                }
        }
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    struct EigenVectorComparatorHelper<ScalarT, EigenDim, 1> {
        static bool result(const Eigen::Matrix<ScalarT,EigenDim,1> &p1, const Eigen::Matrix<ScalarT, EigenDim, 1> &p2) {
            enum { coeff = EigenDim - 1 };
            return p1[coeff] < p2[coeff];
        }
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    struct EigenVectorComparator {
        inline bool operator()(const Eigen::Matrix<ScalarT,EigenDim,1> &p1, const Eigen::Matrix<ScalarT,EigenDim,1> &p2) const {
            return EigenVectorComparatorHelper<ScalarT,EigenDim,EigenDim>::result(p1, p2);
        }
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    class CartesianGrid {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        CartesianGrid(const ConstDataMatrixMap<ScalarT,EigenDim> &data, const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,1> > &bin_size)
                : data_map_(data),
                  bin_size_(bin_size)
        {
            build_index_();
        }

        CartesianGrid(const ConstDataMatrixMap<ScalarT,EigenDim> &data, ScalarT bin_size)
                : data_map_(data),
                  bin_size_(Eigen::Matrix<ScalarT,EigenDim,1>::Constant(bin_size))
        {
            build_index_();
        }

        ~CartesianGrid() {}

        const Eigen::Matrix<ScalarT,EigenDim,1>& getBinSize() const { return bin_size_; }

        const std::map<Eigen::Matrix<ptrdiff_t,EigenDim,1>,std::vector<size_t>,EigenVectorComparator<ptrdiff_t,EigenDim> >& getOccupiedBinMap() const { return grid_lookup_table_; }

        const std::vector<typename std::map<Eigen::Matrix<ptrdiff_t,EigenDim,1>,std::vector<size_t>,EigenVectorComparator<ptrdiff_t,EigenDim> >::iterator>& getOccupiedBinIterators() const { return bin_iterators_; }

        const std::vector<size_t>& getPointBinNeighbors(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,1> > &point) const {
            Eigen::Matrix<ptrdiff_t,EigenDim,1> grid_coords;
            for (size_t i = 0; i < EigenDim; i++) {
                grid_coords[i] = std::floor(point[i]/bin_size_[i]);
            }
            auto it = grid_lookup_table_.find(grid_coords);
            if (it == grid_lookup_table_.end()) return empty_set_of_indices_;
            return it->second;
        }

        const std::vector<size_t>& getPointBinNeighbors(size_t point_ind) const {
            return getPointBinNeighbors(data_map_.col(point_ind));
        }

        Eigen::Matrix<ptrdiff_t,EigenDim,1> getPointGridCoordinates(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,1> > &point) const {
            Eigen::Matrix<ptrdiff_t,EigenDim,1> grid_coords;
            for (size_t i = 0; i < EigenDim; i++) {
                grid_coords[i] = std::floor(point[i]/bin_size_[i]);
            }
            return grid_coords;
        }

        Eigen::Matrix<ptrdiff_t,EigenDim,1> getPointGridCoordinates(size_t point_ind) const {
            return getGridCoordinates(data_map_.col(point_ind));
        }

        Eigen::Matrix<ScalarT,EigenDim,1> getBinCornerCoordinates(const Eigen::Ref<const Eigen::Matrix<ptrdiff_t,EigenDim,1> > &grid_point) const {
            Eigen::Matrix<ScalarT,EigenDim,1> point;
            for (size_t i = 0; i < EigenDim; i++) {
                point[i] = grid_point[i]*bin_size_[i];
            }
            return point;
        }

    protected:
        static std::vector<size_t> empty_set_of_indices_;

        ConstDataMatrixMap<ScalarT,EigenDim> data_map_;
        Eigen::Matrix<ScalarT,EigenDim,1> bin_size_;

        std::map<Eigen::Matrix<ptrdiff_t,EigenDim,1>,std::vector<size_t>,EigenVectorComparator<ptrdiff_t,EigenDim> > grid_lookup_table_;
        std::vector<typename std::map<Eigen::Matrix<ptrdiff_t,EigenDim,1>,std::vector<size_t>,EigenVectorComparator<ptrdiff_t,EigenDim> >::iterator> bin_iterators_;

        void build_index_() {
            if (data_map_.cols() == 0) return;

            bin_iterators_.reserve(data_map_.cols());
            Eigen::Matrix<ptrdiff_t,EigenDim,1> grid_coords;
            for (size_t i = 0; i < data_map_.cols(); i++) {
                for (size_t j = 0; j < EigenDim; j++) {
//                    ScalarT val = data_map_(j,i)/bin_size_[j];
//                    grid_coords[j] = data_map_(j,i)/bin_size_[j];
//                    if (grid_coords[j] > val) grid_coords[j]--;
                    grid_coords[j] = std::floor(data_map_(j,i)/bin_size_[j]);
                }
                auto lb = grid_lookup_table_.lower_bound(grid_coords);
                if (lb != grid_lookup_table_.end() && !(grid_lookup_table_.key_comp()(grid_coords, lb->first))) {
                    lb->second.emplace_back(i);
                } else {
                    bin_iterators_.emplace_back(grid_lookup_table_.emplace_hint(lb, grid_coords, std::vector<size_t>(1, i)));
                }
            }
        }
    };

    typedef CartesianGrid<float,2> CartesianGrid2D;
    typedef CartesianGrid<float,3> CartesianGrid3D;
}
