#pragma once

#include <algorithm>
#include <limits>
#include <map>

#include <cilantro/core/data_containers.hpp>
#include <cilantro/utilities/random.hpp>

namespace cilantro {

    template <typename ScalarT, ptrdiff_t EigenDim>
    class Covariance {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Covariance() = default;

        inline bool operator()(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, Vector<ScalarT,EigenDim>& mean, Eigen::Matrix<ScalarT,EigenDim,EigenDim>& cov) const {
            if (points.cols() < points.rows()) return false;
            mean = points.rowwise().mean();
            auto centered = points.rowwise() - mean;  // Lazy evaluation
            cov.noalias() =  (ScalarT)(1.0)/(points.cols() - 1) * (centered * centered.transpose());
            return true;
        }

        template <typename NeighborhoodResultIteratorT>
        inline bool operator()(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, NeighborhoodResultIteratorT begin, NeighborhoodResultIteratorT end, Vector<ScalarT,EigenDim>& mean, Eigen::Matrix<ScalarT,EigenDim,EigenDim>& cov) const {
            size_t size = std::distance(begin, end);
            if (size < points.rows()) return false;

            mean.setZero();
            for (NeighborhoodResultIteratorT it = begin; it != end; ++it) {
                mean += points.col(it->index);
            }
            mean *= (ScalarT)(1.0)/size;

            cov.setZero();
            for (NeighborhoodResultIteratorT it = begin; it != end; ++it) {
                auto tmp = points.col(it->index) - mean;
                cov += tmp*tmp.transpose();
            }
            cov *= (ScalarT)(1.0)/(size - 1);
            return true;
        }

        template <typename NeighborhoodResultT>
        inline bool operator()(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, const NeighborhoodResultT &nn, Vector<ScalarT,EigenDim>& mean, Eigen::Matrix<ScalarT,EigenDim,EigenDim>& cov) const {
            return (*this)(points, nn.begin(), nn.end(), mean, cov);
        }
    };

    template <typename ScalarT, ptrdiff_t EigenDim, typename CovarianceT=Covariance<ScalarT, EigenDim>>
    class MinimumCovarianceDeterminant {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        MinimumCovarianceDeterminant() = default;

        template <typename NeighborhoodResultIteratorT>
        inline bool operator()(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, NeighborhoodResultIteratorT begin, NeighborhoodResultIteratorT end, Vector<ScalarT,EigenDim>& mean, Eigen::Matrix<ScalarT,EigenDim,EigenDim>& cov) const {
            const int first_idx = begin->index;

            const size_t size = std::distance(begin, end);
            if (size < points.rows()) return false;
            if (size == points.rows()) return compute_mean_and_covariance_(points, begin, end, mean, cov);

            random_selector<> random {};

            using NeighborT = typename NeighborhoodResultIteratorT::value_type;

            std::vector<NeighborT> subset(points.rows() + 1);
            const size_t h = static_cast<size_t>(std::ceil(std::max((ScalarT)0.5, outlier_rate_) * (size + EigenDim + 1)));
            Vector<ScalarT,EigenDim> best_mean;
            Eigen::Matrix<ScalarT,EigenDim,EigenDim> best_cov;
            ScalarT best_determinant = std::numeric_limits<ScalarT>::max();
            for (int j = 0; j < num_trials_; ++j) {
                std::generate(subset.begin(), subset.end(), [&begin, &end, &random]() { return *random(begin, end); });
                compute_mean_and_covariance_(points, subset.begin(), subset.end(), mean, cov);
                for (int l = 0; l < num_refinements_; ++l) {
                    std::map<size_t, ScalarT> index_to_distance = mahalanobisDistance(points, begin, end, mean, cov.inverse());
                    std::partial_sort(begin, begin + h, end, [&index_to_distance](const NeighborT& a, const NeighborT& b) {
                        return index_to_distance[a.index] < index_to_distance[b.index];
                    });
                    compute_mean_and_covariance_(points, begin, begin + h, mean, cov);
                }
                ScalarT determinant = cov.determinant();
                if (determinant < best_determinant) {
                    best_determinant = determinant;
                    best_cov = cov;
                    best_mean = mean;
                }
            }
            mean = best_mean;
            cov = best_cov;
            if (chi_square_threshold_ <= 0) return true;
            auto demeaned = points.col(first_idx) - mean;
            return demeaned.transpose() * cov.inverse() * demeaned <= chi_square_threshold_;
        }

        template <typename NeighborhoodResultT>
        inline bool operator()(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, NeighborhoodResultT &nn, Vector<ScalarT,EigenDim>& mean, Eigen::Matrix<ScalarT,EigenDim,EigenDim>& cov) const {
            return (*this)(points, nn.begin(), nn.end(), mean, cov);
        }

        inline int getNumberOfTrials() const { return num_trials_; }

        inline MinimumCovarianceDeterminant& setNumberOfTrials(int num_trials) {
            num_trials_ = num_trials;
            return *this;
        }

        inline int getNumberOfRefinements() const { return num_refinements_; }

        inline MinimumCovarianceDeterminant& setNumberOfRefinements(int num_refinements) {
            num_refinements_ = num_refinements;
            return *this;
        }

        inline ScalarT getOutlierRate() const { return outlier_rate_; }

        inline MinimumCovarianceDeterminant& setOutlierRate(int outlier_rate) {
            outlier_rate_ = outlier_rate;
            return *this;
        }

        inline ScalarT getChiSquareThreshold() const { return chi_square_threshold_; }

        inline MinimumCovarianceDeterminant& setChiSquareThreshold(int chi_square_threshold) {
            chi_square_threshold_ = chi_square_threshold;
            return *this;
        }

    protected:
        template <typename NeighborhoodResultIteratorT>
        std::map<size_t, ScalarT> mahalanobisDistance(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, NeighborhoodResultIteratorT begin, NeighborhoodResultIteratorT end, const Vector<ScalarT,EigenDim> &mean, const Eigen::Matrix<ScalarT,EigenDim,EigenDim> &cov_inverse) const {
            std::map<size_t, ScalarT> index_to_distance;
            std::transform(begin, end, std::inserter(index_to_distance, index_to_distance.begin()), [&points, &mean, &cov_inverse](const typename NeighborhoodResultIteratorT::value_type &n) {
                auto demeaned = points.col(n.index) - mean;
                return std::make_pair(n.index, demeaned.transpose() * cov_inverse * demeaned);
            });
            return index_to_distance;
        }

        // The number of random trials to take:
        // Can be estimated as log(1 - P) / log(1 - (1 - e)^dim),
        // where P is the desired probability to find an outlier free set and e is the outlier rate.
        int num_trials_ = 6;
        int num_refinements_ = 3;
        ScalarT outlier_rate_ = (ScalarT)0.75;
        // If > 0, the covariance ellipse will be used to label the point as in/outlier.
        ScalarT chi_square_threshold_ = (ScalarT)-1;
        CovarianceT compute_mean_and_covariance_;
    };
}