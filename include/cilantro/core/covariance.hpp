#pragma once

#include <algorithm>
#include <iterator>
#include <limits>
#include <cilantro/config.hpp>
#include <cilantro/core/data_containers.hpp>
#include <cilantro/core/nearest_neighbors.hpp>
#include <cilantro/core/random.hpp>
#include <cilantro/core/openmp_reductions.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class Covariance {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum { Dimension = EigenDim };

        Covariance() = default;

        inline bool operator()(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, Vector<ScalarT,EigenDim>& mean, Eigen::Matrix<ScalarT,EigenDim,EigenDim>& cov, bool parallel = false) const {
            if (points.cols() < 2) {
                if (points.cols() == 0) {
                    mean.setConstant(points.rows(), 1, std::numeric_limits<ScalarT>::quiet_NaN());
                } else {
                    mean = points.rowwise().mean();
                }
                cov.setConstant(points.rows(), points.rows(), std::numeric_limits<ScalarT>::quiet_NaN());
                return false;
            }

            if (parallel) {
                Vector<ScalarT,EigenDim> mean_sum(Vector<ScalarT,EigenDim>::Zero(points.rows(), 1));
#ifdef ENABLE_NON_DETERMINISTIC_PARALLELISM
DECLARE_MATRIX_SUM_REDUCTION(ScalarT,EigenDim,1)
#pragma omp parallel for MATRIX_SUM_REDUCTION(ScalarT,EigenDim,1,mean_sum)
//#pragma omp parallel for reduction (internal::MatrixReductions<ScalarT,EigenDim,1>::operator+: mean_sum)
#endif
                for (size_t i = 0; i < points.cols(); i++) {
                    mean_sum.noalias() += points.col(i);
                }
                mean.noalias() = (ScalarT(1.0)/static_cast<ScalarT>(points.cols()))*mean_sum;

                Eigen::Matrix<ScalarT,EigenDim,EigenDim> cov_sum(Eigen::Matrix<ScalarT,EigenDim,EigenDim>::Zero(points.rows(), points.rows()));
#ifdef ENABLE_NON_DETERMINISTIC_PARALLELISM
DECLARE_MATRIX_SUM_REDUCTION(ScalarT,EigenDim,EigenDim)
#pragma omp parallel for MATRIX_SUM_REDUCTION(ScalarT,EigenDim,EigenDim,cov_sum)
//#pragma omp parallel for reduction (internal::MatrixReductions<ScalarT,EigenDim,EigenDim>::operator+: cov_sum)
#endif
                for (size_t i = 0; i < points.cols(); i++) {
                    Vector<ScalarT,EigenDim> tmp = points.col(i) - mean;
                    cov_sum.noalias() += tmp*tmp.transpose();
                }
                cov.noalias() = (ScalarT(1.0)/static_cast<ScalarT>(points.cols() - 1))*cov_sum;
            } else {
                Vector<ScalarT,EigenDim> mean_sum(Vector<ScalarT,EigenDim>::Zero(points.rows(), 1));
                for (size_t i = 0; i < points.cols(); i++) {
                    mean_sum.noalias() += points.col(i);
                }
                mean.noalias() = (ScalarT(1.0)/static_cast<ScalarT>(points.cols()))*mean_sum;

                Eigen::Matrix<ScalarT,EigenDim,EigenDim> cov_sum(Eigen::Matrix<ScalarT,EigenDim,EigenDim>::Zero(points.rows(), points.rows()));
                for (size_t i = 0; i < points.cols(); i++) {
                    Vector<ScalarT,EigenDim> tmp = points.col(i) - mean;
                    cov_sum.noalias() += tmp*tmp.transpose();
                }
                cov.noalias() = (ScalarT(1.0)/static_cast<ScalarT>(points.cols() - 1))*cov_sum;
            }

            return points.cols() >= points.rows();
        }

        // Conditionally enable parallelization if iterators are random access
        template <typename IteratorT, typename std::enable_if<std::is_same<typename std::iterator_traits<IteratorT>::iterator_category, std::random_access_iterator_tag>::value, int>::type = 0>
        inline bool operator()(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, IteratorT begin, IteratorT end, Vector<ScalarT,EigenDim>& mean, Eigen::Matrix<ScalarT,EigenDim,EigenDim>& cov, bool parallel = false) const {
            const size_t size = std::distance(begin, end);
            if (size < 2) {
                if (size == 0) {
                    mean.setConstant(points.rows(), 1, std::numeric_limits<ScalarT>::quiet_NaN());
                } else {
                    mean = points.rowwise().mean();
                }
                cov.setConstant(points.rows(), points.rows(), std::numeric_limits<ScalarT>::quiet_NaN());
                return false;
            }

            if (parallel) {
                Vector<ScalarT,EigenDim> mean_sum(Vector<ScalarT,EigenDim>::Zero(points.rows(), 1));
#ifdef ENABLE_NON_DETERMINISTIC_PARALLELISM
DECLARE_MATRIX_SUM_REDUCTION(ScalarT,EigenDim,1)
#pragma omp parallel for MATRIX_SUM_REDUCTION(ScalarT,EigenDim,1,mean_sum)
//#pragma omp parallel for reduction (internal::MatrixReductions<ScalarT,EigenDim,1>::operator+: mean_sum)
#endif
                for (IteratorT it = begin; it < end; ++it) {
                    mean_sum.noalias() += points.col(*it);
                }
                mean.noalias() = (ScalarT(1.0)/static_cast<ScalarT>(size))*mean_sum;

                Eigen::Matrix<ScalarT,EigenDim,EigenDim> cov_sum(Eigen::Matrix<ScalarT,EigenDim,EigenDim>::Zero(points.rows(), points.rows()));
#ifdef ENABLE_NON_DETERMINISTIC_PARALLELISM
DECLARE_MATRIX_SUM_REDUCTION(ScalarT,EigenDim,EigenDim)
#pragma omp parallel for MATRIX_SUM_REDUCTION(ScalarT,EigenDim,EigenDim,cov_sum)
//#pragma omp parallel for reduction (internal::MatrixReductions<ScalarT,EigenDim,EigenDim>::operator+: cov_sum)
#endif
                for (IteratorT it = begin; it < end; ++it) {
                    Vector<ScalarT,EigenDim> tmp = points.col(*it) - mean;
                    cov_sum.noalias() += tmp*tmp.transpose();
                }
                cov.noalias() = (ScalarT(1.0)/static_cast<ScalarT>(size - 1))*cov_sum;
            } else {
                Vector<ScalarT,EigenDim> mean_sum(Vector<ScalarT,EigenDim>::Zero(points.rows(), 1));
                for (IteratorT it = begin; it != end; ++it) {
                    mean_sum.noalias() += points.col(*it);
                }
                mean.noalias() = (ScalarT(1.0)/static_cast<ScalarT>(size))*mean_sum;

                Eigen::Matrix<ScalarT,EigenDim,EigenDim> cov_sum(Eigen::Matrix<ScalarT,EigenDim,EigenDim>::Zero(points.rows(), points.rows()));
                for (IteratorT it = begin; it != end; ++it) {
                    Vector<ScalarT,EigenDim> tmp = points.col(*it) - mean;
                    cov_sum.noalias() += tmp*tmp.transpose();
                }
                cov.noalias() = (ScalarT(1.0)/static_cast<ScalarT>(size - 1))*cov_sum;
            }

            return size >= points.rows();
        }

        // Non-random access iterators: this overload ignores the parallelization flag
        template <typename IteratorT, typename std::enable_if<std::is_same<typename std::iterator_traits<IteratorT>::iterator_category, std::random_access_iterator_tag>::value == false, int>::type = 0>
        inline bool operator()(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, IteratorT begin, IteratorT end, Vector<ScalarT,EigenDim>& mean, Eigen::Matrix<ScalarT,EigenDim,EigenDim>& cov, bool) const {
            const size_t size = std::distance(begin, end);
            if (size < 2) {
                if (size == 0) {
                    mean.setConstant(points.rows(), 1, std::numeric_limits<ScalarT>::quiet_NaN());
                } else {
                    mean = points.rowwise().mean();
                }
                cov.setConstant(points.rows(), points.rows(), std::numeric_limits<ScalarT>::quiet_NaN());
                return false;
            }

            Vector<ScalarT,EigenDim> mean_sum(Vector<ScalarT,EigenDim>::Zero(points.rows(), 1));
            for (IteratorT it = begin; it != end; ++it) {
                mean_sum.noalias() += points.col(*it);
            }
            mean.noalias() = (ScalarT(1.0)/static_cast<ScalarT>(size))*mean_sum;

            Eigen::Matrix<ScalarT,EigenDim,EigenDim> cov_sum(Eigen::Matrix<ScalarT,EigenDim,EigenDim>::Zero(points.rows(), points.rows()));
            for (IteratorT it = begin; it != end; ++it) {
                Vector<ScalarT,EigenDim> tmp = points.col(*it) - mean;
                cov_sum.noalias() += tmp*tmp.transpose();
            }
            cov.noalias() = (ScalarT(1.0)/static_cast<ScalarT>(size - 1))*cov_sum;

            return size >= points.rows();
        }

        template <typename ContainerT>
        inline bool operator()(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, const ContainerT &subset, Vector<ScalarT,EigenDim>& mean, Eigen::Matrix<ScalarT,EigenDim,EigenDim>& cov, bool parallel = false) const {
            return (*this)(points, subset.begin(), subset.end(), mean, cov, parallel);
        }
    };

    template <typename ScalarT, ptrdiff_t EigenDim, typename CovarianceT = Covariance<ScalarT, EigenDim>, typename RandomGeneratorT = std::default_random_engine>
    class MinimumCovarianceDeterminant {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum { Dimension = EigenDim };

        typedef CovarianceT Covariance;

        typedef RandomGeneratorT RandomGenerator;

        MinimumCovarianceDeterminant() = default;

        inline bool operator()(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, Vector<ScalarT,EigenDim>& mean, Eigen::Matrix<ScalarT,EigenDim,EigenDim>& cov, bool parallel = false) const {
            // Neighborhood<ScalarT> subset(points.cols());
            // size_t count = 0;
            // std::generate(subset.begin(), subset.end(), [&count]() mutable { return Neighbor<ScalarT>(count++, ScalarT(0.0)); });
            std::vector<size_t> subset(points.cols());
            for (size_t i = 0; i < subset.size(); i++) subset[i] = i;
            return (*this)(points, subset, mean, cov, parallel);
        }

        template <typename IteratorT>
        inline bool operator()(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, IteratorT begin, IteratorT end, Vector<ScalarT,EigenDim>& mean, Eigen::Matrix<ScalarT,EigenDim,EigenDim>& cov, bool parallel = false) const {
            const size_t size = std::distance(begin, end);
            if (size <= points.rows()) return compute_mean_and_covariance_(points, begin, end, mean, cov, false);

            Neighborhood<ScalarT> range_copy(size);
            size_t k = 0;
            for (auto it = begin; it != end; ++it) {
                // range_copy[k++].index = it->index;
                range_copy[k++].index = *it;
            }

            auto copy_begin = range_copy.begin();
            auto copy_end = range_copy.end();

            const auto first_idx = copy_begin->index;

            RandomElementSelector<RandomGeneratorT> random{};

            // Neighborhood<ScalarT> subset(points.rows());
            std::vector<size_t> subset(points.rows());
            size_t h = static_cast<size_t>(std::ceil(outlier_rate_ * (size + points.rows() + 1)));
            if (h > size) h = size - 1;
            Vector<ScalarT,EigenDim> best_mean;
            Eigen::Matrix<ScalarT,EigenDim,EigenDim> best_cov;
            ScalarT best_determinant = std::numeric_limits<ScalarT>::max();
            for (int j = 0; j < num_trials_; ++j) {
                std::generate(subset.begin(), subset.end(), [&copy_begin, &copy_end, &random]() { return *random(copy_begin, copy_end); });
                compute_mean_and_covariance_(points, subset.begin(), subset.end(), mean, cov, false);
                for (int l = 0; l < num_refinements_; ++l) {
                    mahalanobisDistance(points, copy_begin, copy_end, mean, cov.inverse());
                    std::partial_sort(copy_begin, copy_begin + h, copy_end, typename Neighbor<ScalarT>::ValueLessComparator());
                    compute_mean_and_covariance_(points, copy_begin, copy_begin + h, mean, cov, parallel);
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
            // mahalanobisDistance(points, copy_begin, copy_end, mean, cov.inverse()); // this looks redundant
            if (chi_square_threshold_ <= ScalarT(0.0)) return true;
            auto demeaned = points.col(first_idx) - mean;
            return demeaned.transpose() * cov.inverse() * demeaned <= chi_square_threshold_;
        }

        template <typename ContainerT>
        inline bool operator()(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, const ContainerT &subset, Vector<ScalarT,EigenDim>& mean, Eigen::Matrix<ScalarT,EigenDim,EigenDim>& cov, bool parallel = false) const {
            return (*this)(points, subset.begin(), subset.end(), mean, cov, parallel);
        }

        inline const Covariance& evaluator() const { return compute_mean_and_covariance_; }

        inline Covariance& evaluator() { return compute_mean_and_covariance_; }

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

        inline MinimumCovarianceDeterminant& setOutlierRate(ScalarT outlier_rate) {
            outlier_rate_ = std::max(ScalarT(0.5), outlier_rate);
            return *this;
        }

        inline ScalarT getChiSquareThreshold() const { return chi_square_threshold_; }

        inline MinimumCovarianceDeterminant& setChiSquareThreshold(ScalarT chi_square_threshold) {
            chi_square_threshold_ = chi_square_threshold;
            return *this;
        }

    protected:
        template <typename NeighborhoodResultIteratorT>
        inline void mahalanobisDistance(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, NeighborhoodResultIteratorT begin, NeighborhoodResultIteratorT end, const Vector<ScalarT,EigenDim> &mean, const Eigen::Matrix<ScalarT,EigenDim,EigenDim> &cov_inverse) const {
            std::for_each(begin, end, [&points, &mean, &cov_inverse](typename NeighborhoodResultIteratorT::value_type &n) {
                auto demeaned = points.col(n.index) - mean;
                n.value = demeaned.transpose() * cov_inverse * demeaned;
            });
        }

        // The number of random trials to take:
        // Can be estimated as log(1 - P) / log(1 - (1 - e)^dim),
        // where P is the desired probability to find an outlier free set and e is the outlier rate.
        int num_trials_ = 6;
        int num_refinements_ = 3;
        ScalarT outlier_rate_ = ScalarT(0.75);
        // If > 0, the covariance ellipse will be used to label the point as in/outlier.
        ScalarT chi_square_threshold_ = ScalarT(-1);
        CovarianceT compute_mean_and_covariance_;
    };
}
