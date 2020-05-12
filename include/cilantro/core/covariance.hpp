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

        inline size_t getMinValidSampleSize() const { return min_sample_size_; }

        inline Covariance& setMinValidSampleSize(size_t min_size) {
            min_sample_size_ = min_size;
            return *this;
        }

        inline bool operator()(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, Vector<ScalarT,EigenDim>& mean, Eigen::Matrix<ScalarT,EigenDim,EigenDim>& cov, bool parallel = false) const {
            if (points.cols() < min_sample_size_) {
                mean.setConstant(points.rows(), 1, std::numeric_limits<ScalarT>::quiet_NaN());
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

            return true;
        }

        // Conditionally enable parallelization if iterators are random access
        template <typename IteratorT, typename std::enable_if<std::is_same<typename std::iterator_traits<IteratorT>::iterator_category, std::random_access_iterator_tag>::value, int>::type = 0>
        inline bool operator()(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, IteratorT begin, IteratorT end, Vector<ScalarT,EigenDim>& mean, Eigen::Matrix<ScalarT,EigenDim,EigenDim>& cov, bool parallel = false) const {
            const size_t size = std::distance(begin, end);
            if (size < min_sample_size_) {
                mean.setConstant(points.rows(), 1, std::numeric_limits<ScalarT>::quiet_NaN());
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

            return true;
        }

        // Non-random access iterators: this overload ignores the parallelization flag
        template <typename IteratorT, typename std::enable_if<std::is_same<typename std::iterator_traits<IteratorT>::iterator_category, std::random_access_iterator_tag>::value == false, int>::type = 0>
        inline bool operator()(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, IteratorT begin, IteratorT end, Vector<ScalarT,EigenDim>& mean, Eigen::Matrix<ScalarT,EigenDim,EigenDim>& cov, bool) const {
            const size_t size = std::distance(begin, end);
            if (size < min_sample_size_) {
                mean.setConstant(points.rows(), 1, std::numeric_limits<ScalarT>::quiet_NaN());
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

            return true;
        }

        template <typename ContainerT>
        inline bool operator()(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, const ContainerT &subset, Vector<ScalarT,EigenDim>& mean, Eigen::Matrix<ScalarT,EigenDim,EigenDim>& cov, bool parallel = false) const {
            return (*this)(points, subset.begin(), subset.end(), mean, cov, parallel);
        }

    protected:
        size_t min_sample_size_ = 2;
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
            if (points.cols() <= compute_mean_and_covariance_.getMinValidSampleSize()) return compute_mean_and_covariance_(points, mean, cov, false);

            Neighborhood<ScalarT> range_copy(points.cols());
            for (size_t i = 0; i < points.cols(); i++) {
                range_copy[i].index = i;
            }

            return computeOnMutableNeighborhood(points, range_copy.begin(), range_copy.end(), mean, cov, parallel);
        }

        template <typename IteratorT>
        inline bool operator()(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, IteratorT begin, IteratorT end, Vector<ScalarT,EigenDim>& mean, Eigen::Matrix<ScalarT,EigenDim,EigenDim>& cov, bool parallel = false) const {
            const size_t size = std::distance(begin, end);
            if (size <= compute_mean_and_covariance_.getMinValidSampleSize()) return compute_mean_and_covariance_(points, begin, end, mean, cov, false);

            Neighborhood<ScalarT> range_copy(size);
            size_t k = 0;
            for (auto it = begin; it != end; ++it) {
                range_copy[k++].index = *it;
            }

            return computeOnMutableNeighborhood(points, range_copy.begin(), range_copy.end(), mean, cov, parallel);
        }

        template <typename ContainerT>
        inline bool operator()(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, const ContainerT &subset, Vector<ScalarT,EigenDim>& mean, Eigen::Matrix<ScalarT,EigenDim,EigenDim>& cov, bool parallel = false) const {
            return (*this)(points, subset.begin(), subset.end(), mean, cov, parallel);
        }

        inline const Covariance& evaluator() const { return compute_mean_and_covariance_; }

        inline Covariance& evaluator() { return compute_mean_and_covariance_; }

        inline size_t getMinValidSampleSize() const { return compute_mean_and_covariance_.getMinValidSampleSize(); }

        inline MinimumCovarianceDeterminant& setMinValidSampleSize(size_t min_size) {
            compute_mean_and_covariance_.setMinValidSampleSize(min_size);
            return *this;
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

        inline ScalarT getInlierRatio() const { return inlier_ratio_; }

        inline MinimumCovarianceDeterminant& setInlierRatio(ScalarT inlier_ratio) {
            // inlier_ratio_ = std::max(ScalarT(0.5), inlier_ratio);
            inlier_ratio_ = inlier_ratio;
            return *this;
        }

        inline ScalarT getChiSquareThreshold() const { return chi_square_threshold_; }

        inline MinimumCovarianceDeterminant& setChiSquareThreshold(ScalarT chi_square_threshold) {
            chi_square_threshold_ = chi_square_threshold;
            return *this;
        }

    protected:
        template <typename NeighborhoodIteratorT>
        inline void mahalanobisDistance(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, NeighborhoodIteratorT begin, NeighborhoodIteratorT end, const Vector<ScalarT,EigenDim> &mean, const Eigen::Matrix<ScalarT,EigenDim,EigenDim> &cov_inverse, bool parallel) const {
            if (parallel) {
#pragma omp parallel for
                for (NeighborhoodIteratorT it = begin; it < end; ++it) {
                    Vector<ScalarT,EigenDim> demeaned = points.col(it->index) - mean;
                    it->value = demeaned.transpose() * cov_inverse * demeaned;
                }
            } else {
                for (NeighborhoodIteratorT it = begin; it != end; ++it) {
                    Vector<ScalarT,EigenDim> demeaned = points.col(it->index) - mean;
                    it->value = demeaned.transpose() * cov_inverse * demeaned;
                }
            }
        }

        template <typename NeighborhoodIteratorT>
        inline bool computeOnMutableNeighborhood(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points, NeighborhoodIteratorT begin, NeighborhoodIteratorT end, Vector<ScalarT,EigenDim>& mean, Eigen::Matrix<ScalarT,EigenDim,EigenDim>& cov, bool parallel) const {
            const size_t size = std::distance(begin, end);
            const auto first_idx = begin->index;

            // size_t h = static_cast<size_t>(std::ceil(inlier_ratio_ * (size + points.rows() + 1)));
            // if (h > size) h = size - 1;
            const size_t h = std::min(std::max(compute_mean_and_covariance_.getMinValidSampleSize(), static_cast<size_t>(std::llround(inlier_ratio_*size))), size);

            if (h < size) {
                RandomElementSelector<RandomGeneratorT> random{};
                std::vector<size_t> subset(compute_mean_and_covariance_.getMinValidSampleSize());

                Vector<ScalarT,EigenDim> best_mean;
                Eigen::Matrix<ScalarT,EigenDim,EigenDim> best_cov;
                ScalarT best_determinant = std::numeric_limits<ScalarT>::max();

                for (int j = 0; j < num_trials_; ++j) {
                    std::generate(subset.begin(), subset.end(), [&begin, &end, &random]() { return *random(begin, end); });
                    compute_mean_and_covariance_(points, subset.begin(), subset.end(), mean, cov, false);
                    for (int l = 0; l < num_refinements_; ++l) {
                        mahalanobisDistance(points, begin, end, mean, cov.inverse(), parallel);
                        std::partial_sort(begin, begin + h, end, typename Neighbor<ScalarT>::ValueLessComparator());
                        compute_mean_and_covariance_(points, begin, begin + h, mean, cov, parallel);
                    }
                    ScalarT determinant = cov.determinant();
                    if (determinant < best_determinant) {
                        best_mean = mean;
                        best_cov = cov;
                        best_determinant = determinant;
                    }
                }

                mean = best_mean;
                cov = best_cov;
            } else {
                // h == size
                compute_mean_and_covariance_(points, begin, end, mean, cov, parallel);
            }

            // mahalanobisDistance(points, begin, end, mean, cov.inverse(), parallel);   // this looks redundant
            if (chi_square_threshold_ <= ScalarT(0.0)) return true;

            Vector<ScalarT,EigenDim> demeaned = points.col(first_idx) - mean;
            return demeaned.transpose() * cov.inverse() * demeaned <= chi_square_threshold_;
        }

        // The number of random trials to take:
        // Can be estimated as log(1 - P) / log(1 - (1 - e)^dim),
        // where P is the desired probability to find an outlier free set and e is the outlier rate.
        int num_trials_ = 6;
        int num_refinements_ = 3;
        ScalarT inlier_ratio_ = ScalarT(0.75);
        // If > 0, the covariance ellipse will be used to label the point as in/outlier.
        ScalarT chi_square_threshold_ = ScalarT(-1);
        CovarianceT compute_mean_and_covariance_;
    };
}
