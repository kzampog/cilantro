#pragma once

#include <cilantro/core/space_transformations.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class PointFeaturesAdaptor {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum { FeatureDimension = EigenDim };

        PointFeaturesAdaptor(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points)
                : data_map_(points),
                  transformed_data_(points.rows(), points.cols()),
                  transformed_data_map_(transformed_data_)
        {}

        PointFeaturesAdaptor& transformFeatures() {
#pragma omp parallel for
            for (size_t i = 0; i < data_map_.cols(); i++) {
                transformed_data_.col(i) = data_map_.col(i);
            }
            return *this;
        }

        template <class TransformT>
        inline PointFeaturesAdaptor& transformFeatures(const TransformT &tform) {
#pragma omp parallel for
            for (size_t i = 0; i < data_map_.cols(); i++) {
                transformed_data_.col(i).noalias() = tform.linear()*data_map_.col(i) + tform.translation();
            }
            return *this;
        }

        template <class TransformT>
        inline PointFeaturesAdaptor& transformFeatures(const TransformSet<TransformT> &tforms) {
#pragma omp parallel for
            for (size_t i = 0; i < data_map_.cols(); i++) {
                transformed_data_.col(i).noalias() = tforms[i]*data_map_.col(i);
            }
            return *this;
        }

        inline const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getFeaturesMatrixMap() const {
            return data_map_;
        }

        inline const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getTransformedFeaturesMatrixMap() const {
            return transformed_data_map_;
        }

    protected:
        ConstVectorSetMatrixMap<ScalarT,FeatureDimension> data_map_;
        VectorSet<ScalarT,FeatureDimension> transformed_data_;
        ConstVectorSetMatrixMap<ScalarT,FeatureDimension> transformed_data_map_;
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    class PointNormalFeaturesAdaptor {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum { FeatureDimension = (EigenDim == Eigen::Dynamic) ? Eigen::Dynamic : 2*EigenDim };

        PointNormalFeaturesAdaptor(const ConstVectorSetMatrixMap<ScalarT,FeatureDimension> &data)
                : data_map_(data),
                  transformed_data_(data.rows(), data.cols()),
                  transformed_data_map_(transformed_data_)
        {}

        PointNormalFeaturesAdaptor(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                   const ConstVectorSetMatrixMap<ScalarT,EigenDim> &normals,
                                   ScalarT normal_weight)
                : data_(2*points.rows(), points.cols()),
                  data_map_(data_),
                  transformed_data_(data_.rows(), data_.cols()),
                  transformed_data_map_(transformed_data_)
        {
            data_.topRows(points.rows()) = points;
            data_.bottomRows(normals.rows()) = normal_weight*normals;
        }

        PointNormalFeaturesAdaptor& transformFeatures() {
#pragma omp parallel for
            for (size_t i = 0; i < data_map_.cols(); i++) {
                transformed_data_.col(i) = data_map_.col(i);
            }
            return *this;
        }

        template <class TransformT>
        PointNormalFeaturesAdaptor& transformFeatures(const TransformT &tform) {
            const size_t dim = data_map_.rows()/2;
            if (int(TransformT::Mode) == int(Eigen::Isometry)) {
#pragma omp parallel for
                for (size_t i = 0; i < data_map_.cols(); i++) {
                    auto res_col = transformed_data_.col(i);
                    auto data_col = data_map_.col(i);
                    res_col.head(dim).noalias() = tform.linear()*data_col.head(dim) + tform.translation();
                    res_col.tail(dim).noalias() = tform.linear()*data_col.tail(dim);
                }
            } else {
                const ScalarT normal_weight = (data_map_.cols() > 0) ? data_map_.template block<3,1>(dim,0).norm() : (ScalarT)0.0;
#pragma omp parallel for
                for (size_t i = 0; i < data_map_.cols(); i++) {
                    auto res_col = transformed_data_.col(i);
                    auto data_col = data_map_.col(i);
                    res_col.head(dim).noalias() = tform.linear()*data_col.head(dim) + tform.translation();
                    res_col.tail(dim).noalias() = normal_weight*(tform.linear().inverse().transpose()*data_col.tail(dim)).normalized();
                }
            }
            return *this;
        }

        template <class TransformT>
        PointNormalFeaturesAdaptor& transformFeatures(const TransformSet<TransformT> &tforms) {
            const size_t dim = data_map_.rows()/2;
            if (int(TransformT::Mode) == int(Eigen::Isometry)) {
#pragma omp parallel for
                for (size_t i = 0; i < data_map_.cols(); i++) {
                    auto res_col = transformed_data_.col(i);
                    auto data_col = data_map_.col(i);
                    res_col.head(dim).noalias() = tforms[i].linear()*data_col.head(dim) + tforms[i].translation();
                    res_col.tail(dim).noalias() = tforms[i].linear()*data_col.tail(dim);
                }
            } else {
                const ScalarT normal_weight = (data_map_.cols() > 0) ? data_map_.template block<3,1>(dim,0).norm() : (ScalarT)0.0;
#pragma omp parallel for
                for (size_t i = 0; i < data_map_.cols(); i++) {
                    auto res_col = transformed_data_.col(i);
                    auto data_col = data_map_.col(i);
                    res_col.head(dim).noalias() = tforms[i].linear()*data_col.head(dim) + tforms[i].translation();
                    res_col.tail(dim).noalias() = normal_weight*(tforms[i].linear().inverse().transpose()*data_col.tail(dim)).normalized();
                }
            }
            return *this;
        }

        inline const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getFeaturesMatrixMap() const {
            return data_map_;
        }

        inline const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getTransformedFeaturesMatrixMap() const {
            return transformed_data_map_;
        }

    protected:
        VectorSet<ScalarT,FeatureDimension> data_;
        ConstVectorSetMatrixMap<ScalarT,FeatureDimension> data_map_;
        VectorSet<ScalarT,FeatureDimension> transformed_data_;
        ConstVectorSetMatrixMap<ScalarT,FeatureDimension> transformed_data_map_;
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    class PointColorFeaturesAdaptor {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum { FeatureDimension = (EigenDim == Eigen::Dynamic) ? Eigen::Dynamic : EigenDim + 3 };

        PointColorFeaturesAdaptor(const ConstVectorSetMatrixMap<ScalarT,FeatureDimension> &data)
                : data_map_(data),
                  transformed_data_(data.rows(), data.cols()),
                  transformed_data_map_(transformed_data_)
        {}

        PointColorFeaturesAdaptor(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                  const ConstVectorSetMatrixMap<float,3> &colors,
                                  ScalarT color_weight)
                : data_(points.rows()+3, points.cols()),
                  data_map_(data_),
                  transformed_data_(data_.rows(), data_.cols()),
                  transformed_data_map_(transformed_data_)
        {
            data_.topRows(points.rows()) = points;
            data_.bottomRows(3) = color_weight*colors.template cast<ScalarT>();
        }

        PointColorFeaturesAdaptor& transformFeatures() {
#pragma omp parallel for
            for (size_t i = 0; i < data_map_.cols(); i++) {
                transformed_data_.col(i) = data_map_.col(i);
            }
            return *this;
        }

        template <class TransformT>
        PointColorFeaturesAdaptor& transformFeatures(const TransformT &tform) {
            const size_t dim = data_map_.rows() - 3;
#pragma omp parallel for
            for (size_t i = 0; i < data_map_.cols(); i++) {
                auto res_col = transformed_data_.col(i);
                auto data_col = data_map_.col(i);
                res_col.head(dim).noalias() = tform.linear()*data_col.head(dim) + tform.translation();
                res_col.tail(3) = data_col.tail(3);
            }
            return *this;
        }

        template <class TransformT>
        PointColorFeaturesAdaptor& transformFeatures(const TransformSet<TransformT> &tforms) {
            const size_t dim = data_map_.rows() - 3;
#pragma omp parallel for
            for (size_t i = 0; i < data_map_.cols(); i++) {
                auto res_col = transformed_data_.col(i);
                auto data_col = data_map_.col(i);
                res_col.head(dim).noalias() = tforms[i].linear()*data_col.head(dim) + tforms[i].translation();
                res_col.tail(3) = data_col.tail(3);
            }
            return *this;
        }

        inline const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getFeaturesMatrixMap() const {
            return data_map_;
        }

        inline const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getTransformedFeaturesMatrixMap() const {
            return transformed_data_map_;
        }

    protected:
        VectorSet<ScalarT,FeatureDimension> data_;
        ConstVectorSetMatrixMap<ScalarT,FeatureDimension> data_map_;
        VectorSet<ScalarT,FeatureDimension> transformed_data_;
        ConstVectorSetMatrixMap<ScalarT,FeatureDimension> transformed_data_map_;
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    class PointNormalColorFeaturesAdaptor {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum { FeatureDimension = (EigenDim == Eigen::Dynamic) ? Eigen::Dynamic : 2*EigenDim + 3 };

        PointNormalColorFeaturesAdaptor(const ConstVectorSetMatrixMap<ScalarT,FeatureDimension> &data)
                : data_map_(data),
                  transformed_data_(data.rows(), data.cols()),
                  transformed_data_map_(transformed_data_)
        {}

        PointNormalColorFeaturesAdaptor(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                        const ConstVectorSetMatrixMap<ScalarT,EigenDim> &normals,
                                        const ConstVectorSetMatrixMap<float,3> &colors,
                                        ScalarT normal_weight, ScalarT color_weight)
                : data_(2*points.rows()+3, points.cols()),
                  data_map_(data_),
                  transformed_data_(data_.rows(), data_.cols()),
                  transformed_data_map_(transformed_data_)
        {
            data_.topRows(points.rows()) = points;
            data_.block(points.rows(),0,normals.rows(),normals.cols()) = normal_weight*normals;
            data_.bottomRows(3) = color_weight*colors.template cast<ScalarT>();
        }

        PointNormalColorFeaturesAdaptor& transformFeatures() {
#pragma omp parallel for
            for (size_t i = 0; i < data_map_.cols(); i++) {
                transformed_data_.col(i) = data_map_.col(i);
            }
            return *this;
        }

        template <class TransformT>
        PointNormalColorFeaturesAdaptor& transformFeatures(const TransformT &tform) {
            const size_t dim = (data_map_.rows() - 3)/2;
            if (int(TransformT::Mode) == int(Eigen::Isometry)) {
#pragma omp parallel for
                for (size_t i = 0; i < data_map_.cols(); i++) {
                    auto res_col = transformed_data_.col(i);
                    auto data_col = data_map_.col(i);
                    res_col.head(dim).noalias() = tform.linear()*data_col.head(dim) + tform.translation();
                    res_col.segment(dim,dim).noalias() = tform.linear()*data_col.segment(dim,dim);
                    res_col.tail(3) = data_col.tail(3);
                }
            } else {
                const ScalarT normal_weight = (data_map_.cols() > 0) ? data_map_.template block<3,1>(dim,0).norm() : (ScalarT)0.0;
#pragma omp parallel for
                for (size_t i = 0; i < data_map_.cols(); i++) {
                    auto res_col = transformed_data_.col(i);
                    auto data_col = data_map_.col(i);
                    res_col.head(dim).noalias() = tform.linear()*data_col.head(dim) + tform.translation();
                    res_col.segment(dim,dim).noalias() = normal_weight*(tform.linear().inverse().transpose()*data_col.segment(dim,dim)).normalized();
                    res_col.tail(3) = data_col.tail(3);
                }
            }
            return *this;
        }

        template <class TransformT>
        PointNormalColorFeaturesAdaptor& transformFeatures(const TransformSet<TransformT> &tforms) {
            const size_t dim = (data_map_.rows() - 3)/2;
            if (int(TransformT::Mode) == int(Eigen::Isometry)) {
#pragma omp parallel for
                for (size_t i = 0; i < data_map_.cols(); i++) {
                    auto res_col = transformed_data_.col(i);
                    auto data_col = data_map_.col(i);
                    res_col.head(dim).noalias() = tforms[i].linear()*data_col.head(dim) + tforms[i].translation();
                    res_col.segment(dim,dim).noalias() = tforms[i].linear()*data_col.segment(dim,dim);
                    res_col.tail(3) = data_col.tail(3);
                }
            } else {
                const ScalarT normal_weight = (data_map_.cols() > 0) ? data_map_.template block<3,1>(dim,0).norm() : (ScalarT)0.0;
#pragma omp parallel for
                for (size_t i = 0; i < data_map_.cols(); i++) {
                    auto res_col = transformed_data_.col(i);
                    auto data_col = data_map_.col(i);
                    res_col.head(dim).noalias() = tforms[i].linear()*data_col.head(dim) + tforms[i].translation();
                    res_col.segment(dim,dim).noalias() = normal_weight*(tforms[i].linear().inverse().transpose()*data_col.segment(dim,dim)).normalized();
                    res_col.tail(3) = data_col.tail(3);
                }
            }
            return *this;
        }

        inline const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getFeaturesMatrixMap() const {
            return data_map_;
        }

        inline const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getTransformedFeaturesMatrixMap() const {
            return transformed_data_map_;
        }

    protected:
        VectorSet<ScalarT,FeatureDimension> data_;
        ConstVectorSetMatrixMap<ScalarT,FeatureDimension> data_map_;
        VectorSet<ScalarT,FeatureDimension> transformed_data_;
        ConstVectorSetMatrixMap<ScalarT,FeatureDimension> transformed_data_map_;
    };

    typedef PointFeaturesAdaptor<float,2> PointFeaturesAdaptor2f;
    typedef PointFeaturesAdaptor<double,2> PointFeaturesAdaptor2d;
    typedef PointFeaturesAdaptor<float,3> PointFeaturesAdaptor3f;
    typedef PointFeaturesAdaptor<double,3> PointFeaturesAdaptor3d;

    typedef PointNormalFeaturesAdaptor<float,2> PointNormalFeaturesAdaptor2f;
    typedef PointNormalFeaturesAdaptor<double,2> PointNormalFeaturesAdaptor2d;
    typedef PointNormalFeaturesAdaptor<float,3> PointNormalFeaturesAdaptor3f;
    typedef PointNormalFeaturesAdaptor<double,3> PointNormalFeaturesAdaptor3d;

    typedef PointColorFeaturesAdaptor<float,2> PointColorFeaturesAdaptor2f;
    typedef PointColorFeaturesAdaptor<double,2> PointColorFeaturesAdaptor2d;
    typedef PointColorFeaturesAdaptor<float,3> PointColorFeaturesAdaptor3f;
    typedef PointColorFeaturesAdaptor<double,3> PointColorFeaturesAdaptor3d;

    typedef PointNormalColorFeaturesAdaptor<float,2> PointNormalColorFeaturesAdaptor2f;
    typedef PointNormalColorFeaturesAdaptor<double,2> PointNormalColorFeaturesAdaptor2d;
    typedef PointNormalColorFeaturesAdaptor<float,3> PointNormalColorFeaturesAdaptor3f;
    typedef PointNormalColorFeaturesAdaptor<double,3> PointNormalColorFeaturesAdaptor3d;
}
