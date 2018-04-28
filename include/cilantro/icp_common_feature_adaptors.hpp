#pragma once

#include <cilantro/space_transformations.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class PointFeaturesAdaptor {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum {FeatureDimension = EigenDim};

        PointFeaturesAdaptor(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points)
                : data_map_(points),
                  transformed_data_(points.rows(), points.cols()),
                  transformed_data_map_(transformed_data_)
        {}

        inline const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getFeatureData() const { return data_map_; }

        inline const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getTransformedFeatureData() const { return transformed_data_map_; }

        inline const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getTransformedFeatureData(const RigidTransformation<ScalarT,EigenDim> &tform) {
#pragma omp parallel for
            for (size_t i = 0; i < data_map_.cols(); i++) {
                transformed_data_.col(i) = tform*data_map_.col(i);
            }
            return transformed_data_map_;
        }

        const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getTransformedFeatureData(const RigidTransformationSet<ScalarT,EigenDim> &tforms) {
#pragma omp parallel for
            for (size_t i = 0; i < data_map_.cols(); i++) {
                transformed_data_.col(i) = tforms[i]*data_map_.col(i);
            }
            return transformed_data_map_;
        }

    private:
        ConstVectorSetMatrixMap<ScalarT,FeatureDimension> data_map_;
        VectorSet<ScalarT,FeatureDimension> transformed_data_;
        ConstVectorSetMatrixMap<ScalarT,FeatureDimension> transformed_data_map_;
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    class PointNormalFeaturesAdaptor {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum {FeatureDimension = (EigenDim == Eigen::Dynamic) ? Eigen::Dynamic : 2*EigenDim};

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

        inline const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getFeatureData() const { return data_map_; }

        inline const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getTransformedFeatureData() const { return transformed_data_map_; }

        const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getTransformedFeatureData(const RigidTransformation<ScalarT,EigenDim> &tform) {
            size_t dim = data_map_.rows()/2;
#pragma omp parallel for
            for (size_t i = 0; i < data_map_.cols(); i++) {
                auto res_col = transformed_data_.col(i);
                auto data_col = data_map_.col(i);
                res_col.head(dim) = tform.linear()*data_col.head(dim) + tform.translation();
                res_col.tail(dim) = tform.linear()*data_col.tail(dim);
            }
            return transformed_data_map_;
        }

        const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getTransformedFeatureData(const RigidTransformationSet<ScalarT,EigenDim> &tforms) {
            size_t dim = data_map_.rows()/2;
#pragma omp parallel for
            for (size_t i = 0; i < data_map_.cols(); i++) {
                auto res_col = transformed_data_.col(i);
                auto data_col = data_map_.col(i);
                res_col.head(dim) = tforms[i].linear()*data_col.head(dim) + tforms[i].translation();
                res_col.tail(dim) = tforms[i].linear()*data_col.tail(dim);
            }
            return transformed_data_map_;
        }

    private:
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

        enum {FeatureDimension = (EigenDim == Eigen::Dynamic) ? Eigen::Dynamic : EigenDim + 3};

        PointColorFeaturesAdaptor(const ConstVectorSetMatrixMap<ScalarT,FeatureDimension> &data)
                : data_map_(data),
                  transformed_data_(data.rows(), data.cols()),
                  transformed_data_map_(transformed_data_)
        {}

        PointColorFeaturesAdaptor(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                  const ConstVectorSetMatrixMap<ScalarT,3> &colors,
                                  ScalarT color_weight)
                : data_(points.rows()+3, points.cols()),
                  data_map_(data_),
                  transformed_data_(data_.rows(), data_.cols()),
                  transformed_data_map_(transformed_data_)
        {
            data_.topRows(points.rows()) = points;
            data_.bottomRows(3) = color_weight*colors;
        }

        inline const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getFeatureData() const { return data_map_; }

        inline const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getTransformedFeatureData() const { return transformed_data_map_; }

        const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getTransformedFeatureData(const RigidTransformation<ScalarT,EigenDim> &tform) {
            size_t dim = data_map_.rows() - 3;
#pragma omp parallel for
            for (size_t i = 0; i < data_map_.cols(); i++) {
                auto res_col = transformed_data_.col(i);
                auto data_col = data_map_.col(i);
                res_col.head(dim) = tform.linear()*data_col.head(dim) + tform.translation();
                res_col.tail(3) = data_col.tail(3);
            }
            return transformed_data_map_;
        }

        const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getTransformedFeatureData(const RigidTransformationSet<ScalarT,EigenDim> &tforms) {
            size_t dim = data_map_.rows() - 3;
#pragma omp parallel for
            for (size_t i = 0; i < data_map_.cols(); i++) {
                auto res_col = transformed_data_.col(i);
                auto data_col = data_map_.col(i);
                res_col.head(dim) = tforms[i].linear()*data_col.head(dim) + tforms[i].translation();
                res_col.tail(3) = data_col.tail(3);
            }
            return transformed_data_map_;
        }

    private:
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

        enum {FeatureDimension = (EigenDim == Eigen::Dynamic) ? Eigen::Dynamic : 2*EigenDim + 3};

        PointNormalColorFeaturesAdaptor(const ConstVectorSetMatrixMap<ScalarT,FeatureDimension> &data)
                : data_map_(data),
                  transformed_data_(data.rows(), data.cols()),
                  transformed_data_map_(transformed_data_)
        {}

        PointNormalColorFeaturesAdaptor(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                        const ConstVectorSetMatrixMap<ScalarT,EigenDim> &normals,
                                        const ConstVectorSetMatrixMap<ScalarT,3> &colors,
                                        ScalarT normal_weight, ScalarT color_weight)
                : data_(2*points.rows()+3, points.cols()),
                  data_map_(data_),
                  transformed_data_(data_.rows(), data_.cols()),
                  transformed_data_map_(transformed_data_)
        {
            data_.topRows(points.rows()) = points;
            data_.block(points.rows(),0,normals.rows(),normals.cols()) = normal_weight*normals;
            data_.bottomRows(3) = color_weight*colors;
        }

        inline const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getFeatureData() const { return data_map_; }

        inline const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getTransformedFeatureData() const { return transformed_data_map_; }

        const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getTransformedFeatureData(const RigidTransformation<ScalarT,EigenDim> &tform) {
            size_t dim = (data_map_.rows() - 3)/2;
#pragma omp parallel for
            for (size_t i = 0; i < data_map_.cols(); i++) {
                auto res_col = transformed_data_.col(i);
                auto data_col = data_map_.col(i);
                res_col.head(dim) = tform.linear()*data_col.head(dim) + tform.translation();
                res_col.segment(dim,dim) = tform.linear()*data_col.segment(dim,dim);
                res_col.tail(3) = data_col.tail(3);
            }
            return transformed_data_map_;
        }

        const ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getTransformedFeatureData(const RigidTransformationSet<ScalarT,EigenDim> &tforms) {
            size_t dim = (data_map_.rows() - 3)/2;
#pragma omp parallel for
            for (size_t i = 0; i < data_map_.cols(); i++) {
                auto res_col = transformed_data_.col(i);
                auto data_col = data_map_.col(i);
                res_col.head(dim) = tforms[i].linear()*data_col.head(dim) + tforms[i].translation();
                res_col.segment(dim,dim) = tforms[i].linear()*data_col.segment(dim,dim);
                res_col.tail(3) = data_col.tail(3);
            }
            return transformed_data_map_;
        }

    private:
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
