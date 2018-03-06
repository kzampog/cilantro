#pragma once

#include <cilantro/rigid_transformation.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class PointsAdaptor {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum {FeatureDimension = EigenDim};

        PointsAdaptor(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points)
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
    class PointsNormalsAdaptor {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum {FeatureDimension = (EigenDim == Eigen::Dynamic) ? Eigen::Dynamic : 2*EigenDim};

        PointsNormalsAdaptor(const ConstVectorSetMatrixMap<ScalarT,FeatureDimension> &data)
                : data_map_(data),
                  transformed_data_(data.rows(), data.cols()),
                  transformed_data_map_(transformed_data_)
        {}

        PointsNormalsAdaptor(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
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
    class PointsColorsAdaptor {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum {FeatureDimension = (EigenDim == Eigen::Dynamic) ? Eigen::Dynamic : EigenDim + 3};

        PointsColorsAdaptor(const ConstVectorSetMatrixMap<ScalarT,FeatureDimension> &data)
                : data_map_(data),
                  transformed_data_(data.rows(), data.cols()),
                  transformed_data_map_(transformed_data_)
        {}

        PointsColorsAdaptor(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
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
    class PointsNormalsColorsAdaptor {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum {FeatureDimension = (EigenDim == Eigen::Dynamic) ? Eigen::Dynamic : 2*EigenDim + 3};

        PointsNormalsColorsAdaptor(const ConstVectorSetMatrixMap<ScalarT,FeatureDimension> &data)
                : data_map_(data),
                  transformed_data_(data.rows(), data.cols()),
                  transformed_data_map_(transformed_data_)
        {}

        PointsNormalsColorsAdaptor(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
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
}
