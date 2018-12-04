#pragma once

#include <cilantro/data_containers.hpp>

namespace cilantro {
    namespace internal {
        template <ptrdiff_t HDim>
        struct SpaceDimensionFromHomogeneous {
            enum { value = (HDim == Eigen::Dynamic) ? Eigen::Dynamic : HDim-1 };
        };

        template <ptrdiff_t Dim>
        struct HomogeneousDimensionFromSpace {
            enum { value = (Dim == Eigen::Dynamic) ? Eigen::Dynamic : Dim+1 };
        };
    }

    template <typename ScalarT, ptrdiff_t EigenDim>
    using RigidTransform = Eigen::Transform<ScalarT,EigenDim,Eigen::Isometry>;

    template <typename ScalarT, ptrdiff_t EigenDim>
    using AffineTransform = Eigen::Transform<ScalarT,EigenDim,Eigen::Affine>;

    template <class TransformT>
#ifdef _MSC_VER
    using TransformSetBase = std::vector<TransformT,Eigen::aligned_allocator<TransformT>>;
#else
    using TransformSetBase = typename std::conditional<TransformT::Dim != Eigen::Dynamic && sizeof(TransformT) % 16 == 0,
            std::vector<TransformT,Eigen::aligned_allocator<TransformT>>,
            std::vector<TransformT>>::type;
#endif

    template <class TransformT>
    class TransformSet : public TransformSetBase<TransformT> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef TransformT Transform;

        typedef typename TransformT::Scalar Scalar;

        enum { Dim = TransformT::Dim };

        enum { Dimension = TransformT::Dim };

        TransformSet() {}

        TransformSet(size_t size) : TransformSetBase<TransformT>(size) {}

        TransformSet(const TransformSet<TransformT> &other) : TransformSetBase<TransformT>(other) {}

        TransformSet(size_t size, const TransformT &tform) : TransformSetBase<TransformT>(size, tform) {}

        TransformSet& setIdentity() {
#pragma omp parallel for
            for (size_t i = 0; i < this->size(); i++) (*this)[i].setIdentity();
            return *this;
        }

        TransformSet& setConstant(const TransformT &transform) {
#pragma omp parallel for
            for (size_t i = 0; i < this->size(); i++) (*this)[i] = transform;
            return *this;
        }

        TransformSet inverse() const {
            TransformSet res(this->size());
#pragma omp parallel for
            for (size_t i = 0; i < res.size(); i++) res[i] = (*this)[i].inverse();
            return res;
        }

        TransformSet& invert() {
#pragma omp parallel for
            for (size_t i = 0; i < this->size(); i++) (*this)[i] = (*this)[i].inverse();
            return *this;
        }

        TransformSet& preApply(const TransformSet<TransformT> &other) {
            if (int(TransformT::Mode) == int(Eigen::Isometry)) {
#pragma omp parallel for
                for (size_t i = 0; i < this->size(); i++) {
                    (*this)[i] = other[i]*(*this)[i];
                    (*this)[i].linear() = (*this)[i].rotation();
                }
            } else {
#pragma omp parallel for
                for (size_t i = 0; i < this->size(); i++) {
                    (*this)[i] = other[i]*(*this)[i];
                }
            }
            return *this;
        }

        TransformSet& postApply(const TransformSet<TransformT> &other) {
            if (int(TransformT::Mode) == int(Eigen::Isometry)) {
#pragma omp parallel for
                for (size_t i = 0; i < this->size(); i++) {
                    (*this)[i] = (*this)[i]*other[i];
                    (*this)[i].linear() = (*this)[i].rotation();
                }
            } else {
#pragma omp parallel for
                for (size_t i = 0; i < this->size(); i++) {
                    (*this)[i] = (*this)[i]*other[i];
                }
            }
            return *this;
        }
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    using RigidTransformSet = TransformSet<RigidTransform<ScalarT,EigenDim>>;

    template <typename ScalarT, ptrdiff_t EigenDim>
    using AffineTransformSet = TransformSet<AffineTransform<ScalarT,EigenDim>>;

    // Point set transformations

    template <class LinearT, class TranslationT>
    void transformPoints(const LinearT &linear,
                         const TranslationT &translation,
                         DataMatrixMap<typename LinearT::Scalar,LinearT::ColsAtCompileTime> points)
    {
#pragma omp parallel for
        for (size_t i = 0; i < points.cols(); i++) {
            points.col(i) = linear*points.col(i) + translation;
        }
    }

    template <class MatrixT>
    inline typename std::enable_if<MatrixT::ColsAtCompileTime != Eigen::Dynamic,void>::type
    transformPoints(const MatrixT &tform,
                    DataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> points)
    {
        transformPoints(tform.template block<internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value>(0, 0),
                        tform.template block<internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value,1>(0, internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value),
                        points);
    }

    template <class MatrixT>
    inline typename std::enable_if<MatrixT::ColsAtCompileTime == Eigen::Dynamic,void>::type
    transformPoints(const MatrixT &tform,
                    DataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> points)
    {
        transformPoints(tform.block(0, 0, points.rows(), points.rows()), tform.block(0, points.rows(), points.rows(), 1), points);
    }

    template <class TransformT>
    void transformPoints(const TransformT &tform,
                         DataMatrixMap<typename TransformT::Scalar,TransformT::Dim> points)
    {
#pragma omp parallel for
        for (size_t i = 0; i < points.cols(); i++) {
            points.col(i) = tform*points.col(i);
        }
    }

    template <class TransformT>
    void transformPoints(const TransformSet<TransformT> &tforms,
                         DataMatrixMap<typename TransformT::Scalar,TransformT::Dim> points)
    {
#pragma omp parallel for
        for (size_t i = 0; i < points.cols(); i++) {
            points.col(i) = tforms[i]*points.col(i);
        }
    }

    template <class LinearT, class TranslationT>
    void transformPoints(const LinearT &linear,
                         const TranslationT &translation,
                         const ConstDataMatrixMap<typename LinearT::Scalar,LinearT::ColsAtCompileTime> &points,
                         DataMatrixMap<typename LinearT::Scalar,LinearT::ColsAtCompileTime> result)
    {
        if (result.data() == points.data()) {
            transformPoints(linear, translation, result);
            return;
        }

#pragma omp parallel for
        for (size_t i = 0; i < points.cols(); i++) {
            result.col(i).noalias() = linear*points.col(i) + translation;
        }
    }

    template <class MatrixT>
    inline typename std::enable_if<MatrixT::ColsAtCompileTime != Eigen::Dynamic,void>::type
    transformPoints(const MatrixT &tform,
                    const ConstDataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> &points,
                    DataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> result)
    {
        transformPoints(tform.template block<internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value>(0, 0),
                        tform.template block<internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value,1>(0, internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value),
                        points,
                        result);
    }

    template <class MatrixT>
    inline typename std::enable_if<MatrixT::ColsAtCompileTime == Eigen::Dynamic,void>::type
    transformPoints(const MatrixT &tform,
                    const ConstDataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> &points,
                    DataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> result)
    {
        transformPoints(tform.block(0, 0, points.rows(), points.rows()), tform.block(0, points.rows(), points.rows(), 1), points, result);
    }

    template <class TransformT>
    void transformPoints(const TransformT &tform,
                         const ConstDataMatrixMap<typename TransformT::Scalar,TransformT::Dim> &points,
                         DataMatrixMap<typename TransformT::Scalar,TransformT::Dim> result)
    {
        if (result.data() == points.data()) {
            transformPoints(tform, result);
            return;
        }

#pragma omp parallel for
        for (size_t i = 0; i < points.cols(); i++) {
            result.col(i).noalias() = tform*points.col(i);
        }
    }

    template <class TransformT>
    void transformPoints(const TransformSet<TransformT> &tforms,
                         const ConstDataMatrixMap<typename TransformT::Scalar,TransformT::Dim> &points,
                         DataMatrixMap<typename TransformT::Scalar,TransformT::Dim> result)
    {
        if (result.data() == points.data()) {
            transformPoints(tforms, result);
            return;
        }

#pragma omp parallel for
        for (size_t i = 0; i < points.cols(); i++) {
            result.col(i).noalias() = tforms[i]*points.col(i);
        }
    }

    template <class LinearT, class TranslationT>
    inline VectorSet<typename LinearT::Scalar,LinearT::ColsAtCompileTime>
    getTransformedPoints(const LinearT &linear,
                         const TranslationT &translation,
                         const ConstDataMatrixMap<typename LinearT::Scalar,LinearT::ColsAtCompileTime> &points)
    {
        VectorSet<typename LinearT::Scalar,LinearT::ColsAtCompileTime> result(points.rows(), points.cols());
        transformPoints(linear, translation, points, result);
        return result;
    }

    template <class MatrixT>
    inline VectorSet<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value>
    getTransformedPoints(const MatrixT &tform,
                         const ConstDataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> &points)
    {
        VectorSet<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> result(points.rows(), points.cols());
        transformPoints(tform, points, result);
        return result;
    }

    template <class TransformT>
    inline VectorSet<typename TransformT::Scalar,TransformT::Dim>
    getTransformedPoints(const TransformT &tform,
                         const ConstDataMatrixMap<typename TransformT::Scalar,TransformT::Dim> &points)
    {
        VectorSet<typename TransformT::Scalar,TransformT::Dim> result(points.rows(), points.cols());
        transformPoints(tform, points, result);
        return result;
    }

    // Normal transformations

    template <class LinearT>
    typename std::enable_if<LinearT::ColsAtCompileTime != Eigen::Dynamic,void>::type
    transformNormals(const LinearT &linear,
                     DataMatrixMap<typename LinearT::Scalar,LinearT::ColsAtCompileTime> normals)
    {
        Eigen::Matrix<typename LinearT::Scalar,LinearT::ColsAtCompileTime,LinearT::ColsAtCompileTime> normal_tform = linear.inverse().transpose();
        if ((normal_tform - linear).squaredNorm() < std::numeric_limits<typename LinearT::Scalar>::epsilon()) {
#pragma omp parallel for
            for (size_t i = 0; i < normals.cols(); i++) {
                normals.col(i) = normal_tform*normals.col(i);
            }
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < normals.cols(); i++) {
                normals.col(i) = (normal_tform*normals.col(i)).normalized();
            }
        }
    }

    template <class MatrixT>
    inline typename std::enable_if<MatrixT::ColsAtCompileTime != Eigen::Dynamic,void>::type
    transformNormals(const MatrixT &tform,
                     DataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> normals)
    {
        transformNormals(tform.template block<internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value>(0, 0), normals);
    }

    template <class MatrixT>
    typename std::enable_if<MatrixT::ColsAtCompileTime == Eigen::Dynamic,void>::type
    transformNormals(const MatrixT &tform,
                     DataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> normals)
    {
        if (MatrixT::ColsAtCompileTime == Eigen::Dynamic && tform.cols() > normals.rows()) {
            transformNormals(tform.block(0, 0, normals.rows(), normals.rows()), normals);
            return;
        }

        Eigen::Matrix<typename MatrixT::Scalar,MatrixT::ColsAtCompileTime,MatrixT::ColsAtCompileTime> normal_tform = tform.inverse().transpose();
        if ((normal_tform - tform).squaredNorm() < std::numeric_limits<typename MatrixT::Scalar>::epsilon()) {
#pragma omp parallel for
            for (size_t i = 0; i < normals.cols(); i++) {
                normals.col(i) = normal_tform*normals.col(i);
            }
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < normals.cols(); i++) {
                normals.col(i) = (normal_tform*normals.col(i)).normalized();
            }
        }
    }

    template <class TransformT>
    void transformNormals(const TransformT &tform,
                          DataMatrixMap<typename TransformT::Scalar,TransformT::Dim> normals)
    {
        if (int(TransformT::Mode) == int(Eigen::Isometry)) {
#pragma omp parallel for
            for (size_t i = 0; i < normals.cols(); i++) {
                normals.col(i) = tform.linear()*normals.col(i);
            }
        } else {
            Eigen::Matrix<typename TransformT::Scalar,TransformT::Dim,TransformT::Dim> normal_tform = tform.linear().inverse().transpose();
#pragma omp parallel for
            for (size_t i = 0; i < normals.cols(); i++) {
                normals.col(i) = (normal_tform*normals.col(i)).normalized();
            }
        }
    }

    template <class TransformT>
    void transformNormals(const TransformSet<TransformT> &tforms,
                          DataMatrixMap<typename TransformT::Scalar,TransformT::Dim> normals)
    {
        if (int(TransformT::Mode) == int(Eigen::Isometry)) {
#pragma omp parallel for
            for (size_t i = 0; i < normals.cols(); i++) {
                normals.col(i) = tforms[i].linear()*normals.col(i);
            }
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < normals.cols(); i++) {
                normals.col(i) = (tforms[i].linear().inverse().transpose()*normals.col(i)).normalized();
            }
        }
    }

    template <class LinearT>
    typename std::enable_if<LinearT::ColsAtCompileTime != Eigen::Dynamic,void>::type
    transformNormals(const LinearT &linear,
                     const ConstDataMatrixMap<typename LinearT::Scalar,LinearT::ColsAtCompileTime> &normals,
                     DataMatrixMap<typename LinearT::Scalar,LinearT::ColsAtCompileTime> result)
    {
        if (result.data() == normals.data()) {
            transformNormals(linear, result);
            return;
        }

        Eigen::Matrix<typename LinearT::Scalar,LinearT::ColsAtCompileTime,LinearT::ColsAtCompileTime> normal_tform = linear.inverse().transpose();
        if ((normal_tform - linear).squaredNorm() < std::numeric_limits<typename LinearT::Scalar>::epsilon()) {
#pragma omp parallel for
            for (size_t i = 0; i < normals.cols(); i++) {
                result.col(i).noalias() = normal_tform*normals.col(i);
            }
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < normals.cols(); i++) {
                result.col(i).noalias() = (normal_tform*normals.col(i)).normalized();
            }
        }
    }

    template <class MatrixT>
    inline typename std::enable_if<MatrixT::ColsAtCompileTime != Eigen::Dynamic,void>::type
    transformNormals(const MatrixT &tform,
                     const ConstDataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> &normals,
                     DataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> result)
    {
        transformNormals(tform.template block<internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value>(0, 0), normals, result);
    }

    template <class MatrixT>
    typename std::enable_if<MatrixT::ColsAtCompileTime == Eigen::Dynamic,void>::type
    transformNormals(const MatrixT &tform,
                     const ConstDataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> &normals,
                     DataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> result)
    {
        if (MatrixT::ColsAtCompileTime == Eigen::Dynamic && tform.cols() > normals.rows()) {
            transformNormals(tform.block(0, 0, normals.rows(), normals.rows()), normals, result);
            return;
        }

        if (result.data() == normals.data()) {
            transformNormals(tform, result);
            return;
        }

        Eigen::Matrix<typename MatrixT::Scalar,MatrixT::ColsAtCompileTime,MatrixT::ColsAtCompileTime> normal_tform = tform.inverse().transpose();
        if ((normal_tform - tform).squaredNorm() < std::numeric_limits<typename MatrixT::Scalar>::epsilon()) {
#pragma omp parallel for
            for (size_t i = 0; i < normals.cols(); i++) {
                result.col(i).noalias() = normal_tform*normals.col(i);
            }
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < normals.cols(); i++) {
                result.col(i).noalias() = (normal_tform*normals.col(i)).normalized();
            }
        }
    }

    template <class TransformT>
    void transformNormals(const TransformT &tform,
                          const ConstDataMatrixMap<typename TransformT::Scalar,TransformT::Dim> &normals,
                          DataMatrixMap<typename TransformT::Scalar,TransformT::Dim> result)
    {
        if (result.data() == normals.data()) {
            transformNormals(tform, result);
            return;
        }

        if (int(TransformT::Mode) == int(Eigen::Isometry)) {
#pragma omp parallel for
            for (size_t i = 0; i < normals.cols(); i++) {
                result.col(i).noalias() = tform.linear()*normals.col(i);
            }
        } else {
            Eigen::Matrix<typename TransformT::Scalar,TransformT::Dim,TransformT::Dim> normal_tform = tform.linear().inverse().transpose();
#pragma omp parallel for
            for (size_t i = 0; i < normals.cols(); i++) {
                result.col(i).noalias() = (normal_tform*normals.col(i)).normalized();
            }
        }
    }

    template <class TransformT>
    void transformNormals(const TransformSet<TransformT> &tforms,
                          const ConstDataMatrixMap<typename TransformT::Scalar,TransformT::Dim> &normals,
                          DataMatrixMap<typename TransformT::Scalar,TransformT::Dim> result)
    {
        if (result.data() == normals.data()) {
            transformNormals(tforms, result);
            return;
        }

        if (int(TransformT::Mode) == int(Eigen::Isometry)) {
#pragma omp parallel for
            for (size_t i = 0; i < normals.cols(); i++) {
                result.col(i).noalias() = tforms[i].linear()*normals.col(i);
            }
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < normals.cols(); i++) {
                result.col(i).noalias() = (tforms[i].linear().inverse().transpose()*normals.col(i)).normalized();
            }
        }
    }

    template <class MatrixT>
    inline VectorSet<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value>
    getTransformedNormals(const MatrixT &tform,
                          const ConstDataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> &normals)
    {
        VectorSet<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> result(normals.rows(), normals.cols());
        transformNormals(tform, normals, result);
        return result;
    }

    template <class TransformT>
    inline VectorSet<typename TransformT::Scalar,TransformT::Dim>
    getTransformedNormals(const TransformT &tform,
                          const ConstDataMatrixMap<typename TransformT::Scalar,TransformT::Dim> &normals)
    {
        VectorSet<typename TransformT::Scalar,TransformT::Dim> result(normals.rows(), normals.cols());
        transformNormals(tform, normals, result);
        return result;
    }

    // Point & normal transformations

    template <class LinearT, class TranslationT>
    void transformPointsNormals(const LinearT &linear,
                                const TranslationT &translation,
                                DataMatrixMap<typename LinearT::Scalar,LinearT::ColsAtCompileTime> points,
                                DataMatrixMap<typename LinearT::Scalar,LinearT::ColsAtCompileTime> normals)
    {
        Eigen::Matrix<typename LinearT::Scalar,LinearT::ColsAtCompileTime,LinearT::ColsAtCompileTime> normal_tform = linear.inverse().transpose();
        if ((normal_tform - linear).squaredNorm() < std::numeric_limits<typename LinearT::Scalar>::epsilon()) {
#pragma omp parallel for
            for (size_t i = 0; i < points.cols(); i++) {
                points.col(i) = linear*points.col(i) + translation;
                normals.col(i) = normal_tform*normals.col(i);
            }
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < points.cols(); i++) {
                points.col(i) = linear*points.col(i) + translation;
                normals.col(i) = (normal_tform*normals.col(i)).normalized();
            }
        }
    }

    template <class MatrixT>
    inline typename std::enable_if<MatrixT::ColsAtCompileTime != Eigen::Dynamic,void>::type
    transformPointsNormals(const MatrixT &tform,
                           DataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> points,
                           DataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> normals)
    {
        transformPointsNormals(tform.template block<internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value>(0, 0),
                               tform.template block<internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value,1>(0, internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value),
                               points,
                               normals);
    }

    template <class MatrixT>
    inline typename std::enable_if<MatrixT::ColsAtCompileTime == Eigen::Dynamic,void>::type
    transformPointsNormals(const MatrixT &tform,
                           DataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> points,
                           DataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> normals)
    {
        transformPointsNormals(tform.block(0, 0, points.rows(), points.rows()), tform.block(0, points.rows(), points.rows(), 1), points, normals);
    }

    template <class TransformT>
    void transformPointsNormals(const TransformT &tform,
                                DataMatrixMap<typename TransformT::Scalar,TransformT::Dim> points,
                                DataMatrixMap<typename TransformT::Scalar,TransformT::Dim> normals)
    {
        if (int(TransformT::Mode) == int(Eigen::Isometry)) {
#pragma omp parallel for
            for (size_t i = 0; i < points.cols(); i++) {
                points.col(i) = tform*points.col(i);
                normals.col(i) = tform.linear()*normals.col(i);
            }
        } else {
            Eigen::Matrix<typename TransformT::Scalar,TransformT::Dim,TransformT::Dim> normal_tform = tform.linear().inverse().transpose();
#pragma omp parallel for
            for (size_t i = 0; i < points.cols(); i++) {
                points.col(i) = tform*points.col(i);
                normals.col(i) = (normal_tform*normals.col(i)).normalized();
            }
        }
    }

    template <class TransformT>
    void transformPointsNormals(const TransformSet<TransformT> &tforms,
                                DataMatrixMap<typename TransformT::Scalar,TransformT::Dim> points,
                                DataMatrixMap<typename TransformT::Scalar,TransformT::Dim> normals)
    {
        if (int(TransformT::Mode) == int(Eigen::Isometry)) {
#pragma omp parallel for
            for (size_t i = 0; i < points.cols(); i++) {
                points.col(i) = tforms[i]*points.col(i);
                normals.col(i) = tforms[i].linear()*normals.col(i);
            }
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < points.cols(); i++) {
                points.col(i) = tforms[i]*points.col(i);
                normals.col(i) = (tforms[i].linear().inverse().transpose()*normals.col(i)).normalized();
            }
        }
    }

    template <class LinearT, class TranslationT>
    void transformPointsNormals(const LinearT &linear,
                                const TranslationT &translation,
                                const ConstDataMatrixMap<typename LinearT::Scalar,LinearT::ColsAtCompileTime> &points,
                                const ConstDataMatrixMap<typename LinearT::Scalar,LinearT::ColsAtCompileTime> &normals,
                                DataMatrixMap<typename LinearT::Scalar,LinearT::ColsAtCompileTime> points_trans,
                                DataMatrixMap<typename LinearT::Scalar,LinearT::ColsAtCompileTime> normals_trans)
    {
        if (points_trans.data() == points.data() || normals_trans.data() == normals.data()) {
            transformPoints(linear, translation, points, points_trans);
            transformNormals(linear, normals, normals_trans);
        }
        
        Eigen::Matrix<typename LinearT::Scalar,LinearT::ColsAtCompileTime,LinearT::ColsAtCompileTime> normal_tform = linear.inverse().transpose();
        if ((normal_tform - linear).squaredNorm() < std::numeric_limits<typename LinearT::Scalar>::epsilon()) {
#pragma omp parallel for
            for (size_t i = 0; i < points.cols(); i++) {
                points_trans.col(i).noalias() = linear*points.col(i) + translation;
                normals_trans.col(i).noalias() = normal_tform*normals.col(i);
            }
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < points.cols(); i++) {
                points_trans.col(i).noalias() = linear*points.col(i) + translation;
                normals_trans.col(i).noalias() = (normal_tform*normals.col(i)).normalized();
            }
        }
    }

    template <class MatrixT>
    inline typename std::enable_if<MatrixT::ColsAtCompileTime != Eigen::Dynamic,void>::type
    transformPointsNormals(const MatrixT &tform,
                           const ConstDataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> &points,
                           const ConstDataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> &normals,
                           DataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> points_trans,
                           DataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> normals_trans)
    {
        transformPointsNormals(tform.template block<internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value>(0, 0),
                               tform.template block<internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value,1>(0, internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value),
                               points,
                               normals,
                               points_trans,
                               normals_trans);
    }

    template <class MatrixT>
    inline typename std::enable_if<MatrixT::ColsAtCompileTime == Eigen::Dynamic,void>::type
    transformPointsNormals(const MatrixT &tform,
                           const ConstDataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> &points,
                           const ConstDataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> &normals,
                           DataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> points_trans,
                           DataMatrixMap<typename MatrixT::Scalar,internal::SpaceDimensionFromHomogeneous<MatrixT::ColsAtCompileTime>::value> normals_trans)
    {
        transformPointsNormals(tform.block(0, 0, points.rows(), points.rows()), tform.block(0, points.rows(), points.rows(), 1), points, normals, points_trans, normals_trans);
    }

    template <class TransformT>
    void transformPointsNormals(const TransformT &tform,
                                const ConstDataMatrixMap<typename TransformT::Scalar,TransformT::Dim> &points,
                                const ConstDataMatrixMap<typename TransformT::Scalar,TransformT::Dim> &normals,
                                DataMatrixMap<typename TransformT::Scalar,TransformT::Dim> points_trans,
                                DataMatrixMap<typename TransformT::Scalar,TransformT::Dim> normals_trans)
    {
        if (points_trans.data() == points.data() || normals_trans.data() == normals.data()) {
            transformPoints(tform, points, points_trans);
            transformNormals(tform, normals, normals_trans);
        }

        if (int(TransformT::Mode) == int(Eigen::Isometry)) {
#pragma omp parallel for
            for (size_t i = 0; i < normals.cols(); i++) {
                points_trans.col(i).noalias() = tform*points.col(i);
                normals_trans.col(i).noalias() = tform.linear()*normals.col(i);
            }
        } else {
            Eigen::Matrix<typename TransformT::Scalar,TransformT::Dim,TransformT::Dim> normal_tform = tform.linear().inverse().transpose();
#pragma omp parallel for
            for (size_t i = 0; i < normals.cols(); i++) {
                points_trans.col(i).noalias() = tform*points.col(i);
                normals_trans.col(i).noalias() = (normal_tform*normals.col(i)).normalized();
            }
        }
    }

    template <class TransformT>
    void transformPointsNormals(const TransformSet<TransformT> &tforms,
                                const ConstDataMatrixMap<typename TransformT::Scalar,TransformT::Dim> &points,
                                const ConstDataMatrixMap<typename TransformT::Scalar,TransformT::Dim> &normals,
                                DataMatrixMap<typename TransformT::Scalar,TransformT::Dim> points_trans,
                                DataMatrixMap<typename TransformT::Scalar,TransformT::Dim> normals_trans)
    {
        if (points_trans.data() == points.data() || normals_trans.data() == normals.data()) {
            transformPoints(tforms, points, points_trans);
            transformNormals(tforms, normals, normals_trans);
        }

        if (int(TransformT::Mode) == int(Eigen::Isometry)) {
#pragma omp parallel for
            for (size_t i = 0; i < normals.cols(); i++) {
                points_trans.col(i).noalias() = tforms[i]*points.col(i);
                normals_trans.col(i).noalias() = tforms[i].linear()*normals.col(i);
            }
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < normals.cols(); i++) {
                points_trans.col(i).noalias() = tforms[i]*points.col(i);
                normals_trans.col(i).noalias() = (tforms[i].linear().inverse().transpose()*normals.col(i)).normalized();
            }
        }
    }

    typedef RigidTransform<float,2> RigidTransform2f;
    typedef RigidTransform<double,2> RigidTransform2d;
    typedef RigidTransform<float,3> RigidTransform3f;
    typedef RigidTransform<double,3> RigidTransform3d;

    typedef AffineTransform<float,2> AffineTransform2f;
    typedef AffineTransform<double,2> AffineTransform2d;
    typedef AffineTransform<float,3> AffineTransform3f;
    typedef AffineTransform<double,3> AffineTransform3d;

    typedef RigidTransformSet<float,2> RigidTransformSet2f;
    typedef RigidTransformSet<double,2> RigidTransformSet2d;
    typedef RigidTransformSet<float,3> RigidTransformSet3f;
    typedef RigidTransformSet<double,3> RigidTransformSet3d;

    typedef AffineTransformSet<float,2> AffineTransformSet2f;
    typedef AffineTransformSet<double,2> AffineTransformSet2d;
    typedef AffineTransformSet<float,3> AffineTransformSet3f;
    typedef AffineTransformSet<double,3> AffineTransformSet3d;
}
