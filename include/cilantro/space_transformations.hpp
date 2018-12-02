#pragma once

#include <cilantro/data_containers.hpp>

namespace cilantro {
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

    template <class TransformT>
    inline VectorSet<typename TransformT::Scalar,TransformT::Dim> getTransformedPoints(const TransformT &tform,
                                                                                       const ConstDataMatrixMap<typename TransformT::Scalar,TransformT::Dim> &points)
    {
        VectorSet<typename TransformT::Scalar,TransformT::Dim> result(points.rows(), points.cols());
        transformPoints(tform, points, result);
        return result;
    }

    // Normal transformations
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

    template <class TransformT>
    inline VectorSet<typename TransformT::Scalar,TransformT::Dim> getTransformedNormals(const TransformT &tform,
                                                                                        const ConstDataMatrixMap<typename TransformT::Scalar,TransformT::Dim> &normals)
    {
        VectorSet<typename TransformT::Scalar,TransformT::Dim> result(normals.rows(), normals.cols());
        transformNormals(tform, normals, result);
        return result;
    }

    // Point & normal transformations
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
