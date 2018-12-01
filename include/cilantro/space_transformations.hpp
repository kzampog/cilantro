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

        VectorSet<Scalar,Dim> getTransformedPoints(const ConstVectorSetMatrixMap<Scalar,Dim> &points) const {
            VectorSet<Scalar,Dim> points_t(points.rows(), points.cols());
#pragma omp parallel for
            for (size_t i = 0; i < points_t.cols(); i++) {
                points_t.col(i).noalias() = (*this)[i]*points.col(i);
            }
            return points_t;
        }

        const TransformSet& transformPoints(DataMatrixMap<Scalar,Dim> points) const {
#pragma omp parallel for
            for (size_t i = 0; i < points.cols(); i++) {
                points.col(i) = (*this)[i]*points.col(i);
            }
            return *this;
        }

        VectorSet<Scalar,Dim> getTransformedNormals(const ConstVectorSetMatrixMap<Scalar,Dim> &normals) const {
            VectorSet<Scalar,Dim> normals_t(normals.rows(), normals.cols());
            if (int(TransformT::Mode) == int(Eigen::Isometry)) {
#pragma omp parallel for
                for (size_t i = 0; i < normals_t.cols(); i++) {
                    normals_t.col(i).noalias() = (*this)[i].linear()*normals.col(i);
                }
            } else {
#pragma omp parallel for
                for (size_t i = 0; i < normals_t.cols(); i++) {
                    normals_t.col(i).noalias() = ((*this)[i].linear().inverse().transpose()*normals.col(i)).normalized();
                }
            }
            return normals_t;
        }

        const TransformSet& transformNormals(DataMatrixMap<Scalar,Dim> normals) const {
            if (int(TransformT::Mode) == int(Eigen::Isometry)) {
#pragma omp parallel for
                for (size_t i = 0; i < normals.cols(); i++) {
                    normals.col(i) = (*this)[i].linear()*normals.col(i);
                }
            } else {
#pragma omp parallel for
                for (size_t i = 0; i < normals.cols(); i++) {
                    normals.col(i) = ((*this)[i].linear().inverse().transpose()*normals.col(i)).normalized();
                }
            }
            return *this;
        }

        const TransformSet& transformPointsNormals(DataMatrixMap<Scalar,Dim> points,
                                                   DataMatrixMap<Scalar,Dim> normals) const
        {
            if (int(TransformT::Mode) == int(Eigen::Isometry)) {
#pragma omp parallel for
                for (size_t i = 0; i < points.cols(); i++) {
                    points.col(i) = (*this)[i]*points.col(i);
                    normals.col(i) = (*this)[i].linear()*normals.col(i);
                }
            } else {
#pragma omp parallel for
                for (size_t i = 0; i < points.cols(); i++) {
                    points.col(i) = (*this)[i]*points.col(i);
                    normals.col(i) = ((*this)[i].linear().inverse().transpose()*normals.col(i)).normalized();
                }
            }
            return *this;
        }
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    using RigidTransformSet = TransformSet<RigidTransform<ScalarT,EigenDim>>;

    template <typename ScalarT, ptrdiff_t EigenDim>
    using AffineTransformSet = TransformSet<AffineTransform<ScalarT,EigenDim>>;

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
