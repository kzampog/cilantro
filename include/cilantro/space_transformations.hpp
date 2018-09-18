#pragma once

#include <cilantro/data_containers.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    using RigidTransformation = Eigen::Transform<ScalarT,EigenDim,Eigen::Isometry>;

    template <typename ScalarT, ptrdiff_t EigenDim>
#ifdef _MSC_VER
    using RigidTransformationSetBase = std::vector<RigidTransformation<ScalarT, EigenDim>>;
#else
    using RigidTransformationSetBase = typename std::conditional<EigenDim != Eigen::Dynamic && sizeof(RigidTransformation<ScalarT,EigenDim>) % 16 == 0,
            std::vector<RigidTransformation<ScalarT,EigenDim>,Eigen::aligned_allocator<RigidTransformation<ScalarT,EigenDim>>>,
            std::vector<RigidTransformation<ScalarT,EigenDim>>>::type;
#endif
    template <typename ScalarT, ptrdiff_t EigenDim>
    class RigidTransformationSet : public RigidTransformationSetBase<ScalarT,EigenDim> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;

        enum { Dim = EigenDim };

        enum { Dimension = EigenDim };

        RigidTransformationSet() {}

        RigidTransformationSet(size_t size)
            : RigidTransformationSetBase<ScalarT,EigenDim>(size)
        {}

        RigidTransformationSet(const RigidTransformationSet<ScalarT,EigenDim> &other)
            : RigidTransformationSetBase<ScalarT,EigenDim>(other)
        {}

        RigidTransformationSet(size_t size, const RigidTransformation<ScalarT,EigenDim> &tform)
            : RigidTransformationSetBase<ScalarT,EigenDim>(size, tform)
        {}

        RigidTransformationSet& setIdentity() {
#pragma omp parallel for
            for (size_t i = 0; i < this->size(); i++) (*this)[i].setIdentity();
            return *this;
        }

        RigidTransformationSet& setConstant(const RigidTransformation<ScalarT,EigenDim> &transform) {
#pragma omp parallel for
            for (size_t i = 0; i < this->size(); i++) (*this)[i] = transform;
            return *this;
        }

        RigidTransformationSet inverse() const {
            RigidTransformationSet res(this->size());
#pragma omp parallel for
            for (size_t i = 0; i < res.size(); i++) res[i] = (*this)[i].inverse();
            return res;
        }

        RigidTransformationSet& invert() {
#pragma omp parallel for
            for (size_t i = 0; i < this->size(); i++) (*this)[i] = (*this)[i].inverse();
            return *this;
        }

        RigidTransformationSet& preApply(const RigidTransformationSet<ScalarT,EigenDim> &other) {
#pragma omp parallel for
            for (size_t i = 0; i < this->size(); i++) {
                (*this)[i] = other[i]*(*this)[i];
                (*this)[i].linear() = (*this)[i].rotation();
            }
            return *this;
        }

        RigidTransformationSet& postApply(const RigidTransformationSet<ScalarT,EigenDim> &other) {
#pragma omp parallel for
            for (size_t i = 0; i < this->size(); i++) {
                (*this)[i] = (*this)[i]*other[i];
                (*this)[i].linear() = (*this)[i].rotation();
            }
            return *this;
        }

        VectorSet<ScalarT,EigenDim> getWarpedPoints(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points) const {
            VectorSet<ScalarT,EigenDim> points_t(points.rows(), points.cols());
#pragma omp parallel for
            for (size_t i = 0; i < points_t.cols(); i++) {
                points_t.col(i).noalias() = (*this)[i]*points.col(i);
            }
            return points_t;
        }

        const RigidTransformationSet& warpPoints(DataMatrixMap<ScalarT,EigenDim> points) const {
#pragma omp parallel for
            for (size_t i = 0; i < points.cols(); i++) {
                points.col(i) = (*this)[i]*points.col(i);
            }
            return *this;
        }

        VectorSet<ScalarT,EigenDim> getWarpedNormals(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &normals) const {
            VectorSet<ScalarT,EigenDim> normals_t(normals.rows(), normals.cols());
#pragma omp parallel for
            for (size_t i = 0; i < normals_t.cols(); i++) {
                normals_t.col(i).noalias() = (*this)[i].linear()*normals.col(i);
            }
            return normals_t;
        }

        const RigidTransformationSet& warpNormals(DataMatrixMap<ScalarT,EigenDim> normals) const {
#pragma omp parallel for
            for (size_t i = 0; i < normals.cols(); i++) {
                normals.col(i) = (*this)[i].linear()*normals.col(i);
            }
            return *this;
        }

        const RigidTransformationSet& warpPointsNormals(DataMatrixMap<ScalarT,EigenDim> points,
                                                        DataMatrixMap<ScalarT,EigenDim> normals) const
        {
#pragma omp parallel for
            for (size_t i = 0; i < points.cols(); i++) {
                points.col(i) = (*this)[i]*points.col(i);
                normals.col(i) = (*this)[i].linear()*normals.col(i);
            }
            return *this;
        }
    };

    typedef RigidTransformation<float,2> RigidTransformation2f;
    typedef RigidTransformation<double,2> RigidTransformation2d;
    typedef RigidTransformation<float,3> RigidTransformation3f;
    typedef RigidTransformation<double,3> RigidTransformation3d;

    typedef RigidTransformationSet<float,2> RigidTransformationSet2f;
    typedef RigidTransformationSet<double,2> RigidTransformationSet2d;
    typedef RigidTransformationSet<float,3> RigidTransformationSet3f;
    typedef RigidTransformationSet<double,3> RigidTransformationSet3d;
}
