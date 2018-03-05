#pragma once

#include <vector>
#include <Eigen/Dense>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    using RigidTransformation = Eigen::Transform<ScalarT,EigenDim,Eigen::Isometry>;

    template <typename ScalarT, ptrdiff_t EigenDim>
    using RigidTransformationSetBase = typename std::conditional<EigenDim != Eigen::Dynamic && sizeof(RigidTransformation<ScalarT,EigenDim>) % 16 == 0,
            std::vector<RigidTransformation<ScalarT,EigenDim>,Eigen::aligned_allocator<RigidTransformation<ScalarT,EigenDim>>>,
            std::vector<RigidTransformation<ScalarT,EigenDim>>>::type;

    template <typename ScalarT, ptrdiff_t EigenDim>
    class RigidTransformationSet : public RigidTransformationSetBase<ScalarT,EigenDim> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        RigidTransformationSet& setIdentity() {
            for (size_t i = 0; i < this->size(); i++) (*this)[i].setIdentity();
            return *this;
        }

        RigidTransformationSet& setConstant(const RigidTransformation<ScalarT,EigenDim> &transform) {
            for (size_t i = 0; i < this->size(); i++) (*this)[i] = transform;
            return *this;
        }

        RigidTransformationSet inverse() {
            RigidTransformationSet res(this->size());
            for (size_t i = 0; i < res.size(); i++) res[i] = (*this)[i].inverse();
            return res;
        }

        RigidTransformationSet& invert() {
            for (size_t i = 0; i < this->size(); i++) (*this)[i] = (*this)[i].inverse();
            return *this;
        }
    };

    typedef RigidTransformation<float,2> RigidTransformation2D;
    typedef RigidTransformation<float,3> RigidTransformation3D;

    typedef RigidTransformationSet<float,2> RigidTransformationSet2D;
    typedef RigidTransformationSet<float,3> RigidTransformationSet3D;
}
