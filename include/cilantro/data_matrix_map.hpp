#pragma once

#include <vector>
#include <Eigen/Dense>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class DataMatrixMap : public Eigen::Map<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        DataMatrixMap(Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> &data)
                : Eigen::Map<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>>(data.data(), data.rows(), data.cols())
        {}

        DataMatrixMap(Eigen::Map<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>> data)
                : Eigen::Map<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>>(data.data(), data.rows(), data.cols())
        {}

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        DataMatrixMap(std::vector<Eigen::Matrix<ScalarT,EigenDim,1>> &data)
                : Eigen::Map<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>>((ScalarT *)data.data(), EigenDim, data.size())
        {}

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        DataMatrixMap(ScalarT * data, size_t num_points)
                : Eigen::Map<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>>(data, EigenDim, num_points)
        {}

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim == Eigen::Dynamic>::type>
        DataMatrixMap(ScalarT * data, size_t dim, size_t num_points)
                : Eigen::Map<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>>(data, dim, num_points)
        {}

        inline Eigen::Map<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>>& eigenMap() { return (*(Eigen::Map<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>> *)this); }
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    class ConstDataMatrixMap : public Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ConstDataMatrixMap(const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> &data)
                : Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>>(data.data(), data.rows(), data.cols())
        {}

        ConstDataMatrixMap(Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>> data)
                : Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>>(data.data(), data.rows(), data.cols())
        {}

        ConstDataMatrixMap(Eigen::Map<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>> data)
                : Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>>(data.data(), data.rows(), data.cols())
        {}

//        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic && sizeof(Eigen::Matrix<ScalarT,Dim,1>) % 16 != 0>::type>
        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        ConstDataMatrixMap(const std::vector<Eigen::Matrix<ScalarT,EigenDim,1>> &data)
                : Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>>((const ScalarT *)data.data(), EigenDim, data.size())
        {}

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        ConstDataMatrixMap(const ScalarT * data, size_t num_points)
                : Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>>(data, EigenDim, num_points)
        {}

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim == Eigen::Dynamic>::type>
        ConstDataMatrixMap(const ScalarT * data, size_t dim, size_t num_points)
                : Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>>(data, dim, num_points)
        {}

        inline Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>>& eigenMap() { return (*(Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>> *)this); }
    };
}
