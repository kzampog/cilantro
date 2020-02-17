#pragma once

#include <vector>
#include <Eigen/Dense>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    class DataMatrixMap : public Eigen::Map<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef Eigen::Map<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>> Base;

        typedef ScalarT Scalar;

        enum { Dimension = EigenDim };

        DataMatrixMap(Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> &data)
                : Base(data.data(), data.rows(), data.cols())
        {}

        DataMatrixMap(Eigen::Map<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>> data)
                : Base(data.data(), data.rows(), data.cols())
        {}

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        DataMatrixMap(std::vector<ScalarT> &data)
                : Base((ScalarT *)data.data(), EigenDim, data.size()/EigenDim)
        {}

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim == Eigen::Dynamic>::type>
        DataMatrixMap(std::vector<ScalarT> &data, size_t dim)
                : Base((ScalarT *)data.data(), dim, data.size()/dim)
        {}

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic && sizeof(Eigen::Matrix<ScalarT,Dim,1>) % 16 != 0>::type>
        DataMatrixMap(std::vector<Eigen::Matrix<ScalarT,EigenDim,1>> &data)
                : Base((ScalarT *)data.data(), EigenDim, data.size())
        {}

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        DataMatrixMap(std::vector<Eigen::Matrix<ScalarT,EigenDim,1>,Eigen::aligned_allocator<Eigen::Matrix<ScalarT,EigenDim,1>>> &data)
                : Base((ScalarT *)data.data(), EigenDim, data.size())
        {}

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        DataMatrixMap(ScalarT * data, size_t num_points = 0)
                : Base(data, EigenDim, num_points)
        {}

        DataMatrixMap(ScalarT * data, size_t dim, size_t num_points)
                : Base(data, dim, num_points)
        {}

        inline Base& base() {
            return *static_cast<Base *>(this);
        }
    };

    typedef DataMatrixMap<float,2> DataMatrixMap2f;
    typedef DataMatrixMap<double,2> DataMatrixMap2d;
    typedef DataMatrixMap<float,3> DataMatrixMap3f;
    typedef DataMatrixMap<double,3> DataMatrixMap3d;
    typedef DataMatrixMap<float,Eigen::Dynamic> DataMatrixMapXf;
    typedef DataMatrixMap<double,Eigen::Dynamic> DataMatrixMapXd;

    template <typename ScalarT, ptrdiff_t EigenDim>
    using VectorSetMatrixMap = DataMatrixMap<ScalarT,EigenDim>;

    typedef VectorSetMatrixMap<float,2> VectorSetMatrixMap2f;
    typedef VectorSetMatrixMap<double,2> VectorSetMatrixMap2d;
    typedef VectorSetMatrixMap<float,3> VectorSetMatrixMap3f;
    typedef VectorSetMatrixMap<double,3> VectorSetMatrixMap3d;
    typedef VectorSetMatrixMap<float,Eigen::Dynamic> VectorSetMatrixMapXf;
    typedef VectorSetMatrixMap<double,Eigen::Dynamic> VectorSetMatrixMapXd;

    template <typename ScalarT, ptrdiff_t EigenDim>
    using HomogeneousVectorSetMatrixMap = typename std::conditional<EigenDim == Eigen::Dynamic, DataMatrixMap<ScalarT,EigenDim>, DataMatrixMap<ScalarT,EigenDim+1>>::type;

    typedef HomogeneousVectorSetMatrixMap<float,2> HomogeneousVectorSetMatrixMap2f;
    typedef HomogeneousVectorSetMatrixMap<double,2> HomogeneousVectorSetMatrixMap2d;
    typedef HomogeneousVectorSetMatrixMap<float,3> HomogeneousVectorSetMatrixMap3f;
    typedef HomogeneousVectorSetMatrixMap<double,3> HomogeneousVectorSetMatrixMap3d;
    typedef HomogeneousVectorSetMatrixMap<float,Eigen::Dynamic> HomogeneousVectorSetMatrixMapXf;
    typedef HomogeneousVectorSetMatrixMap<double,Eigen::Dynamic> HomogeneousVectorSetMatrixMapXd;

    // Read-only Eigen Map (for inputs)
    template <typename ScalarT, ptrdiff_t EigenDim>
    class ConstDataMatrixMap : public Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>> Base;

        typedef ScalarT Scalar;

        enum { Dimension = EigenDim };

        ConstDataMatrixMap(const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic> &data)
                : Base(data.data(), data.rows(), data.cols())
        {}

        ConstDataMatrixMap(Eigen::Map<const Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>> data)
                : Base(data.data(), data.rows(), data.cols())
        {}

        ConstDataMatrixMap(Eigen::Map<Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>> data)
                : Base(data.data(), data.rows(), data.cols())
        {}

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        ConstDataMatrixMap(const std::vector<ScalarT> &data)
                : Base((const ScalarT *)data.data(), EigenDim, data.size()/EigenDim)
        {}

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim == Eigen::Dynamic>::type>
        ConstDataMatrixMap(const std::vector<ScalarT> &data, size_t dim)
                : Base((const ScalarT *)data.data(), dim, data.size()/dim)
        {}

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic && sizeof(Eigen::Matrix<ScalarT,Dim,1>) % 16 != 0>::type>
        ConstDataMatrixMap(const std::vector<Eigen::Matrix<ScalarT,EigenDim,1>,Eigen::aligned_allocator<Eigen::Matrix<ScalarT,EigenDim,1>>> &data)
                : Base((const ScalarT *)data.data(), EigenDim, data.size())
        {}

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        ConstDataMatrixMap(const std::vector<Eigen::Matrix<ScalarT,EigenDim,1>> &data)
                : Base((const ScalarT *)data.data(), EigenDim, data.size())
        {}

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        ConstDataMatrixMap(const ScalarT * data, size_t num_points = 0)
                : Base(data, EigenDim, num_points)
        {}

        ConstDataMatrixMap(const ScalarT * data, size_t dim, size_t num_points)
                : Base(data, dim, num_points)
        {}

        inline const Base& base() {
            return *static_cast<Base *>(this);
        }
    };

    typedef ConstDataMatrixMap<float,2> ConstDataMatrixMap2f;
    typedef ConstDataMatrixMap<double,2> ConstDataMatrixMap2d;
    typedef ConstDataMatrixMap<float,3> ConstDataMatrixMap3f;
    typedef ConstDataMatrixMap<double,3> ConstDataMatrixMap3d;
    typedef ConstDataMatrixMap<float,Eigen::Dynamic> ConstDataMatrixMapXf;
    typedef ConstDataMatrixMap<double,Eigen::Dynamic> ConstDataMatrixMapXd;

    template <typename ScalarT, ptrdiff_t EigenDim>
    using ConstVectorSetMatrixMap = ConstDataMatrixMap<ScalarT,EigenDim>;

    typedef ConstVectorSetMatrixMap<float,2> ConstVectorSetMatrixMap2f;
    typedef ConstVectorSetMatrixMap<double,2> ConstVectorSetMatrixMap2d;
    typedef ConstVectorSetMatrixMap<float,3> ConstVectorSetMatrixMap3f;
    typedef ConstVectorSetMatrixMap<double,3> ConstVectorSetMatrixMap3d;
    typedef ConstVectorSetMatrixMap<float,Eigen::Dynamic> ConstVectorSetMatrixMapXf;
    typedef ConstVectorSetMatrixMap<double,Eigen::Dynamic> ConstVectorSetMatrixMapXd;

    template <typename ScalarT, ptrdiff_t EigenDim>
    using ConstHomogeneousVectorSetMatrixMap = typename std::conditional<EigenDim == Eigen::Dynamic, ConstDataMatrixMap<ScalarT,EigenDim>, ConstDataMatrixMap<ScalarT,EigenDim+1>>::type;

    typedef ConstHomogeneousVectorSetMatrixMap<float,2> ConstHomogeneousVectorSetMatrixMap2f;
    typedef ConstHomogeneousVectorSetMatrixMap<double,2> ConstHomogeneousVectorSetMatrixMap2d;
    typedef ConstHomogeneousVectorSetMatrixMap<float,3> ConstHomogeneousVectorSetMatrixMap3f;
    typedef ConstHomogeneousVectorSetMatrixMap<double,3> ConstHomogeneousVectorSetMatrixMap3d;
    typedef ConstHomogeneousVectorSetMatrixMap<float,Eigen::Dynamic> ConstHomogeneousVectorSetMatrixMapXf;
    typedef ConstHomogeneousVectorSetMatrixMap<double,Eigen::Dynamic> ConstHomogeneousVectorSetMatrixMapXd;

    template <typename ScalarT, ptrdiff_t EigenDim>
    using Vector = Eigen::Matrix<ScalarT,EigenDim,1>;

    typedef Vector<float,2> Vector2f;
    typedef Vector<double,2> Vector2d;
    typedef Vector<float,3> Vector3f;
    typedef Vector<double,3> Vector3d;
    typedef Vector<float,Eigen::Dynamic> VectorXf;
    typedef Vector<double,Eigen::Dynamic> VectorXd;

    template <typename ScalarT, ptrdiff_t EigenDim>
    using VectorSet = Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>;

    typedef VectorSet<float,2> VectorSet2f;
    typedef VectorSet<double,2> VectorSet2d;
    typedef VectorSet<float,3> VectorSet3f;
    typedef VectorSet<double,3> VectorSet3d;
    typedef VectorSet<float,Eigen::Dynamic> VectorSetXf;
    typedef VectorSet<double,Eigen::Dynamic> VectorSetXd;

    template <typename ScalarT, ptrdiff_t EigenDim>
    using HomogeneousVector = typename std::conditional<EigenDim == Eigen::Dynamic, Eigen::Matrix<ScalarT,EigenDim,1>, Eigen::Matrix<ScalarT,EigenDim+1,1>>::type;

    typedef HomogeneousVector<float,2> HomogeneousVector2f;
    typedef HomogeneousVector<double,2> HomogeneousVector2d;
    typedef HomogeneousVector<float,3> HomogeneousVector3f;
    typedef HomogeneousVector<double,3> HomogeneousVector3d;
    typedef HomogeneousVector<float,Eigen::Dynamic> HomogeneousVectorXf;
    typedef HomogeneousVector<double,Eigen::Dynamic> HomogeneousVectorXd;

    template <typename ScalarT, ptrdiff_t EigenDim>
    using HomogeneousVectorSet = typename std::conditional<EigenDim == Eigen::Dynamic, Eigen::Matrix<ScalarT,EigenDim,Eigen::Dynamic>, Eigen::Matrix<ScalarT,EigenDim+1,Eigen::Dynamic>>::type;

    typedef HomogeneousVectorSet<float,2> HomogeneousVectorSet2f;
    typedef HomogeneousVectorSet<double,2> HomogeneousVectorSet2d;
    typedef HomogeneousVectorSet<float,3> HomogeneousVectorSet3f;
    typedef HomogeneousVectorSet<double,3> HomogeneousVectorSet3d;
    typedef HomogeneousVectorSet<float,Eigen::Dynamic> HomogeneousVectorSetXf;
    typedef HomogeneousVectorSet<double,Eigen::Dynamic> HomogeneousVectorSetXd;

    enum struct DataMatrixViewMode { Matrix, MatrixRef, ConstMatrixRef, MatrixMap, ConstMatrixMap };

    template <typename ScalarT, ptrdiff_t EigenDim, DataMatrixViewMode Mode>
    struct DataMatrixView {};

    template <typename ScalarT, ptrdiff_t EigenDim>
    struct DataMatrixView<ScalarT,EigenDim,DataMatrixViewMode::Matrix> { typedef VectorSet<ScalarT,EigenDim> Container; };

    template <typename ScalarT, ptrdiff_t EigenDim>
    struct DataMatrixView<ScalarT,EigenDim,DataMatrixViewMode::MatrixRef> { typedef VectorSet<ScalarT,EigenDim>& Container; };

    template <typename ScalarT, ptrdiff_t EigenDim>
    struct DataMatrixView<ScalarT,EigenDim,DataMatrixViewMode::ConstMatrixRef> { typedef const VectorSet<ScalarT,EigenDim>& Container; };

    template <typename ScalarT, ptrdiff_t EigenDim>
    struct DataMatrixView<ScalarT,EigenDim,DataMatrixViewMode::MatrixMap> { typedef DataMatrixMap<ScalarT,EigenDim> Container; };

    template <typename ScalarT, ptrdiff_t EigenDim>
    struct DataMatrixView<ScalarT,EigenDim,DataMatrixViewMode::ConstMatrixMap> { typedef ConstDataMatrixMap<ScalarT,EigenDim> Container; };
}
