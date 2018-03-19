// Copyright (C) 2016-2018 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef DENSE_SYM_SHIFT_SOLVE_H
#define DENSE_SYM_SHIFT_SOLVE_H

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <stdexcept>

namespace Spectra {


///
/// \ingroup MatOp
///
/// This class defines the shift-solve operation on a real symmetric matrix \f$A\f$,
/// i.e., calculating \f$y=(A-\sigma I)^{-1}x\f$ for any real \f$\sigma\f$ and
/// vector \f$x\f$. It is mainly used in the SymEigsShiftSolver eigen solver.
///
template <typename Scalar, int Uplo = Eigen::Lower>
class DenseSymShiftSolve
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Matrix> MapConstMat;
    typedef Eigen::Map<const Vector> MapConstVec;
    typedef Eigen::Map<Vector> MapVec;

    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;

    const MapConstMat m_mat;
    const int m_n;
    Eigen::LDLT<Matrix, Uplo> m_solver;

public:
    ///
    /// Constructor to create the matrix operation object.
    ///
    /// \param mat_ An **Eigen** matrix object, whose type can be
    /// `Eigen::Matrix<Scalar, ...>` (e.g. `Eigen::MatrixXd` and
    /// `Eigen::MatrixXf`), or its mapped version
    /// (e.g. `Eigen::Map<Eigen::MatrixXd>`).
    ///
    DenseSymShiftSolve(ConstGenericMatrix& mat_) :
        m_mat(mat_.data(), mat_.rows(), mat_.cols()),
        m_n(mat_.rows())
    {
        if(mat_.rows() != mat_.cols())
            throw std::invalid_argument("DenseSymShiftSolve: matrix must be square");
    }

    ///
    /// Return the number of rows of the underlying matrix.
    ///
    int rows() const { return m_n; }
    ///
    /// Return the number of columns of the underlying matrix.
    ///
    int cols() const { return m_n; }

    ///
    /// Set the real shift \f$\sigma\f$.
    ///
    void set_shift(Scalar sigma)
    {
        m_solver.compute(m_mat - sigma * Matrix::Identity(m_n, m_n));
    }

    ///
    /// Perform the shift-solve operation \f$y=(A-\sigma I)^{-1}x\f$.
    ///
    /// \param x_in  Pointer to the \f$x\f$ vector.
    /// \param y_out Pointer to the \f$y\f$ vector.
    ///
    // y_out = inv(A - sigma * I) * x_in
    void perform_op(const Scalar* x_in, Scalar* y_out) const
    {
        MapConstVec x(x_in,  m_n);
        MapVec      y(y_out, m_n);
        y.noalias() = m_solver.solve(x);
    }
};


} // namespace Spectra

#endif // DENSE_SYM_SHIFT_SOLVE_H
