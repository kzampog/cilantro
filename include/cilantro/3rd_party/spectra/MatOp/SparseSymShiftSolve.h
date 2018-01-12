// Copyright (C) 2016-2017 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SPARSE_SYM_SHIFT_SOLVE_H
#define SPARSE_SYM_SHIFT_SOLVE_H

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <stdexcept>

namespace Spectra {


///
/// \ingroup MatOp
///
/// This class defines the shift-solve operation on a sparse real symmetric matrix \f$A\f$,
/// i.e., calculating \f$y=(A-\sigma I)^{-1}x\f$ for any real \f$\sigma\f$ and
/// vector \f$x\f$. It is mainly used in the SymEigsShiftSolver eigen solver.
///
template <typename Scalar, int Uplo = Eigen::Lower, int Flags = 0, typename StorageIndex = int>
class SparseSymShiftSolve
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Vector> MapConstVec;
    typedef Eigen::Map<Vector> MapVec;
    typedef Eigen::SparseMatrix<Scalar, Flags, StorageIndex> SparseMatrix;

    const SparseMatrix& m_mat;
    const int m_n;
    Eigen::SimplicialLDLT<SparseMatrix, Uplo> m_solver;

public:
    ///
    /// Constructor to create the matrix operation object.
    ///
    /// \param mat_ An **Eigen** sparse matrix object, whose type is
    /// `Eigen::SparseMatrix<Scalar, ...>`.
    ///
    SparseSymShiftSolve(const SparseMatrix& mat_) :
        m_mat(mat_),
        m_n(mat_.rows())
    {
        if(mat_.rows() != mat_.cols())
            throw std::invalid_argument("SparseSymShiftSolve: matrix must be square");
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
        m_solver.setShift(-sigma);
        m_solver.compute(m_mat);
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

#endif // SPARSE_SYM_SHIFT_SOLVE_H
