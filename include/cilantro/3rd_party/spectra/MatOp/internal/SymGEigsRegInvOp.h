// Copyright (C) 2017-2018 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SYM_GEIGS_REG_INV_OP_H
#define SYM_GEIGS_REG_INV_OP_H

#include <Eigen/Core>
#include "../SparseSymMatProd.h"
#include "../SparseRegularInverse.h"

namespace Spectra {


///
/// \ingroup MatOp
///
/// This class defines the matrix operation for generalized eigen solver in the
/// regular inverse mode. This class is intended for internal use.
///
template < typename Scalar = double,
           typename OpType = SparseSymMatProd<double>,
           typename BOpType = SparseRegularInverse<double> >
class SymGEigsRegInvOp
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    OpType&  m_op;
    BOpType& m_Bop;
    Vector   m_cache;  // temporary working space

public:
    ///
    /// Constructor to create the matrix operation object.
    ///
    /// \param op_  Pointer to the \f$A\f$ matrix operation object.
    /// \param Bop_ Pointer to the \f$B\f$ matrix operation object.
    ///
    SymGEigsRegInvOp(OpType& op_, BOpType& Bop_) :
        m_op(op_), m_Bop(Bop_), m_cache(op_.rows())
    {}

    ///
    /// Return the number of rows of the underlying matrix.
    ///
    int rows() const { return m_Bop.rows(); }
    ///
    /// Return the number of columns of the underlying matrix.
    ///
    int cols() const { return m_Bop.rows(); }

    ///
    /// Perform the matrix operation \f$y=B^{-1}Ax\f$.
    ///
    /// \param x_in  Pointer to the \f$x\f$ vector.
    /// \param y_out Pointer to the \f$y\f$ vector.
    ///
    // y_out = inv(B) * A * x_in
    void perform_op(const Scalar* x_in, Scalar* y_out)
    {
        m_op.perform_op(x_in, m_cache.data());
        m_Bop.solve(m_cache.data(), y_out);
    }
};


} // namespace Spectra

#endif // SYM_GEIGS_REG_INV_OP_H
