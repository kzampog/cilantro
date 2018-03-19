// Copyright (C) 2016-2018 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef GEN_EIGS_COMPLEX_SHIFT_SOLVER_H
#define GEN_EIGS_COMPLEX_SHIFT_SOLVER_H

#include "GenEigsSolver.h"
#include "MatOp/DenseGenComplexShiftSolve.h"

namespace Spectra {


///
/// \ingroup EigenSolver
///
/// This class implements the eigen solver for general real matrices with
/// a complex shift value in the **shift-and-invert mode**. The background
/// knowledge of the shift-and-invert mode can be found in the documentation
/// of the SymEigsShiftSolver class.
///
/// \tparam Scalar        The element type of the matrix.
///                       Currently supported types are `float`, `double` and `long double`.
/// \tparam SelectionRule An enumeration value indicating the selection rule of
///                       the shifted-and-inverted eigenvalues.
///                       The full list of enumeration values can be found in
///                       \ref Enumerations.
/// \tparam OpType        The name of the matrix operation class. Users could either
///                       use the DenseGenComplexShiftSolve wrapper class, or define their
///                       own that impelemnts all the public member functions as in
///                       DenseGenComplexShiftSolve.
///
template <typename Scalar = double,
          int SelectionRule = LARGEST_MAGN,
          typename OpType = DenseGenComplexShiftSolve<double> >
class GenEigsComplexShiftSolver: public GenEigsSolver<Scalar, SelectionRule, OpType>
{
private:
    typedef std::complex<Scalar> Complex;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Complex, Eigen::Dynamic, 1> ComplexVector;


    Scalar sigmar;
    Scalar sigmai;

    // First transform back the ritz values, and then sort
    void sort_ritzpair(int sort_rule)
    {
        using std::abs;
        using std::sqrt;
        using std::norm;

        // The eigenvalus we get from the iteration is
        //     nu = 0.5 * (1 / (lambda - sigma) + 1 / (lambda - conj(sigma)))
        // So the eigenvalues of the original problem is
        //                       1 \pm sqrt(1 - 4 * nu^2 * sigmai^2)
        //     lambda = sigmar + -----------------------------------
        //                                     2 * nu
        // We need to pick the correct root
        // Let (lambdaj, vj) be the j-th eigen pair, then A * vj = lambdaj * vj
        // and inv(A - r * I) * vj = 1 / (lambdaj - r) * vj
        // where r is any shift value.
        // We can use this identity to determine lambdaj
        //
        // op(v) computes Re(inv(A - r * I) * v) for any real v
        // If r is real, then op(v) is also real. Let a = Re(vj), b = Im(vj),
        // then op(vj) = op(a) + op(b) * i
        // By comparing op(vj) and [1 / (lambdaj - r) * vj], we can determine
        // which one is the correct root

        // Select a random shift value
        SimpleRandom<Scalar> rng(0);
        Scalar shiftr = rng.random() * sigmar + rng.random();
        Complex shift = Complex(shiftr, Scalar(0));
        this->m_op->set_shift(shiftr, Scalar(0));

        // Calculate inv(A - r * I) * vj
        Vector v_real(this->m_n), v_imag(this->m_n), OPv_real(this->m_n), OPv_imag(this->m_n);
        Scalar eps = Eigen::NumTraits<Scalar>::epsilon();
        for(int i = 0; i < this->m_nev; i++)
        {
            v_real.noalias() = this->m_fac_V * this->m_ritz_vec.col(i).real();
            v_imag.noalias() = this->m_fac_V * this->m_ritz_vec.col(i).imag();
            this->m_op->perform_op(v_real.data(), OPv_real.data());
            this->m_op->perform_op(v_imag.data(), OPv_imag.data());

            // Two roots computed from the quadratic equation
            Complex nu = this->m_ritz_val[i];
            Complex root_part1 = sigmar + Scalar(0.5) / nu;
            Complex root_part2 = Scalar(0.5) * sqrt(Scalar(1) - Scalar(4) * sigmai * sigmai * (nu * nu)) / nu;
            Complex root1 = root_part1 + root_part2;
            Complex root2 = root_part1 - root_part2;

            // Test roots
            Scalar err1 = Scalar(0), err2 = Scalar(0);
            for(int k = 0; k < this->m_n; k++)
            {
                Complex rhs1 = Complex(v_real[k], v_imag[k]) / (root1 - shift);
                Complex rhs2 = Complex(v_real[k], v_imag[k]) / (root2 - shift);
                Complex OPv = Complex(OPv_real[k], OPv_imag[k]);
                err1 += norm(OPv - rhs1);
                err2 += norm(OPv - rhs2);
            }

            Complex lambdaj = (err1 < err2) ? root1 : root2;
            this->m_ritz_val[i] = lambdaj;

            if(abs(Eigen::numext::imag(lambdaj)) > eps)
            {
                this->m_ritz_val[i + 1] = Eigen::numext::conj(lambdaj);
                i++;
            } else {
                this->m_ritz_val[i] = Complex(Eigen::numext::real(lambdaj), Scalar(0));
            }
        }

        GenEigsSolver<Scalar, SelectionRule, OpType>::sort_ritzpair(sort_rule);
    }
public:
    ///
    /// Constructor to create a eigen solver object using the shift-and-invert mode.
    ///
    /// \param op_     Pointer to the matrix operation object. This class should implement
    ///                the complex shift-solve operation of \f$A\f$: calculating
    ///                \f$\mathrm{Re}\{(A-\sigma I)^{-1}v\}\f$ for any vector \f$v\f$. Users could either
    ///                create the object from the DenseGenComplexShiftSolve wrapper class, or
    ///                define their own that impelemnts all the public member functions
    ///                as in DenseGenComplexShiftSolve.
    /// \param nev_    Number of eigenvalues requested. This should satisfy \f$1\le nev \le n-2\f$,
    ///                where \f$n\f$ is the size of matrix.
    /// \param ncv_    Parameter that controls the convergence speed of the algorithm.
    ///                Typically a larger `ncv_` means faster convergence, but it may
    ///                also result in greater memory use and more matrix operations
    ///                in each iteration. This parameter must satisfy \f$nev+2 \le ncv \le n\f$,
    ///                and is advised to take \f$ncv \ge 2\cdot nev + 1\f$.
    /// \param sigmar_ The real part of the shift.
    /// \param sigmai_ The imaginary part of the shift.
    ///
    GenEigsComplexShiftSolver(OpType* op_, int nev_, int ncv_, Scalar sigmar_, Scalar sigmai_) :
        GenEigsSolver<Scalar, SelectionRule, OpType>(op_, nev_, ncv_),
        sigmar(sigmar_), sigmai(sigmai_)
    {
        this->m_op->set_shift(sigmar, sigmai);
    }
};


} // namespace Spectra

#endif // GEN_EIGS_COMPLEX_SHIFT_SOLVER_H
