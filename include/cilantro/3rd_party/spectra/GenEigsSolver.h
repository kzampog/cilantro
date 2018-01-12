// Copyright (C) 2016-2017 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef GEN_EIGS_SOLVER_H
#define GEN_EIGS_SOLVER_H

#include <Eigen/Core>
#include <vector>     // std::vector
#include <cmath>      // std::abs, std::pow
#include <algorithm>  // std::min, std::copy
#include <complex>    // std::complex, std::conj, std::norm, std::abs
#include <stdexcept>  // std::invalid_argument

#include "Util/SelectionRule.h"
#include "Util/CompInfo.h"
#include "Util/SimpleRandom.h"
#include "LinAlg/UpperHessenbergQR.h"
#include "LinAlg/UpperHessenbergEigen.h"
#include "LinAlg/DoubleShiftQR.h"
#include "MatOp/DenseGenMatProd.h"

namespace Spectra {


///
/// \ingroup EigenSolver
///
/// This class implements the eigen solver for general real matrices, i.e.,
/// to solve \f$Ax=\lambda x\f$ for a possibly non-symmetric \f$A\f$ matrix.
///
/// Most of the background information documented in the SymEigsSolver class
/// also applies to the GenEigsSolver class here, except that the eigenvalues
/// and eigenvectors of a general matrix can now be complex-valued.
///
/// \tparam Scalar        The element type of the matrix.
///                       Currently supported types are `float`, `double` and `long double`.
/// \tparam SelectionRule An enumeration value indicating the selection rule of
///                       the requested eigenvalues, for example `LARGEST_MAGN`
///                       to retrieve eigenvalues with the largest magnitude.
///                       The full list of enumeration values can be found in
///                       \ref Enumerations.
/// \tparam OpType        The name of the matrix operation class. Users could either
///                       use the wrapper classes such as DenseGenMatProd and
///                       SparseGenMatProd, or define their
///                       own that impelemnts all the public member functions as in
///                       DenseGenMatProd.
///
/// An example that illustrates the usage of GenEigsSolver is give below:
///
/// \code{.cpp}
/// #include <Eigen/Core>
/// #include <GenEigsSolver.h>  // Also includes <MatOp/DenseGenMatProd.h>
/// #include <iostream>
///
/// using namespace Spectra;
///
/// int main()
/// {
///     // We are going to calculate the eigenvalues of M
///     Eigen::MatrixXd M = Eigen::MatrixXd::Random(10, 10);
///
///     // Construct matrix operation object using the wrapper class
///     DenseGenMatProd<double> op(M);
///
///     // Construct eigen solver object, requesting the largest
///     // (in magnitude, or norm) three eigenvalues
///     GenEigsSolver< double, LARGEST_MAGN, DenseGenMatProd<double> > eigs(&op, 3, 6);
///
///     // Initialize and compute
///     eigs.init();
///     int nconv = eigs.compute();
///
///     // Retrieve results
///     Eigen::VectorXcd evalues;
///     if(eigs.info() == SUCCESSFUL)
///         evalues = eigs.eigenvalues();
///
///     std::cout << "Eigenvalues found:\n" << evalues << std::endl;
///
///     return 0;
/// }
/// \endcode
///
/// And also an example for sparse matrices:
///
/// \code{.cpp}
/// #include <Eigen/Core>
/// #include <Eigen/SparseCore>
/// #include <GenEigsSolver.h>
/// #include <MatOp/SparseGenMatProd.h>
/// #include <iostream>
///
/// using namespace Spectra;
///
/// int main()
/// {
///     // A band matrix with 1 on the main diagonal, 2 on the below-main subdiagonal,
///     // and 3 on the above-main subdiagonal
///     const int n = 10;
///     Eigen::SparseMatrix<double> M(n, n);
///     M.reserve(Eigen::VectorXi::Constant(n, 3));
///     for(int i = 0; i < n; i++)
///     {
///         M.insert(i, i) = 1.0;
///         if(i > 0)
///             M.insert(i - 1, i) = 3.0;
///         if(i < n - 1)
///             M.insert(i + 1, i) = 2.0;
///     }
///
///     // Construct matrix operation object using the wrapper class SparseGenMatProd
///     SparseGenMatProd<double> op(M);
///
///     // Construct eigen solver object, requesting the largest three eigenvalues
///     GenEigsSolver< double, LARGEST_MAGN, SparseGenMatProd<double> > eigs(&op, 3, 6);
///
///     // Initialize and compute
///     eigs.init();
///     int nconv = eigs.compute();
///
///     // Retrieve results
///     Eigen::VectorXcd evalues;
///     if(eigs.info() == SUCCESSFUL)
///         evalues = eigs.eigenvalues();
///
///     std::cout << "Eigenvalues found:\n" << evalues << std::endl;
///
///     return 0;
/// }
/// \endcode
template < typename Scalar = double,
           int SelectionRule = LARGEST_MAGN,
           typename OpType = DenseGenMatProd<double> >
class GenEigsSolver
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;
    typedef Eigen::Array<bool, Eigen::Dynamic, 1> BoolArray;
    typedef Eigen::Map<Matrix> MapMat;
    typedef Eigen::Map<Vector> MapVec;

    typedef std::complex<Scalar> Complex;
    typedef Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> ComplexMatrix;
    typedef Eigen::Matrix<Complex, Eigen::Dynamic, 1> ComplexVector;

protected:
    OpType *m_op;             // object to conduct matrix operation,
                              // e.g. matrix-vector product
    const int m_n;            // dimension of matrix A
    const int m_nev;          // number of eigenvalues requested

private:
    const int m_ncv;          // number of ritz values
    int m_nmatop;             // number of matrix operations called
    int m_niter;              // number of restarting iterations

protected:
    Matrix m_fac_V;           // V matrix in the Arnoldi factorization
    Matrix m_fac_H;           // H matrix in the Arnoldi factorization
    Vector m_fac_f;           // residual in the Arnoldi factorization

    ComplexVector m_ritz_val; // ritz values
    ComplexMatrix m_ritz_vec; // ritz vectors
    ComplexVector m_ritz_est; // last row of m_ritz_vec

private:
    BoolArray m_ritz_conv;    // indicator of the convergence of ritz values
    int m_info;               // status of the computation

    const Scalar m_eps;       // the machine precision,
                              // e.g. ~= 1e-16 for the "double" type
    const Scalar m_approx_0;  // a number that is approximately zero
                              // m_eps23 = m_eps^(2/3)
                              // used to test whether a number is complex, and
                              // to test the orthogonality of vectors

    // Arnoldi factorization starting from step-k
    void factorize_from(int from_k, int to_m, const Vector& fk)
    {
        if(to_m <= from_k) return;

        m_fac_f = fk;

        Vector w(m_n);
        Scalar beta = m_fac_f.norm();
        // Keep the upperleft k x k submatrix of H and set other elements to 0
        m_fac_H.rightCols(m_ncv - from_k).setZero();
        m_fac_H.block(from_k, 0, m_ncv - from_k, from_k).setZero();
        for(int i = from_k; i <= to_m - 1; i++)
        {
            bool restart = false;
            // If beta = 0, then the next V is not full rank
            // We need to generate a new residual vector that is orthogonal
            // to the current V, which we call a restart
            if(beta < m_eps)
            {
                SimpleRandom<Scalar> rng(2 * i);
                m_fac_f.noalias() = rng.random_vec(m_n);
                // f <- f - V * V' * f, so that f is orthogonal to V
                MapMat V(m_fac_V.data(), m_n, i); // The first i columns
                Vector Vf = V.transpose() * m_fac_f;
                m_fac_f.noalias() -= V * Vf;
                // beta <- ||f||
                beta = m_fac_f.norm();

                restart = true;
            }

            // v <- f / ||f||
            m_fac_V.col(i).noalias() = m_fac_f / beta; // The (i+1)-th column

            // Note that H[i+1, i] equals to the unrestarted beta
            if(restart)
                m_fac_H(i, i - 1) = 0.0;
            else
                m_fac_H(i, i - 1) = beta;

            // w <- A * v, v = m_fac_V.col(i)
            m_op->perform_op(&m_fac_V(0, i), w.data());
            m_nmatop++;

            // First i+1 columns of V
            MapMat Vs(m_fac_V.data(), m_n, i + 1);
            // h = m_fac_H(0:i, i)
            MapVec h(&m_fac_H(0, i), i + 1);
            // h <- V' * w
            h.noalias() = Vs.transpose() * w;

            // f <- w - V * h
            m_fac_f.noalias() = w - Vs * h;
            beta = m_fac_f.norm();

            if(beta > 0.717 * h.norm())
                continue;

            // f/||f|| is going to be the next column of V, so we need to test
            // whether V' * (f/||f||) ~= 0
            Vector Vf = Vs.transpose() * m_fac_f;
            // If not, iteratively correct the residual
            int count = 0;
            while(count < 5 && Vf.cwiseAbs().maxCoeff() > m_approx_0 * beta)
            {
                // f <- f - V * Vf
                m_fac_f.noalias() -= Vs * Vf;
                // h <- h + Vf
                h.noalias() += Vf;
                // beta <- ||f||
                beta = m_fac_f.norm();

                Vf.noalias() = Vs.transpose() * m_fac_f;
                count++;
            }
        }
    }

    // Real Ritz values calculated from UpperHessenbergEigen have exact zero imaginary part
    // Complex Ritz values have exact conjugate pairs
    // So we use exact tests here
    static bool is_complex(const Complex& v) { return v.imag() != Scalar(0); }
    static bool is_conj(const Complex& v1, const Complex& v2) { return v1 == Eigen::numext::conj(v2); }

    // Implicitly restarted Arnoldi factorization
    void restart(int k)
    {
        using std::norm;

        if(k >= m_ncv)
            return;

        DoubleShiftQR<Scalar> decomp_ds(m_ncv);
        UpperHessenbergQR<Scalar> decomp_hb;
        Matrix Q = Matrix::Identity(m_ncv, m_ncv);

        for(int i = k; i < m_ncv; i++)
        {
            if(is_complex(m_ritz_val[i]) && is_conj(m_ritz_val[i], m_ritz_val[i + 1]))
            {
                // H - mu * I = Q1 * R1
                // H <- R1 * Q1 + mu * I = Q1' * H * Q1
                // H - conj(mu) * I = Q2 * R2
                // H <- R2 * Q2 + conj(mu) * I = Q2' * H * Q2
                //
                // (H - mu * I) * (H - conj(mu) * I) = Q1 * Q2 * R2 * R1 = Q * R
                Scalar s = 2 * m_ritz_val[i].real();
                Scalar t = norm(m_ritz_val[i]);

                decomp_ds.compute(m_fac_H, s, t);

                // Q -> Q * Qi
                decomp_ds.apply_YQ(Q);
                // H -> Q'HQ
                // Matrix Q = Matrix::Identity(m_ncv, m_ncv);
                // decomp_ds.apply_YQ(Q);
                // m_fac_H = Q.transpose() * m_fac_H * Q;
                m_fac_H = decomp_ds.matrix_QtHQ();

                i++;
            } else {
                // QR decomposition of H - mu * I, mu is real
                m_fac_H.diagonal().array() -= m_ritz_val[i].real();
                decomp_hb.compute(m_fac_H);

                // Q -> Q * Qi
                decomp_hb.apply_YQ(Q);
                // H -> Q'HQ = RQ + mu * I
                m_fac_H = decomp_hb.matrix_RQ();
                m_fac_H.diagonal().array() += m_ritz_val[i].real();
            }
        }
        // V -> VQ, only need to update the first k+1 columns
        // Q has some elements being zero
        // The first (ncv - k + i) elements of the i-th column of Q are non-zero
        Matrix Vs(m_n, k + 1);
        int nnz;
        for(int i = 0; i < k; i++)
        {
            nnz = m_ncv - k + i + 1;
            MapMat V(m_fac_V.data(), m_n, nnz);
            MapVec q(&Q(0, i), nnz);
            Vs.col(i).noalias() = V * q;
        }
        Vs.col(k).noalias() = m_fac_V * Q.col(k);
        m_fac_V.leftCols(k + 1).noalias() = Vs;

        Vector fk = m_fac_f * Q(m_ncv - 1, k - 1) + m_fac_V.col(k) * m_fac_H(k, k - 1);
        factorize_from(k, m_ncv, fk);
        retrieve_ritzpair();
    }

    // Calculates the number of converged Ritz values
    int num_converged(Scalar tol)
    {
        // thresh = tol * max(m_approx_0, abs(theta)), theta for ritz value
        Array thresh = tol * m_ritz_val.head(m_nev).array().abs().max(m_approx_0);
        Array resid = m_ritz_est.head(m_nev).array().abs() * m_fac_f.norm();
        // Converged "wanted" ritz values
        m_ritz_conv = (resid < thresh);

        return m_ritz_conv.cast<int>().sum();
    }

    // Returns the adjusted nev for restarting
    int nev_adjusted(int nconv)
    {
        using std::abs;

        int nev_new = m_nev;
        for(int i = m_nev; i < m_ncv; i++)
            if(abs(m_ritz_est[i]) < m_eps)  nev_new++;

        // Adjust nev_new, according to dnaup2.f line 660~674 in ARPACK
        nev_new += std::min(nconv, (m_ncv - nev_new) / 2);
        if(nev_new == 1 && m_ncv >= 6)
            nev_new = m_ncv / 2;
        else if(nev_new == 1 && m_ncv > 3)
            nev_new = 2;

        if(nev_new > m_ncv - 2)
            nev_new = m_ncv - 2;

        // Increase nev by one if ritz_val[nev - 1] and
        // ritz_val[nev] are conjugate pairs
        if(is_complex(m_ritz_val[nev_new - 1]) &&
           is_conj(m_ritz_val[nev_new - 1], m_ritz_val[nev_new]))
        {
            nev_new++;
        }

        return nev_new;
    }

    // Retrieves and sorts ritz values and ritz vectors
    void retrieve_ritzpair()
    {
        UpperHessenbergEigen<Scalar> decomp(m_fac_H);
        ComplexVector evals = decomp.eigenvalues();
        ComplexMatrix evecs = decomp.eigenvectors();

        SortEigenvalue<Complex, SelectionRule> sorting(evals.data(), evals.size());
        std::vector<int> ind = sorting.index();

        // Copy the ritz values and vectors to m_ritz_val and m_ritz_vec, respectively
        for(int i = 0; i < m_ncv; i++)
        {
            m_ritz_val[i] = evals[ind[i]];
            m_ritz_est[i] = evecs(m_ncv - 1, ind[i]);
        }
        for(int i = 0; i < m_nev; i++)
        {
            m_ritz_vec.col(i) = evecs.col(ind[i]);
        }
    }

protected:
    // Sorts the first nev Ritz pairs in the specified order
    // This is used to return the final results
    virtual void sort_ritzpair(int sort_rule)
    {
        // First make sure that we have a valid index vector
        SortEigenvalue<Complex, LARGEST_MAGN> sorting(m_ritz_val.data(), m_nev);
        std::vector<int> ind = sorting.index();

        switch(sort_rule)
        {
            case LARGEST_MAGN:
                break;
            case LARGEST_REAL:
            {
                SortEigenvalue<Complex, LARGEST_REAL> sorting(m_ritz_val.data(), m_nev);
                ind = sorting.index();
            }
                break;
            case LARGEST_IMAG:
            {
                SortEigenvalue<Complex, LARGEST_IMAG> sorting(m_ritz_val.data(), m_nev);
                ind = sorting.index();
            }
                break;
            case SMALLEST_MAGN:
            {
                SortEigenvalue<Complex, SMALLEST_MAGN> sorting(m_ritz_val.data(), m_nev);
                ind = sorting.index();
            }
                break;
            case SMALLEST_REAL:
            {
                SortEigenvalue<Complex, SMALLEST_REAL> sorting(m_ritz_val.data(), m_nev);
                ind = sorting.index();
            }
                break;
            case SMALLEST_IMAG:
            {
                SortEigenvalue<Complex, SMALLEST_IMAG> sorting(m_ritz_val.data(), m_nev);
                ind = sorting.index();
            }
                break;
            default:
                throw std::invalid_argument("unsupported sorting rule");
        }

        ComplexVector new_ritz_val(m_ncv);
        ComplexMatrix new_ritz_vec(m_ncv, m_nev);
        BoolArray new_ritz_conv(m_nev);

        for(int i = 0; i < m_nev; i++)
        {
            new_ritz_val[i] = m_ritz_val[ind[i]];
            new_ritz_vec.col(i) = m_ritz_vec.col(ind[i]);
            new_ritz_conv[i] = m_ritz_conv[ind[i]];
        }

        m_ritz_val.swap(new_ritz_val);
        m_ritz_vec.swap(new_ritz_vec);
        m_ritz_conv.swap(new_ritz_conv);
    }

public:
    ///
    /// Constructor to create a solver object.
    ///
    /// \param op_  Pointer to the matrix operation object, which should implement
    ///             the matrix-vector multiplication operation of \f$A\f$:
    ///             calculating \f$Av\f$ for any vector \f$v\f$. Users could either
    ///             create the object from the wrapper class such as DenseGenMatProd, or
    ///             define their own that impelemnts all the public member functions
    ///             as in DenseGenMatProd.
    /// \param nev_ Number of eigenvalues requested. This should satisfy \f$1\le nev \le n-2\f$,
    ///             where \f$n\f$ is the size of matrix.
    /// \param ncv_ Parameter that controls the convergence speed of the algorithm.
    ///             Typically a larger `ncv_` means faster convergence, but it may
    ///             also result in greater memory use and more matrix operations
    ///             in each iteration. This parameter must satisfy \f$nev+2 \le ncv \le n\f$,
    ///             and is advised to take \f$ncv \ge 2\cdot nev + 1\f$.
    ///
    GenEigsSolver(OpType* op_, int nev_, int ncv_) :
        m_op(op_),
        m_n(m_op->rows()),
        m_nev(nev_),
        m_ncv(ncv_ > m_n ? m_n : ncv_),
        m_nmatop(0),
        m_niter(0),
        m_info(NOT_COMPUTED),
        m_eps(Eigen::NumTraits<Scalar>::epsilon()),
        m_approx_0(Eigen::numext::pow(m_eps, Scalar(2.0) / 3))
    {
        if(nev_ < 1 || nev_ > m_n - 2)
            throw std::invalid_argument("nev must satisfy 1 <= nev <= n - 2, n is the size of matrix");

        if(ncv_ < nev_ + 2 || ncv_ > m_n)
            throw std::invalid_argument("ncv must satisfy nev + 2 <= ncv <= n, n is the size of matrix");
    }

    ///
    /// Virtual destructor
    ///
    virtual ~GenEigsSolver() {}

    ///
    /// Initializes the solver by providing an initial residual vector.
    ///
    /// \param init_resid Pointer to the initial residual vector.
    ///
    /// **Spectra** (and also **ARPACK**) uses an iterative algorithm
    /// to find eigenvalues. This function allows the user to provide the initial
    /// residual vector.
    ///
    void init(const Scalar* init_resid)
    {
        // Reset all matrices/vectors to zero
        m_fac_V.resize(m_n, m_ncv);
        m_fac_H.resize(m_ncv, m_ncv);
        m_fac_f.resize(m_n);
        m_ritz_val.resize(m_ncv);
        m_ritz_vec.resize(m_ncv, m_nev);
        m_ritz_est.resize(m_ncv);
        m_ritz_conv.resize(m_nev);

        m_fac_V.setZero();
        m_fac_H.setZero();
        m_fac_f.setZero();
        m_ritz_val.setZero();
        m_ritz_vec.setZero();
        m_ritz_est.setZero();
        m_ritz_conv.setZero();

        // Set the initial vector
        Vector v(m_n);
        std::copy(init_resid, init_resid + m_n, v.data());
        Scalar vnorm = v.norm();
        if(vnorm < m_eps)
            throw std::invalid_argument("initial residual vector cannot be zero");
        v /= vnorm;

        Vector w(m_n);
        m_op->perform_op(v.data(), w.data());
        m_nmatop++;

        m_fac_H(0, 0) = v.dot(w);
        m_fac_f = w - v * m_fac_H(0, 0);
        m_fac_V.col(0) = v;
    }

    ///
    /// Initializes the solver by providing a random initial residual vector.
    ///
    /// This overloaded function generates a random initial residual vector
    /// (with a fixed random seed) for the algorithm. Elements in the vector
    /// follow independent Uniform(-0.5, 0.5) distribution.
    ///
    void init()
    {
        SimpleRandom<Scalar> rng(0);
        Vector init_resid = rng.random_vec(m_n);
        init(init_resid.data());
    }

    ///
    /// Conducts the major computation procedure.
    ///
    /// \param maxit      Maximum number of iterations allowed in the algorithm.
    /// \param tol        Precision parameter for the calculated eigenvalues.
    /// \param sort_rule  Rule to sort the eigenvalues and eigenvectors.
    ///                   Supported values are
    ///                   `Spectra::LARGEST_MAGN`, `Spectra::LARGEST_REAL`,
    ///                   `Spectra::LARGEST_IMAG`, `Spectra::SMALLEST_MAGN`,
    ///                   `Spectra::SMALLEST_REAL` and `Spectra::SMALLEST_IMAG`,
    ///                   for example `LARGEST_MAGN` indicates that eigenvalues
    ///                   with largest magnitude come first.
    ///                   Note that this argument is only used to
    ///                   **sort** the final result, and the **selection** rule
    ///                   (e.g. selecting the largest or smallest eigenvalues in the
    ///                   full spectrum) is specified by the template parameter
    ///                   `SelectionRule` of GenEigsSolver.
    ///
    /// \return Number of converged eigenvalues.
    ///
    int compute(int maxit = 1000, Scalar tol = 1e-10, int sort_rule = LARGEST_MAGN)
    {
        // The m-step Arnoldi factorization
        factorize_from(1, m_ncv, m_fac_f);
        retrieve_ritzpair();
        // Restarting
        int i, nconv = 0, nev_adj;
        for(i = 0; i < maxit; i++)
        {
            nconv = num_converged(tol);
            if(nconv >= m_nev)
                break;

            nev_adj = nev_adjusted(nconv);
            restart(nev_adj);
        }
        // Sorting results
        sort_ritzpair(sort_rule);

        m_niter += i + 1;
        m_info = (nconv >= m_nev) ? SUCCESSFUL : NOT_CONVERGING;

        return std::min(m_nev, nconv);
    }

    ///
    /// Returns the status of the computation.
    /// The full list of enumeration values can be found in \ref Enumerations.
    ///
    int info() const { return m_info; }

    ///
    /// Returns the number of iterations used in the computation.
    ///
    int num_iterations() const { return m_niter; }

    ///
    /// Returns the number of matrix operations used in the computation.
    ///
    int num_operations() const { return m_nmatop; }

    ///
    /// Returns the converged eigenvalues.
    ///
    /// \return A complex-valued vector containing the eigenvalues.
    /// Returned vector type will be `Eigen::Vector<std::complex<Scalar>, ...>`, depending on
    /// the template parameter `Scalar` defined.
    ///
    ComplexVector eigenvalues() const
    {
        int nconv = m_ritz_conv.cast<int>().sum();
        ComplexVector res(nconv);

        if(!nconv)
            return res;

        int j = 0;
        for(int i = 0; i < m_nev; i++)
        {
            if(m_ritz_conv[i])
            {
                res[j] = m_ritz_val[i];
                j++;
            }
        }

        return res;
    }

    ///
    /// Returns the eigenvectors associated with the converged eigenvalues.
    ///
    /// \param nvec The number of eigenvectors to return.
    ///
    /// \return A complex-valued matrix containing the eigenvectors.
    /// Returned matrix type will be `Eigen::Matrix<std::complex<Scalar>, ...>`,
    /// depending on the template parameter `Scalar` defined.
    ///
    ComplexMatrix eigenvectors(int nvec) const
    {
        int nconv = m_ritz_conv.cast<int>().sum();
        nvec = std::min(nvec, nconv);
        ComplexMatrix res(m_n, nvec);

        if(!nvec)
            return res;

        ComplexMatrix ritz_vec_conv(m_ncv, nvec);
        int j = 0;
        for(int i = 0; i < m_nev && j < nvec; i++)
        {
            if(m_ritz_conv[i])
            {
                ritz_vec_conv.col(j) = m_ritz_vec.col(i);
                j++;
            }
        }

        res.noalias() = m_fac_V * ritz_vec_conv;

        return res;
    }

    ///
    /// Returns all converged eigenvectors.
    ///
    ComplexMatrix eigenvectors() const
    {
        return eigenvectors(m_nev);
    }
};


} // namespace Spectra

#endif // GEN_EIGS_SOLVER_H
