#pragma once

#include <Eigen/Sparse>
#include <cilantro/core/space_transformations.hpp>
#include <cilantro/core/common_pair_evaluators.hpp>

namespace cilantro {
    namespace internal {
        template <typename ScalarT>
        inline ScalarT sqrtHuberLoss(ScalarT x, ScalarT delta = (ScalarT)1.0) {
            const ScalarT x_abs = std::abs(x);
            if (x_abs > delta) {
                return std::sqrt(delta*(x_abs - (ScalarT)(0.5)*delta));
            } else {
                return std::sqrt((ScalarT)(0.5))*x_abs;
            }
        }

        template <typename ScalarT>
        inline ScalarT sqrtHuberLossDerivative(ScalarT x, ScalarT delta = (ScalarT)1.0) {
            const ScalarT x_abs = std::abs(x);
            if (x < (ScalarT)0.0) {
                if (x_abs > delta) {
                    return -delta/((ScalarT)(2.0)*std::sqrt(delta*(x_abs - (ScalarT)(0.5)*delta)));
                } else {
                    return -std::sqrt((ScalarT)(0.5));
                }
            } else {
                if (x_abs > delta) {
                    return delta/((ScalarT)(2.0)*std::sqrt(delta*(x_abs - (ScalarT)(0.5)*delta)));
                } else {
                    return std::sqrt((ScalarT)0.5);
                }
            }
        }

        template <typename ScalarT>
        void computeRotationTerms(ScalarT a, ScalarT b, ScalarT c,
                                  Eigen::Matrix<ScalarT,3,3> &rot_coeffs,
                                  Eigen::Matrix<ScalarT,3,3> &d_rot_coeffs_da,
                                  Eigen::Matrix<ScalarT,3,3> &d_rot_coeffs_db,
                                  Eigen::Matrix<ScalarT,3,3> &d_rot_coeffs_dc)
        {
            const ScalarT sina = std::sin(a);
            const ScalarT cosa = std::cos(a);
            const ScalarT sinb = std::sin(b);
            const ScalarT cosb = std::cos(b);
            const ScalarT sinc = std::sin(c);
            const ScalarT cosc = std::cos(c);

            rot_coeffs(0,0) = cosc*cosb;
            rot_coeffs(1,0) = -sinc*cosa + cosc*sinb*sina;
            rot_coeffs(2,0) = sinc*sina + cosc*sinb*cosa;
            rot_coeffs(0,1) = sinc*cosb;
            rot_coeffs(1,1) = cosc*cosa + sinc*sinb*sina;
            rot_coeffs(2,1) = -cosc*sina + sinc*sinb*cosa;
            rot_coeffs(0,2) = -sinb;
            rot_coeffs(1,2) = cosb*sina;
            rot_coeffs(2,2) = cosb*cosa;

            d_rot_coeffs_da(0,0) = (ScalarT)0.0;
            d_rot_coeffs_da(1,0) = sinc*sina + cosc*sinb*cosa;
            d_rot_coeffs_da(2,0) = sinc*cosa - cosc*sinb*sina;
            d_rot_coeffs_da(0,1) = (ScalarT)0.0;
            d_rot_coeffs_da(1,1) = -cosc*sina + sinc*sinb*cosa;
            d_rot_coeffs_da(2,1) = -cosc*cosa - sinc*sinb*sina;
            d_rot_coeffs_da(0,2) = (ScalarT)0.0;
            d_rot_coeffs_da(1,2) = cosb*cosa;
            d_rot_coeffs_da(2,2) = -cosb*sina;

            d_rot_coeffs_db(0,0) = -cosc*sinb;
            d_rot_coeffs_db(1,0) = cosc*cosb*sina;
            d_rot_coeffs_db(2,0) = cosc*cosb*cosa;
            d_rot_coeffs_db(0,1) = -sinc*sinb;
            d_rot_coeffs_db(1,1) = sinc*cosb*sina;
            d_rot_coeffs_db(2,1) = sinc*cosb*cosa;
            d_rot_coeffs_db(0,2) = -cosb;
            d_rot_coeffs_db(1,2) = -sinb*sina;
            d_rot_coeffs_db(2,2) = -sinb*cosa;

            d_rot_coeffs_dc(0,0) = -sinc*cosb;
            d_rot_coeffs_dc(1,0) = -cosc*cosa - sinc*sinb*sina;
            d_rot_coeffs_dc(2,0) = cosc*sina - sinc*sinb*cosa;
            d_rot_coeffs_dc(0,1) = cosc*cosb;
            d_rot_coeffs_dc(1,1) = -sinc*cosa + cosc*sinb*sina;
            d_rot_coeffs_dc(2,1) = sinc*sina + cosc*sinb*cosa;
            d_rot_coeffs_dc(0,2) = (ScalarT)0.0;
            d_rot_coeffs_dc(1,2) = (ScalarT)0.0;
            d_rot_coeffs_dc(2,2) = (ScalarT)0.0;
        }
    } // namespace internal

    // Locally rigid dense warp field, 2D
    template <class TransformT, class PointCorrSetT, class PlaneCorrSetT, class RegNeighborhoodSetT, class PointCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class PlaneCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class RegWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>>
    typename std::enable_if<int(TransformT::Mode) == int(Eigen::Isometry) && TransformT::Dim == 2,bool>::type
    estimateDenseWarpFieldCombinedMetric(const ConstVectorSetMatrixMap<typename TransformT::Scalar,2> &dst_p,
                                         const ConstVectorSetMatrixMap<typename TransformT::Scalar,2> &dst_n,
                                         const ConstVectorSetMatrixMap<typename TransformT::Scalar,2> &src_p,
                                         const PointCorrSetT &point_to_point_correspondences,
                                         typename TransformT::Scalar point_to_point_weight,
                                         const PlaneCorrSetT &point_to_plane_correspondences,
                                         typename TransformT::Scalar point_to_plane_weight,
                                         const RegNeighborhoodSetT &regularization_neighborhoods,
                                         typename TransformT::Scalar regularization_weight,
                                         TransformSet<TransformT> &transforms,
                                         typename TransformT::Scalar huber_boundary = (typename TransformT::Scalar)(1e-4),
                                         size_t max_gn_iter = 10,
                                         typename TransformT::Scalar gn_conv_tol = (typename TransformT::Scalar)1e-5,
                                         size_t max_cg_iter = 1000,
                                         typename TransformT::Scalar cg_conv_tol = (typename TransformT::Scalar)1e-5,
                                         const PointCorrWeightEvaluatorT &point_corr_evaluator = PointCorrWeightEvaluatorT(),
                                         const PlaneCorrWeightEvaluatorT &plane_corr_evaluator = PlaneCorrWeightEvaluatorT(),
                                         const RegWeightEvaluatorT &reg_evaluator = RegWeightEvaluatorT())
    {
        typedef typename TransformT::Scalar ScalarT;

        const bool has_point_to_point_terms = !point_to_point_correspondences.empty() && (point_to_point_weight > (ScalarT)0.0);
        const bool has_point_to_plane_terms = !point_to_plane_correspondences.empty() && (point_to_plane_weight > (ScalarT)0.0);

        if ((!has_point_to_point_terms && !has_point_to_plane_terms) ||
            (has_point_to_plane_terms && dst_p.cols() != dst_n.cols()))
        {
            transforms.resize(src_p.cols());
            transforms.setIdentity();
            return false;
        }

        // Get regularization equation count and indices
        std::vector<size_t> reg_eq_ind(regularization_neighborhoods.size());
        size_t num_reg_arcs = 0;
        if (!regularization_neighborhoods.empty()) {
            reg_eq_ind[0] = 0;
            num_reg_arcs = std::max((size_t)0, regularization_neighborhoods[0].size() - 1);
        }
        for (size_t i = 1; i < regularization_neighborhoods.size(); i++) {
            reg_eq_ind[i] = reg_eq_ind[i-1] + 3*std::max((size_t)0, regularization_neighborhoods[i-1].size() - 1);
            num_reg_arcs += std::max((size_t)0, regularization_neighborhoods[i].size() - 1);
        }

        // Compute number of equations and unknowns
        const size_t num_unknowns = 3*src_p.cols();
        const size_t num_point_to_point_equations = 2*has_point_to_point_terms*point_to_point_correspondences.size();
        const size_t num_point_to_plane_equations = has_point_to_plane_terms*point_to_plane_correspondences.size();
        const size_t num_data_term_equations = num_point_to_point_equations + num_point_to_plane_equations;
        const size_t num_regularization_equations = 3*num_reg_arcs;
        const size_t num_equations = num_data_term_equations + num_regularization_equations;
        const size_t num_non_zeros = 3*num_data_term_equations + 2*num_regularization_equations;

        // Jacobian
        Eigen::SparseMatrix<ScalarT> At(num_unknowns, num_equations);
        At.reserve(num_non_zeros);
        // Values
        ScalarT * const values = At.valuePtr();
        // Outer pointers
        typename Eigen::SparseMatrix<ScalarT>::StorageIndex * const outer_ptr = At.outerIndexPtr();
#pragma omp parallel
        {
#pragma omp for nowait
            for (size_t i = 0; i < num_data_term_equations + 1; i++) {
                outer_ptr[i] = 3*i;
            }
#pragma omp for nowait
            for (size_t i = 1; i < num_regularization_equations + 1; i++) {
                outer_ptr[num_data_term_equations + i] = 3*num_data_term_equations + 2*i;
            }
        }
        // Inner indices
        typename Eigen::SparseMatrix<ScalarT>::StorageIndex * const inner_ind = At.innerIndexPtr();

        // Vector of (negative) residuals
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> b(num_equations);

        // Vector of unknowns (rotation angle and translation offsets per point)
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> tforms_vec(Eigen::Matrix<ScalarT,Eigen::Dynamic,1>::Zero(num_unknowns, 1));

        // Conjugate Gradient solver
//        Eigen::ConjugateGradient<Eigen::SparseMatrix<ScalarT>,Eigen::Lower|Eigen::Upper,Eigen::IncompleteCholesky<ScalarT,Eigen::Lower|Eigen::Upper>> solver;
//        Eigen::ConjugateGradient<Eigen::SparseMatrix<ScalarT>,Eigen::Lower|Eigen::Upper,Eigen::IdentityPreconditioner> solver;
        Eigen::ConjugateGradient<Eigen::SparseMatrix<ScalarT>,Eigen::Lower|Eigen::Upper,Eigen::DiagonalPreconditioner<ScalarT>> solver;
        solver.setMaxIterations(max_cg_iter);
        solver.setTolerance(cg_conv_tol);

        Eigen::SparseMatrix<ScalarT> AtA;
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> Atb;

        // Parameters
        const ScalarT point_to_point_weight_sqrt = std::sqrt(point_to_point_weight);
        const ScalarT point_to_plane_weight_sqrt = std::sqrt(point_to_plane_weight);
        const ScalarT regularization_weight_sqrt = std::sqrt(regularization_weight);
        const ScalarT gn_conv_tol_sq = gn_conv_tol*gn_conv_tol;

        // Temporaries
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> delta;
        ScalarT weight, diff, d_sqrt_huber_loss, curr_delta_sq, max_delta_sq;
        size_t eq_ind, nz_ind;

        bool has_converged = false;
        size_t iter = 0;
        while (iter < max_gn_iter) {
#pragma omp parallel shared (At, b) private (eq_ind, nz_ind, weight, diff, d_sqrt_huber_loss)
            {
                // Data term
                if (has_point_to_point_terms) {
#pragma omp for nowait
                    for (size_t i = 0; i < point_to_point_correspondences.size(); i++) {
                        const auto& corr = point_to_point_correspondences[i];
                        const auto d = dst_p.col(corr.indexInFirst);
                        const auto s = src_p.col(corr.indexInSecond);
                        const auto offset = 3*corr.indexInSecond;
                        weight = point_to_point_weight_sqrt*std::sqrt(point_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value));

                        const ScalarT cosa = std::cos(tforms_vec[offset]);
                        const ScalarT sina = std::sin(tforms_vec[offset]);

                        Eigen::Matrix<ScalarT,2,1> s_t(cosa*s[0] - sina*s[1] + tforms_vec[offset + 1], sina*s[0] + cosa*s[1] + tforms_vec[offset + 2]);

                        eq_ind = 2*i;
                        nz_ind = 6*i;

                        values[nz_ind] = (-sina*s[0] - cosa*s[1])*weight;
                        inner_ind[nz_ind++] = offset;
                        values[nz_ind] = weight;
                        inner_ind[nz_ind++] = offset + 1;
                        values[nz_ind] = (ScalarT)0.0;
                        inner_ind[nz_ind++] = offset + 2;
                        b[eq_ind++] = (d[0] - s_t[0])*weight;

                        values[nz_ind] = (cosa*s[0] - sina*s[1])*weight;
                        inner_ind[nz_ind++] = offset;
                        values[nz_ind] = (ScalarT)0.0;
                        inner_ind[nz_ind++] = offset + 1;
                        values[nz_ind] = weight;
                        inner_ind[nz_ind++] = offset + 2;
                        b[eq_ind++] = (d[1] - s_t[1])*weight;
                    }
                }

                if (has_point_to_plane_terms) {
#pragma omp for nowait
                    for (size_t i = 0; i < point_to_plane_correspondences.size(); i++) {
                        const auto& corr = point_to_plane_correspondences[i];
                        const auto d = dst_p.col(corr.indexInFirst);
                        const auto n = dst_n.col(corr.indexInFirst);
                        const auto s = src_p.col(corr.indexInSecond);
                        const auto offset = 3*corr.indexInSecond;
                        weight = point_to_plane_weight_sqrt*std::sqrt(plane_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value));

                        const ScalarT cosa = std::cos(tforms_vec[offset]);
                        const ScalarT sina = std::sin(tforms_vec[offset]);

                        Eigen::Matrix<ScalarT,2,1> s_t(cosa*s[0] - sina*s[1] + tforms_vec[offset + 1], sina*s[0] + cosa*s[1] + tforms_vec[offset + 2]);

                        eq_ind = num_point_to_point_equations + i;
                        nz_ind = 3*num_point_to_point_equations + 3*i;

                        values[nz_ind] = (n[0]*(-sina*s[0] - cosa*s[1]) + n[1]*(cosa*s[0] - sina*s[1]))*weight;
                        inner_ind[nz_ind++] = offset;
                        values[nz_ind] = n[0]*weight;
                        inner_ind[nz_ind++] = offset + 1;
                        values[nz_ind] = n[1]*weight;
                        inner_ind[nz_ind++] = offset + 2;

                        b[eq_ind] = n.dot(d - s_t)*weight;
                    }
                }

                // Regularization
#pragma omp for nowait
                for (size_t i = 0; i < regularization_neighborhoods.size(); i++) {
                    eq_ind = num_data_term_equations + reg_eq_ind[i];
                    nz_ind = 3*num_data_term_equations + 2*reg_eq_ind[i];
                    const auto& neighbors = regularization_neighborhoods[i];

                    for (size_t j = 1; j < neighbors.size(); j++) {
                        auto s_offset = 3*neighbors[0].index;
                        auto n_offset = 3*neighbors[j].index;
                        weight = regularization_weight_sqrt*std::sqrt(reg_evaluator(neighbors[0].index, neighbors[j].index, neighbors[j].value));

                        if (n_offset < s_offset) std::swap(s_offset, n_offset);

                        diff = tforms_vec[s_offset + 0] - tforms_vec[n_offset + 0];
                        d_sqrt_huber_loss = weight*internal::sqrtHuberLossDerivative<ScalarT>(diff, huber_boundary);
                        values[nz_ind] = d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = s_offset + 0;
                        values[nz_ind] = -d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = n_offset + 0;
                        b[eq_ind++] = -weight*internal::sqrtHuberLoss<ScalarT>(diff, huber_boundary);

                        diff = tforms_vec[s_offset + 1] - tforms_vec[n_offset + 1];
                        d_sqrt_huber_loss = weight*internal::sqrtHuberLossDerivative<ScalarT>(diff, huber_boundary);
                        values[nz_ind] = d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = s_offset + 1;
                        values[nz_ind] = -d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = n_offset + 1;
                        b[eq_ind++] = -weight*internal::sqrtHuberLoss<ScalarT>(diff, huber_boundary);

                        diff = tforms_vec[s_offset + 2] - tforms_vec[n_offset + 2];
                        d_sqrt_huber_loss = weight*internal::sqrtHuberLossDerivative<ScalarT>(diff, huber_boundary);
                        values[nz_ind] = d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = s_offset + 2;
                        values[nz_ind] = -d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = n_offset + 2;
                        b[eq_ind++] = -weight*internal::sqrtHuberLoss<ScalarT>(diff, huber_boundary);
                    }
                }
            }

            // Solve linear system using CG
            AtA = At*At.transpose();
            Atb.noalias() = At*b;

            if (iter == 0) solver.analyzePattern(AtA);
            solver.factorize(AtA);
            delta = solver.solve(Atb);
            tforms_vec += delta;

            iter++;

            // Check for convergence
            max_delta_sq = (ScalarT)0.0;
#pragma omp parallel for private (curr_delta_sq) reduction (max: max_delta_sq)
            for (size_t i = 0; i < src_p.cols(); i++) {
                curr_delta_sq = delta.template segment<3>(3*i).squaredNorm();
                if (curr_delta_sq > max_delta_sq) max_delta_sq = curr_delta_sq;
            }

            if (max_delta_sq < gn_conv_tol_sq) {
                has_converged = true;
                break;
            }
        }

        // Convert to output format
        transforms.resize(src_p.cols());
#pragma omp parallel for
        for (size_t i = 0; i < transforms.size(); i++) {
            transforms[i].linear().noalias() = Eigen::Rotation2D<ScalarT>(tforms_vec[3*i]).toRotationMatrix();
            transforms[i].translation() = tforms_vec.template segment<2>(3*i + 1);
        }

        return has_converged;
    }

    // Locally rigid dense warp field, 3D
    template <class TransformT, class PointCorrSetT, class PlaneCorrSetT, class RegNeighborhoodSetT, class PointCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class PlaneCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class RegWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>>
    typename std::enable_if<int(TransformT::Mode) == int(Eigen::Isometry) && TransformT::Dim == 3,bool>::type
    estimateDenseWarpFieldCombinedMetric(const ConstVectorSetMatrixMap<typename TransformT::Scalar,3> &dst_p,
                                         const ConstVectorSetMatrixMap<typename TransformT::Scalar,3> &dst_n,
                                         const ConstVectorSetMatrixMap<typename TransformT::Scalar,3> &src_p,
                                         const PointCorrSetT &point_to_point_correspondences,
                                         typename TransformT::Scalar point_to_point_weight,
                                         const PlaneCorrSetT &point_to_plane_correspondences,
                                         typename TransformT::Scalar point_to_plane_weight,
                                         const RegNeighborhoodSetT &regularization_neighborhoods,
                                         typename TransformT::Scalar regularization_weight,
                                         TransformSet<TransformT> &transforms,
                                         typename TransformT::Scalar huber_boundary = (typename TransformT::Scalar)(1e-4),
                                         size_t max_gn_iter = 10,
                                         typename TransformT::Scalar gn_conv_tol = (typename TransformT::Scalar)1e-5,
                                         size_t max_cg_iter = 1000,
                                         typename TransformT::Scalar cg_conv_tol = (typename TransformT::Scalar)1e-5,
                                         const PointCorrWeightEvaluatorT &point_corr_evaluator = PointCorrWeightEvaluatorT(),
                                         const PlaneCorrWeightEvaluatorT &plane_corr_evaluator = PlaneCorrWeightEvaluatorT(),
                                         const RegWeightEvaluatorT &reg_evaluator = RegWeightEvaluatorT())
    {
        typedef typename TransformT::Scalar ScalarT;

        const bool has_point_to_point_terms = !point_to_point_correspondences.empty() && (point_to_point_weight > (ScalarT)0.0);
        const bool has_point_to_plane_terms = !point_to_plane_correspondences.empty() && (point_to_plane_weight > (ScalarT)0.0);

        if ((!has_point_to_point_terms && !has_point_to_plane_terms) ||
            (has_point_to_plane_terms && dst_p.cols() != dst_n.cols()))
        {
            transforms.resize(src_p.cols());
            transforms.setIdentity();
            return false;
        }

        // Get regularization equation count and indices
        std::vector<size_t> reg_eq_ind(regularization_neighborhoods.size());
        size_t num_reg_arcs = 0;
        if (!regularization_neighborhoods.empty()) {
            reg_eq_ind[0] = 0;
            num_reg_arcs = std::max((size_t)0, regularization_neighborhoods[0].size() - 1);
        }
        for (size_t i = 1; i < regularization_neighborhoods.size(); i++) {
            reg_eq_ind[i] = reg_eq_ind[i-1] + 6*std::max((size_t)0, regularization_neighborhoods[i-1].size() - 1);
            num_reg_arcs += std::max((size_t)0, regularization_neighborhoods[i].size() - 1);
        }

        // Compute number of equations and unknowns
        const size_t num_unknowns = 6*src_p.cols();
        const size_t num_point_to_point_equations = 3*has_point_to_point_terms*point_to_point_correspondences.size();
        const size_t num_point_to_plane_equations = has_point_to_plane_terms*point_to_plane_correspondences.size();
        const size_t num_data_term_equations = num_point_to_point_equations + num_point_to_plane_equations;
        const size_t num_regularization_equations = 6*num_reg_arcs;
        const size_t num_equations = num_data_term_equations + num_regularization_equations;
        const size_t num_non_zeros = 6*num_data_term_equations + 2*num_regularization_equations;

        // Jacobian
        Eigen::SparseMatrix<ScalarT> At(num_unknowns, num_equations);
        At.reserve(num_non_zeros);
        // Values
        ScalarT * const values = At.valuePtr();
        // Outer pointers
        typename Eigen::SparseMatrix<ScalarT>::StorageIndex * const outer_ptr = At.outerIndexPtr();
#pragma omp parallel
        {
#pragma omp for nowait
            for (size_t i = 0; i < num_data_term_equations + 1; i++) {
                outer_ptr[i] = 6*i;
            }
#pragma omp for nowait
            for (size_t i = 1; i < num_regularization_equations + 1; i++) {
                outer_ptr[num_data_term_equations + i] = 6*num_data_term_equations + 2*i;
            }
        }
        // Inner indices
        typename Eigen::SparseMatrix<ScalarT>::StorageIndex * const inner_ind = At.innerIndexPtr();

        // Vector of (negative) residuals
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> b(num_equations);

        // Vector of unknowns (Euler angles and translation offsets per point)
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> tforms_vec(Eigen::Matrix<ScalarT,Eigen::Dynamic,1>::Zero(num_unknowns, 1));

        // Conjugate Gradient solver
//        Eigen::ConjugateGradient<Eigen::SparseMatrix<ScalarT>,Eigen::Lower|Eigen::Upper,Eigen::IncompleteCholesky<ScalarT,Eigen::Lower|Eigen::Upper>> solver;
//        Eigen::ConjugateGradient<Eigen::SparseMatrix<ScalarT>,Eigen::Lower|Eigen::Upper,Eigen::IdentityPreconditioner> solver;
        Eigen::ConjugateGradient<Eigen::SparseMatrix<ScalarT>,Eigen::Lower|Eigen::Upper,Eigen::DiagonalPreconditioner<ScalarT>> solver;
//        Eigen::ConjugateGradient<Eigen::SparseMatrix<ScalarT>,Eigen::Lower|Eigen::Upper,BlockDiagonalPreconditioner<ScalarT,6>> solver;
        solver.setMaxIterations(max_cg_iter);
        solver.setTolerance(cg_conv_tol);

        Eigen::SparseMatrix<ScalarT> AtA;
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> Atb;

        // Parameters
        const ScalarT point_to_point_weight_sqrt = std::sqrt(point_to_point_weight);
        const ScalarT point_to_plane_weight_sqrt = std::sqrt(point_to_plane_weight);
        const ScalarT regularization_weight_sqrt = std::sqrt(regularization_weight);
        const ScalarT gn_conv_tol_sq = gn_conv_tol*gn_conv_tol;

        // Temporaries
        Eigen::Matrix<ScalarT,3,3> rot_coeffs, d_rot_coeffs_da, d_rot_coeffs_db, d_rot_coeffs_dc;
        Eigen::Matrix<ScalarT,3,1> trans_s, d_rot_da_s, d_rot_db_s, d_rot_dc_s;
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> delta;
        ScalarT weight, diff, d_sqrt_huber_loss, curr_delta_sq, max_delta_sq;
        size_t eq_ind, nz_ind;

        bool has_converged = false;
        size_t iter = 0;
        while (iter < max_gn_iter) {
#pragma omp parallel shared (At, b) private (eq_ind, nz_ind, weight, diff, d_sqrt_huber_loss, rot_coeffs, d_rot_coeffs_da, d_rot_coeffs_db, d_rot_coeffs_dc, trans_s, d_rot_da_s, d_rot_db_s, d_rot_dc_s)
            {
                // Data term
                if (has_point_to_point_terms) {
#pragma omp for nowait
                    for (size_t i = 0; i < point_to_point_correspondences.size(); i++) {
                        const auto& corr = point_to_point_correspondences[i];
                        const auto d = dst_p.col(corr.indexInFirst);
                        const auto s = src_p.col(corr.indexInSecond);
                        const auto offset = 6*corr.indexInSecond;
                        weight = point_to_point_weight_sqrt*std::sqrt(point_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value));

                        internal::computeRotationTerms(tforms_vec[offset], tforms_vec[offset + 1], tforms_vec[offset + 2], rot_coeffs, d_rot_coeffs_da, d_rot_coeffs_db, d_rot_coeffs_dc);
                        const auto trans_coeffs = tforms_vec.template segment<3>(offset + 3);

                        trans_s.noalias() = d - (rot_coeffs.transpose()*s + trans_coeffs);
                        d_rot_da_s.noalias() = d_rot_coeffs_da.transpose()*s;
                        d_rot_db_s.noalias() = d_rot_coeffs_db.transpose()*s;
                        d_rot_dc_s.noalias() = d_rot_coeffs_dc.transpose()*s;

                        eq_ind = 3*i;
                        nz_ind = 18*i;

                        values[nz_ind] = d_rot_da_s[0]*weight;
                        inner_ind[nz_ind++] = offset;
                        values[nz_ind] = d_rot_db_s[0]*weight;
                        inner_ind[nz_ind++] = offset + 1;
                        values[nz_ind] = d_rot_dc_s[0]*weight;
                        inner_ind[nz_ind++] = offset + 2;
                        values[nz_ind] = weight;
                        inner_ind[nz_ind++] = offset + 3;
                        values[nz_ind] = (ScalarT)0.0;
                        inner_ind[nz_ind++] = offset + 4;
                        values[nz_ind] = (ScalarT)0.0;
                        inner_ind[nz_ind++] = offset + 5;
                        b[eq_ind++] = trans_s[0]*weight;

                        values[nz_ind] = d_rot_da_s[1]*weight;
                        inner_ind[nz_ind++] = offset;
                        values[nz_ind] = d_rot_db_s[1]*weight;
                        inner_ind[nz_ind++] = offset + 1;
                        values[nz_ind] = d_rot_dc_s[1]*weight;
                        inner_ind[nz_ind++] = offset + 2;
                        values[nz_ind] = (ScalarT)0.0;
                        inner_ind[nz_ind++] = offset + 3;
                        values[nz_ind] = weight;
                        inner_ind[nz_ind++] = offset + 4;
                        values[nz_ind] = (ScalarT)0.0;
                        inner_ind[nz_ind++] = offset + 5;
                        b[eq_ind++] = trans_s[1]*weight;

                        values[nz_ind] = d_rot_da_s[2]*weight;
                        inner_ind[nz_ind++] = offset;
                        values[nz_ind] = d_rot_db_s[2]*weight;
                        inner_ind[nz_ind++] = offset + 1;
                        values[nz_ind] = d_rot_dc_s[2]*weight;
                        inner_ind[nz_ind++] = offset + 2;
                        values[nz_ind] = (ScalarT)0.0;
                        inner_ind[nz_ind++] = offset + 3;
                        values[nz_ind] = (ScalarT)0.0;
                        inner_ind[nz_ind++] = offset + 4;
                        values[nz_ind] = weight;
                        inner_ind[nz_ind++] = offset + 5;
                        b[eq_ind++] = trans_s[2]*weight;
                    }
                }

                if (has_point_to_plane_terms) {
#pragma omp for nowait
                    for (size_t i = 0; i < point_to_plane_correspondences.size(); i++) {
                        const auto& corr = point_to_plane_correspondences[i];
                        const auto d = dst_p.col(corr.indexInFirst);
                        const auto n = dst_n.col(corr.indexInFirst);
                        const auto s = src_p.col(corr.indexInSecond);
                        const auto offset = 6*corr.indexInSecond;
                        weight = point_to_plane_weight_sqrt*std::sqrt(plane_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value));

                        internal::computeRotationTerms(tforms_vec[offset], tforms_vec[offset + 1], tforms_vec[offset + 2], rot_coeffs, d_rot_coeffs_da, d_rot_coeffs_db, d_rot_coeffs_dc);
                        const auto trans_coeffs = tforms_vec.template segment<3>(offset + 3);

                        trans_s.noalias() = d - (rot_coeffs.transpose()*s + trans_coeffs);
                        d_rot_da_s.noalias() = d_rot_coeffs_da.transpose()*s;
                        d_rot_db_s.noalias() = d_rot_coeffs_db.transpose()*s;
                        d_rot_dc_s.noalias() = d_rot_coeffs_dc.transpose()*s;

                        eq_ind = num_point_to_point_equations + i;
                        nz_ind = 6*num_point_to_point_equations + 6*i;

                        values[nz_ind] = (n.dot(d_rot_da_s))*weight;
                        inner_ind[nz_ind++] = offset;
                        values[nz_ind] = (n.dot(d_rot_db_s))*weight;
                        inner_ind[nz_ind++] = offset + 1;
                        values[nz_ind] = (n.dot(d_rot_dc_s))*weight;
                        inner_ind[nz_ind++] = offset + 2;
                        values[nz_ind] = n[0]*weight;
                        inner_ind[nz_ind++] = offset + 3;
                        values[nz_ind] = n[1]*weight;
                        inner_ind[nz_ind++] = offset + 4;
                        values[nz_ind] = n[2]*weight;
                        inner_ind[nz_ind++] = offset + 5;
                        b[eq_ind] = n.dot(trans_s)*weight;
                    }
                }

                // Regularization
#pragma omp for nowait
                for (size_t i = 0; i < regularization_neighborhoods.size(); i++) {
                    eq_ind = num_data_term_equations + reg_eq_ind[i];
                    nz_ind = 6*num_data_term_equations + 2*reg_eq_ind[i];
                    const auto& neighbors = regularization_neighborhoods[i];

                    for (size_t j = 1; j < neighbors.size(); j++) {
                        auto s_offset = 6*neighbors[0].index;
                        auto n_offset = 6*neighbors[j].index;
                        weight = regularization_weight_sqrt*std::sqrt(reg_evaluator(neighbors[0].index, neighbors[j].index, neighbors[j].value));

                        if (n_offset < s_offset) std::swap(s_offset, n_offset);

                        diff = tforms_vec[s_offset + 0] - tforms_vec[n_offset + 0];
                        d_sqrt_huber_loss = weight*internal::sqrtHuberLossDerivative<ScalarT>(diff, huber_boundary);
                        values[nz_ind] = d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = s_offset;
                        values[nz_ind] = -d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = n_offset;
                        b[eq_ind++] = -weight*internal::sqrtHuberLoss<ScalarT>(diff, huber_boundary);

                        diff = tforms_vec[s_offset + 1] - tforms_vec[n_offset + 1];
                        d_sqrt_huber_loss = weight*internal::sqrtHuberLossDerivative<ScalarT>(diff, huber_boundary);
                        values[nz_ind] = d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = s_offset + 1;
                        values[nz_ind] = -d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = n_offset + 1;
                        b[eq_ind++] = -weight*internal::sqrtHuberLoss<ScalarT>(diff, huber_boundary);

                        diff = tforms_vec[s_offset + 2] - tforms_vec[n_offset + 2];
                        d_sqrt_huber_loss = weight*internal::sqrtHuberLossDerivative<ScalarT>(diff, huber_boundary);
                        values[nz_ind] = d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = s_offset + 2;
                        values[nz_ind] = -d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = n_offset + 2;
                        b[eq_ind++] = -weight*internal::sqrtHuberLoss<ScalarT>(diff, huber_boundary);

                        diff = tforms_vec[s_offset + 3] - tforms_vec[n_offset + 3];
                        d_sqrt_huber_loss = weight*internal::sqrtHuberLossDerivative<ScalarT>(diff, huber_boundary);
                        values[nz_ind] = d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = s_offset + 3;
                        values[nz_ind] = -d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = n_offset + 3;
                        b[eq_ind++] = -weight*internal::sqrtHuberLoss<ScalarT>(diff, huber_boundary);

                        diff = tforms_vec[s_offset + 4] - tforms_vec[n_offset + 4];
                        d_sqrt_huber_loss = weight*internal::sqrtHuberLossDerivative<ScalarT>(diff, huber_boundary);
                        values[nz_ind] = d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = s_offset + 4;
                        values[nz_ind] = -d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = n_offset + 4;
                        b[eq_ind++] = -weight*internal::sqrtHuberLoss<ScalarT>(diff, huber_boundary);

                        diff = tforms_vec[s_offset + 5] - tforms_vec[n_offset + 5];
                        d_sqrt_huber_loss = weight*internal::sqrtHuberLossDerivative<ScalarT>(diff, huber_boundary);
                        values[nz_ind] = d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = s_offset + 5;
                        values[nz_ind] = -d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = n_offset + 5;
                        b[eq_ind++] = -weight*internal::sqrtHuberLoss<ScalarT>(diff, huber_boundary);
                    }
                }
            }

            // Solve linear system using CG
            AtA = At*At.transpose();
            Atb.noalias() = At*b;

            if (iter == 0) solver.analyzePattern(AtA);
            solver.factorize(AtA);
            delta = solver.solve(Atb);
            tforms_vec += delta;

            iter++;

            // Check for convergence
            max_delta_sq = (ScalarT)0.0;
#pragma omp parallel for private (curr_delta_sq) reduction (max: max_delta_sq)
            for (size_t i = 0; i < src_p.cols(); i++) {
                curr_delta_sq = delta.template segment<6>(6*i).squaredNorm();
                if (curr_delta_sq > max_delta_sq) max_delta_sq = curr_delta_sq;
            }

            if (max_delta_sq < gn_conv_tol_sq) {
                has_converged = true;
                break;
            }
        }

        // Convert to output format
        transforms.resize(src_p.cols());
#pragma omp parallel for
        for (size_t i = 0; i < transforms.size(); i++) {
            transforms[i].linear().noalias() = (Eigen::AngleAxis<ScalarT>(tforms_vec[6*i + 2],Eigen::Matrix<ScalarT,3,1>::UnitZ()) *
                                                Eigen::AngleAxis<ScalarT>(tforms_vec[6*i + 1],Eigen::Matrix<ScalarT,3,1>::UnitY()) *
                                                Eigen::AngleAxis<ScalarT>(tforms_vec[6*i + 0],Eigen::Matrix<ScalarT,3,1>::UnitX())).matrix();
            transforms[i].linear() = transforms[i].rotation();
            transforms[i].translation() = tforms_vec.template segment<3>(6*i + 3);
        }

        return has_converged;
    }

    // Locally affine dense warp field, general dimension
    template <class TransformT, class PointCorrSetT, class PlaneCorrSetT, class RegNeighborhoodSetT, class PointCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class PlaneCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class RegWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>>
    typename std::enable_if<int(TransformT::Mode) == int(Eigen::Affine) || int(TransformT::Mode) == int(Eigen::AffineCompact),bool>::type
    estimateDenseWarpFieldCombinedMetric(const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_p,
                                         const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_n,
                                         const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src_p,
                                         const PointCorrSetT &point_to_point_correspondences,
                                         typename TransformT::Scalar point_to_point_weight,
                                         const PlaneCorrSetT &point_to_plane_correspondences,
                                         typename TransformT::Scalar point_to_plane_weight,
                                         const RegNeighborhoodSetT &regularization_neighborhoods,
                                         typename TransformT::Scalar regularization_weight,
                                         TransformSet<TransformT> &transforms,
                                         typename TransformT::Scalar huber_boundary = (typename TransformT::Scalar)(1e-4),
                                         size_t max_gn_iter = 10,
                                         typename TransformT::Scalar gn_conv_tol = (typename TransformT::Scalar)1e-5,
                                         size_t max_cg_iter = 1000,
                                         typename TransformT::Scalar cg_conv_tol = (typename TransformT::Scalar)1e-5,
                                         const PointCorrWeightEvaluatorT &point_corr_evaluator = PointCorrWeightEvaluatorT(),
                                         const PlaneCorrWeightEvaluatorT &plane_corr_evaluator = PlaneCorrWeightEvaluatorT(),
                                         const RegWeightEvaluatorT &reg_evaluator = RegWeightEvaluatorT())
    {
        typedef typename TransformT::Scalar ScalarT;
        enum {
            Dim = TransformT::Dim,
            NumUnknownsLocal = TransformT::Dim*(TransformT::Dim + 1),
            NumNonZerosPointToPoint = TransformT::Dim + 1,
            NumNonZerosPointToPlane = TransformT::Dim*(TransformT::Dim + 1)
        };

        const bool has_point_to_point_terms = !point_to_point_correspondences.empty() && (point_to_point_weight > (ScalarT)0.0);
        const bool has_point_to_plane_terms = !point_to_plane_correspondences.empty() && (point_to_plane_weight > (ScalarT)0.0);

        if ((!has_point_to_point_terms && !has_point_to_plane_terms) ||
            (has_point_to_plane_terms && dst_p.cols() != dst_n.cols()))
        {
            transforms.resize(src_p.cols());
            transforms.setIdentity();
            return false;
        }

        // Get regularization equation count and indices
        std::vector<size_t> reg_eq_ind(regularization_neighborhoods.size());
        size_t num_reg_arcs = 0;
        if (!regularization_neighborhoods.empty()) {
            reg_eq_ind[0] = 0;
            num_reg_arcs = std::max((size_t)0, regularization_neighborhoods[0].size() - 1);
        }
        for (size_t i = 1; i < regularization_neighborhoods.size(); i++) {
            reg_eq_ind[i] = reg_eq_ind[i-1] + NumUnknownsLocal*std::max((size_t)0, regularization_neighborhoods[i-1].size() - 1);
            num_reg_arcs += std::max((size_t)0, regularization_neighborhoods[i].size() - 1);
        }

        // Compute number of equations and unknowns
        const size_t num_unknowns = NumUnknownsLocal*src_p.cols();
        const size_t num_point_to_point_equations = Dim*has_point_to_point_terms*point_to_point_correspondences.size();
        const size_t num_point_to_plane_equations = has_point_to_plane_terms*point_to_plane_correspondences.size();
        const size_t num_data_term_equations = num_point_to_point_equations + num_point_to_plane_equations;
        const size_t num_regularization_equations = NumUnknownsLocal*num_reg_arcs;
        const size_t num_equations = num_data_term_equations + num_regularization_equations;
        const size_t num_non_zeros_data_term = NumNonZerosPointToPoint*num_point_to_point_equations + NumNonZerosPointToPlane*num_point_to_plane_equations;
        const size_t num_non_zeros = num_non_zeros_data_term + 2*num_regularization_equations;

        // Jacobian
        Eigen::SparseMatrix<ScalarT> At(num_unknowns, num_equations);
        At.reserve(num_non_zeros);
        // Values
        ScalarT * const values = At.valuePtr();
        // Outer pointers
        typename Eigen::SparseMatrix<ScalarT>::StorageIndex * const outer_ptr = At.outerIndexPtr();
#pragma omp parallel
        {
#pragma omp for nowait
            for (size_t i = 0; i < num_point_to_point_equations + 1; i++) {
                outer_ptr[i] = NumNonZerosPointToPoint*i;
            }
#pragma omp for nowait
            for (size_t i = 1; i < num_point_to_plane_equations + 1; i++) {
                outer_ptr[num_point_to_point_equations + i] = NumNonZerosPointToPoint*num_point_to_point_equations + NumNonZerosPointToPlane*i;
            }
#pragma omp for nowait
            for (size_t i = 1; i < num_regularization_equations + 1; i++) {
                outer_ptr[num_data_term_equations + i] = num_non_zeros_data_term + 2*i;
            }
        }
        // Inner indices
        typename Eigen::SparseMatrix<ScalarT>::StorageIndex * const inner_ind = At.innerIndexPtr();

        // Vector of (negative) residuals
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> b(num_equations);

        // Vector of unknowns
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> tforms_vec(num_unknowns, 1);
#pragma omp parallel for
        for (size_t i = 0; i < src_p.cols(); i++) {
            Eigen::Map<Eigen::Matrix<ScalarT,Dim,Dim,Eigen::RowMajor>>(tforms_vec.data() + i*NumUnknownsLocal, Dim, Dim).setIdentity();
            tforms_vec.template segment<Dim>(i*NumUnknownsLocal + Dim*Dim).setZero();
        }

        // Conjugate Gradient solver
//        Eigen::ConjugateGradient<Eigen::SparseMatrix<ScalarT>,Eigen::Lower|Eigen::Upper,Eigen::IncompleteCholesky<ScalarT,Eigen::Lower|Eigen::Upper>> solver;
//        Eigen::ConjugateGradient<Eigen::SparseMatrix<ScalarT>,Eigen::Lower|Eigen::Upper,Eigen::IdentityPreconditioner> solver;
        Eigen::ConjugateGradient<Eigen::SparseMatrix<ScalarT>,Eigen::Lower|Eigen::Upper,Eigen::DiagonalPreconditioner<ScalarT>> solver;
        solver.setMaxIterations(max_cg_iter);
        solver.setTolerance(cg_conv_tol);

        Eigen::SparseMatrix<ScalarT> AtA;
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> Atb;

        // Parameters
        const ScalarT point_to_point_weight_sqrt = std::sqrt(point_to_point_weight);
        const ScalarT point_to_plane_weight_sqrt = std::sqrt(point_to_plane_weight);
        const ScalarT regularization_weight_sqrt = std::sqrt(regularization_weight);
        const ScalarT gn_conv_tol_sq = gn_conv_tol*gn_conv_tol;

        // Temporaries
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> delta;
        ScalarT weight, diff, d_sqrt_huber_loss, curr_delta_sq, max_delta_sq;
        size_t eq_ind, nz_ind;

        bool has_converged = false;
        size_t iter = 0;
        while (iter < max_gn_iter) {
#pragma omp parallel shared (At, b) private (eq_ind, nz_ind, weight, diff, d_sqrt_huber_loss)
            {
                // Data term
                if (has_point_to_point_terms) {
#pragma omp for nowait
                    for (size_t i = 0; i < point_to_point_correspondences.size(); i++) {
                        const auto& corr = point_to_point_correspondences[i];
                        const auto d = dst_p.col(corr.indexInFirst);
                        const auto s = src_p.col(corr.indexInSecond);
                        const auto offset = NumUnknownsLocal*corr.indexInSecond;
                        weight = point_to_point_weight_sqrt*std::sqrt(point_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value));

                        auto linear = Eigen::Map<Eigen::Matrix<ScalarT,Dim,Dim,Eigen::RowMajor>>(tforms_vec.data() + offset, Dim, Dim);
                        auto translation = tforms_vec.template segment<Dim>(offset + Dim*Dim);
                        Eigen::Matrix<ScalarT,Dim,1> s_t = linear*s + translation;

                        eq_ind = Dim*i;
                        nz_ind = Dim*NumNonZerosPointToPoint*i;

                        for (size_t eq = 0; eq < Dim; eq++) {
                            for (size_t nz = 0; nz < Dim; nz++) {
                                values[nz_ind] = weight*s_t[nz];
                                inner_ind[nz_ind++] = offset + Dim*eq + nz;
                            }
                            values[nz_ind] = weight;
                            inner_ind[nz_ind++] = offset + Dim*Dim + eq;
                        }
                        b.template segment<Dim>(eq_ind) = weight*(d - s_t);
                    }
                }

                if (has_point_to_plane_terms) {
#pragma omp for nowait
                    for (size_t i = 0; i < point_to_plane_correspondences.size(); i++) {
                        const auto& corr = point_to_plane_correspondences[i];
                        const auto d = dst_p.col(corr.indexInFirst);
                        const auto n = dst_n.col(corr.indexInFirst);
                        const auto s = src_p.col(corr.indexInSecond);
                        const auto offset = NumUnknownsLocal*corr.indexInSecond;
                        weight = point_to_plane_weight_sqrt*std::sqrt(plane_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value));

                        auto linear = Eigen::Map<Eigen::Matrix<ScalarT,Dim,Dim,Eigen::RowMajor>>(tforms_vec.data() + offset, Dim, Dim);
                        auto translation = tforms_vec.template segment<Dim>(offset + Dim*Dim);
                        Eigen::Matrix<ScalarT,Dim,1> s_t = linear*s + translation;

                        eq_ind = num_point_to_point_equations + i;
                        nz_ind = NumNonZerosPointToPoint*num_point_to_point_equations + NumNonZerosPointToPlane*i;

                        for (size_t block = 0; block < Dim; block++) {
                            for (size_t curr = 0; curr < Dim; curr++) {
                                values[nz_ind] = weight*n[block]*s_t[curr];
                                inner_ind[nz_ind++] = offset + Dim*block + curr;
                            }
                        }
                        for (size_t curr = 0; curr < Dim; curr++) {
                            values[nz_ind] = weight*n[curr];
                            inner_ind[nz_ind++] = offset + Dim*Dim + curr;
                        }

                        b[eq_ind] = (n.dot(d - s_t))*weight;
                    }
                }

                // Regularization
#pragma omp for nowait
                for (size_t i = 0; i < regularization_neighborhoods.size(); i++) {
                    eq_ind = num_data_term_equations + reg_eq_ind[i];
                    nz_ind = num_non_zeros_data_term + 2*reg_eq_ind[i];
                    const auto& neighbors = regularization_neighborhoods[i];

                    for (size_t j = 1; j < neighbors.size(); j++) {
                        auto s_offset = NumUnknownsLocal*neighbors[0].index;
                        auto n_offset = NumUnknownsLocal*neighbors[j].index;
                        weight = regularization_weight_sqrt*std::sqrt(reg_evaluator(neighbors[0].index, neighbors[j].index, neighbors[j].value));

                        if (n_offset < s_offset) std::swap(s_offset, n_offset);

                        for (size_t eq = 0; eq < NumUnknownsLocal; eq++) {
                            diff = tforms_vec[s_offset + eq] - tforms_vec[n_offset + eq];
                            d_sqrt_huber_loss = weight*internal::sqrtHuberLossDerivative<ScalarT>(diff, huber_boundary);
                            values[nz_ind] = d_sqrt_huber_loss;
                            inner_ind[nz_ind++] = s_offset + eq;
                            values[nz_ind] = -d_sqrt_huber_loss;
                            inner_ind[nz_ind++] = n_offset + eq;
                            b[eq_ind++] = -weight*internal::sqrtHuberLoss<ScalarT>(diff, huber_boundary);
                        }
                    }
                }
            }

            // Solve linear system using CG
            AtA = At*At.transpose();
            Atb.noalias() = At*b;

            if (iter == 0) solver.analyzePattern(AtA);
            solver.factorize(AtA);
            delta = solver.solve(Atb);
            tforms_vec += delta;

            iter++;

            // Check for convergence
            max_delta_sq = (ScalarT)0.0;
#pragma omp parallel for private (curr_delta_sq) reduction (max: max_delta_sq)
            for (size_t i = 0; i < src_p.cols(); i++) {
                curr_delta_sq = delta.template segment<NumUnknownsLocal>(NumUnknownsLocal*i).squaredNorm();
                if (curr_delta_sq > max_delta_sq) max_delta_sq = curr_delta_sq;
            }

            if (max_delta_sq < gn_conv_tol_sq) {
                has_converged = true;
                break;
            }
        }

        // Convert to output format
        transforms.resize(src_p.cols());
#pragma omp parallel for
        for (size_t i = 0; i < transforms.size(); i++) {
            transforms[i].linear().noalias() = Eigen::Map<Eigen::Matrix<ScalarT,Dim,Dim,Eigen::RowMajor>>(tforms_vec.data() + i*NumUnknownsLocal, Dim, Dim);
            transforms[i].translation() = tforms_vec.template segment<Dim>(NumUnknownsLocal*i + Dim*Dim);
        }

        return has_converged;
    }

    // Locally rigid sparse warp field, 2D
    template <class TransformT, class PointCorrSetT, class PlaneCorrSetT, class CtrlNeighborhoodSetT, class RegNeighborhoodSetT, class PointCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class PlaneCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class ControlWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class RegWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>>
    typename std::enable_if<int(TransformT::Mode) == int(Eigen::Isometry) && TransformT::Dim == 2,bool>::type
    estimateSparseWarpFieldCombinedMetric(const ConstVectorSetMatrixMap<typename TransformT::Scalar,2> &dst_p,
                                          const ConstVectorSetMatrixMap<typename TransformT::Scalar,2> &dst_n,
                                          const ConstVectorSetMatrixMap<typename TransformT::Scalar,2> &src_p,
                                          const PointCorrSetT &point_to_point_correspondences,
                                          typename TransformT::Scalar point_to_point_weight,
                                          const PlaneCorrSetT &point_to_plane_correspondences,
                                          typename TransformT::Scalar point_to_plane_weight,
                                          const CtrlNeighborhoodSetT &src_to_ctrl_neighborhoods,
                                          size_t num_ctrl_points,
                                          const RegNeighborhoodSetT &regularization_neighborhoods,
                                          typename TransformT::Scalar regularization_weight,
                                          TransformSet<TransformT> &transforms,
                                          typename TransformT::Scalar huber_boundary = (typename TransformT::Scalar)(1e-4),
                                          size_t max_gn_iter = 10,
                                          typename TransformT::Scalar gn_conv_tol = (typename TransformT::Scalar)1e-5,
                                          size_t max_cg_iter = 1000,
                                          typename TransformT::Scalar cg_conv_tol = (typename TransformT::Scalar)1e-5,
                                          const PointCorrWeightEvaluatorT &point_corr_evaluator = PointCorrWeightEvaluatorT(),
                                          const PlaneCorrWeightEvaluatorT &plane_corr_evaluator = PlaneCorrWeightEvaluatorT(),
                                          const ControlWeightEvaluatorT &control_evaluator = ControlWeightEvaluatorT(),
                                          const RegWeightEvaluatorT &reg_evaluator = RegWeightEvaluatorT())
    {
        typedef typename TransformT::Scalar ScalarT;

        const bool has_point_to_point_terms = !point_to_point_correspondences.empty() && (point_to_point_weight > (ScalarT)0.0);
        const bool has_point_to_plane_terms = !point_to_plane_correspondences.empty() && (point_to_plane_weight > (ScalarT)0.0);

        if (src_to_ctrl_neighborhoods.size() != src_p.cols() ||
            (!has_point_to_point_terms && !has_point_to_plane_terms) ||
            (has_point_to_plane_terms && dst_p.cols() != dst_n.cols()))
        {
            transforms.resize(num_ctrl_points);
            transforms.setIdentity();
            return false;
        }

        // Sort control nodes by index and compute total weight
        CtrlNeighborhoodSetT src_to_ctrl_sorted(src_to_ctrl_neighborhoods.size());
        std::vector<ScalarT> total_weight(src_to_ctrl_sorted.size());
        std::vector<char> has_data_term(src_to_ctrl_neighborhoods.size(), 0);
#pragma omp parallel shared (src_to_ctrl_sorted, total_weight, has_data_term)
        {
            if (has_point_to_point_terms) {
#pragma omp for nowait
                for (size_t i = 0; i < point_to_point_correspondences.size(); i++) {
                    has_data_term[point_to_point_correspondences[i].indexInSecond] = 1;
                }
            }

            if (has_point_to_plane_terms) {
#pragma omp for
                for (size_t i = 0; i < point_to_plane_correspondences.size(); i++) {
                    has_data_term[point_to_plane_correspondences[i].indexInSecond] = 1;
                }
            }

#pragma omp for schedule (dynamic)
            for (size_t i = 0; i < has_data_term.size(); i++) {
                if (has_data_term[i]) {
                    total_weight[i] = (ScalarT)0.0;
                    src_to_ctrl_sorted[i].resize(src_to_ctrl_neighborhoods[i].size());
                    for (size_t j = 0; j < src_to_ctrl_neighborhoods[i].size(); j++) {
                        src_to_ctrl_sorted[i][j].index = src_to_ctrl_neighborhoods[i][j].index;
                        src_to_ctrl_sorted[i][j].value = control_evaluator(i, src_to_ctrl_neighborhoods[i][j].index, src_to_ctrl_neighborhoods[i][j].value);
                        total_weight[i] += src_to_ctrl_sorted[i][j].value;
                    }
                    std::sort(src_to_ctrl_sorted[i].begin(), src_to_ctrl_sorted[i].end(), typename CtrlNeighborhoodSetT::value_type::value_type::IndexLessComparator());
                }
            }
        }

        // Get regularization equation count and indices
        std::vector<size_t> reg_eq_ind(regularization_neighborhoods.size());
        size_t num_reg_arcs = 0;
        if (!regularization_neighborhoods.empty()) {
            reg_eq_ind[0] = 0;
            num_reg_arcs = std::max((size_t)0, regularization_neighborhoods[0].size() - 1);
        }
        for (size_t i = 1; i < regularization_neighborhoods.size(); i++) {
            reg_eq_ind[i] = reg_eq_ind[i-1] + 3*std::max((size_t)0, regularization_neighborhoods[i-1].size() - 1);
            num_reg_arcs += std::max((size_t)0, regularization_neighborhoods[i].size() - 1);
        }

        // Compute number of equations and unknowns
        const size_t num_unknowns = 3*num_ctrl_points;
        const size_t num_point_to_point_equations = 2*has_point_to_point_terms*point_to_point_correspondences.size();
        const size_t num_point_to_plane_equations = has_point_to_plane_terms*point_to_plane_correspondences.size();
        const size_t num_data_term_equations = num_point_to_point_equations + num_point_to_plane_equations;
        const size_t num_regularization_equations = 3*num_reg_arcs;
        const size_t num_equations = num_data_term_equations + num_regularization_equations;

        // Jacobian
        Eigen::SparseMatrix<ScalarT> At(num_unknowns, num_equations);
        // Outer pointers
        typename Eigen::SparseMatrix<ScalarT>::StorageIndex * const outer_ptr = At.outerIndexPtr();
        outer_ptr[0] = 0;
        if (has_point_to_point_terms) {
            for (size_t i = 0; i < point_to_point_correspondences.size(); i++) {
                const size_t nnz_per_eq = 3*src_to_ctrl_sorted[point_to_point_correspondences[i].indexInSecond].size();
                outer_ptr[2*i + 1] = outer_ptr[2*i] + nnz_per_eq;
                outer_ptr[2*i + 2] = outer_ptr[2*i] + nnz_per_eq + nnz_per_eq;
            }
        }
        if (has_point_to_plane_terms) {
            for (size_t i = 0; i < point_to_plane_correspondences.size(); i++) {
                outer_ptr[num_point_to_point_equations + i + 1] = outer_ptr[num_point_to_point_equations + i] + 3*src_to_ctrl_sorted[point_to_plane_correspondences[i].indexInSecond].size();
            }
        }
#pragma omp parallel for
        for (size_t i = 1; i < num_regularization_equations + 1; i++) {
            outer_ptr[num_data_term_equations + i] = outer_ptr[num_data_term_equations] + 2*i;
        }
        At.reserve(outer_ptr[num_equations]);
        // Values
        ScalarT * const values = At.valuePtr();
        // Inner indices
        typename Eigen::SparseMatrix<ScalarT>::StorageIndex * const inner_ind = At.innerIndexPtr();

        // Vector of (negative) residuals
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> b(num_equations);

        // Vector of unknowns (rotation angle and translation offsets per control node)
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> tforms_vec(Eigen::Matrix<ScalarT,Eigen::Dynamic,1>::Zero(num_unknowns, 1));

        // Conjugate Gradient solver
//        Eigen::ConjugateGradient<Eigen::SparseMatrix<ScalarT>,Eigen::Lower|Eigen::Upper,Eigen::IncompleteCholesky<ScalarT,Eigen::Lower|Eigen::Upper>> solver;
//        Eigen::ConjugateGradient<Eigen::SparseMatrix<ScalarT>,Eigen::Lower|Eigen::Upper,Eigen::IdentityPreconditioner> solver;
        Eigen::ConjugateGradient<Eigen::SparseMatrix<ScalarT>,Eigen::Lower|Eigen::Upper,Eigen::DiagonalPreconditioner<ScalarT>> solver;
        solver.setMaxIterations(max_cg_iter);
        solver.setTolerance(cg_conv_tol);

        Eigen::SparseMatrix<ScalarT> AtA;
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> Atb;

        // Parameters
        const ScalarT point_to_point_weight_sqrt = std::sqrt(point_to_point_weight);
        const ScalarT point_to_plane_weight_sqrt = std::sqrt(point_to_plane_weight);
        const ScalarT regularization_weight_sqrt = std::sqrt(regularization_weight);
        const ScalarT gn_conv_tol_sq = gn_conv_tol*gn_conv_tol;

        // Temporaries
        ScalarT angle_curr;
        Eigen::Matrix<ScalarT,2,1> trans_curr;
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> delta;
        ScalarT weight, corr_weight_sqrt, corr_weight_nrm, diff, d_sqrt_huber_loss, curr_delta_sq, max_delta_sq;
        size_t eq_ind, nz_ind;

        bool has_converged = false;
        size_t iter = 0;
        while (iter < max_gn_iter) {
#pragma omp parallel shared (At, b) private (eq_ind, nz_ind, weight, corr_weight_sqrt, corr_weight_nrm, diff, d_sqrt_huber_loss, angle_curr, trans_curr)
            {
                // Data term
                if (has_point_to_point_terms) {
#pragma omp for nowait
                    for (size_t i = 0; i < point_to_point_correspondences.size(); i++) {
                        const auto& corr = point_to_point_correspondences[i];
                        const auto& ctrl_neighbors = src_to_ctrl_sorted[corr.indexInSecond];

                        // Compute weighted influence from control nodes
                        angle_curr = (ScalarT)0.0;
                        trans_curr.setZero();
                        for (size_t j = 0; j < ctrl_neighbors.size(); j++) {
                            const auto offset = 3*ctrl_neighbors[j].index;
                            angle_curr += ctrl_neighbors[j].value*tforms_vec[offset];
                            trans_curr.noalias() += ctrl_neighbors[j].value*tforms_vec.template segment<2>(offset + 1);
                        }
                        if (total_weight[corr.indexInSecond] != (ScalarT)0.0) {
                            weight = (ScalarT)(1.0)/total_weight[corr.indexInSecond];
                            angle_curr *= weight;
                            trans_curr *= weight;

                            corr_weight_sqrt = point_to_point_weight_sqrt*std::sqrt(point_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value));
                            corr_weight_nrm = corr_weight_sqrt/total_weight[corr.indexInSecond];
                        } else {
                            corr_weight_sqrt = (ScalarT)0.0;
                            corr_weight_nrm = (ScalarT)0.0;
                        }

                        const auto d = dst_p.col(corr.indexInFirst);
                        const auto s = src_p.col(corr.indexInSecond);

                        const ScalarT cosa = std::cos(angle_curr);
                        const ScalarT sina = std::sin(angle_curr);

                        Eigen::Matrix<ScalarT,2,1> s_t(cosa*s[0] - sina*s[1] + trans_curr[0], sina*s[0] + cosa*s[1] + trans_curr[1]);

                        eq_ind = 2*i;

                        const ScalarT coeff1 = -sina*s[0] - cosa*s[1];
                        const ScalarT coeff2 = cosa*s[0] - sina*s[1];

                        for (size_t j = 0; j < ctrl_neighbors.size(); j++) {
                            const auto offset = 3*ctrl_neighbors[j].index;
                            weight = corr_weight_nrm*ctrl_neighbors[j].value;

                            nz_ind = outer_ptr[eq_ind] + 3*j;
                            values[nz_ind] = coeff1*weight;
                            inner_ind[nz_ind++] = offset;
                            values[nz_ind] = weight;
                            inner_ind[nz_ind++] = offset + 1;
                            values[nz_ind] = (ScalarT)0.0;
                            inner_ind[nz_ind++] = offset + 2;

                            nz_ind = outer_ptr[eq_ind + 1] + 3*j;
                            values[nz_ind] = coeff2*weight;
                            inner_ind[nz_ind++] = offset;
                            values[nz_ind] = (ScalarT)0.0;
                            inner_ind[nz_ind++] = offset + 1;
                            values[nz_ind] = weight;
                            inner_ind[nz_ind++] = offset + 2;
                        }

                        b.template segment<2>(eq_ind) = (d - s_t)*corr_weight_sqrt;
                    }
                }

                if (has_point_to_plane_terms) {
#pragma omp for nowait
                    for (size_t i = 0; i < point_to_plane_correspondences.size(); i++) {
                        const auto& corr = point_to_plane_correspondences[i];
                        const auto& ctrl_neighbors = src_to_ctrl_sorted[corr.indexInSecond];

                        // Compute weighted influence from control nodes
                        angle_curr = (ScalarT)0.0;
                        trans_curr.setZero();
                        for (size_t j = 0; j < ctrl_neighbors.size(); j++) {
                            const auto offset = 3*ctrl_neighbors[j].index;
                            angle_curr += ctrl_neighbors[j].value*tforms_vec[offset];
                            trans_curr.noalias() += ctrl_neighbors[j].value*tforms_vec.template segment<2>(offset + 1);
                        }
                        if (total_weight[corr.indexInSecond] != (ScalarT)0.0) {
                            weight = (ScalarT)(1.0)/total_weight[corr.indexInSecond];
                            angle_curr *= weight;
                            trans_curr *= weight;

                            corr_weight_sqrt = point_to_plane_weight_sqrt*std::sqrt(plane_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value));
                            corr_weight_nrm = corr_weight_sqrt/total_weight[corr.indexInSecond];
                        } else {
                            corr_weight_sqrt = (ScalarT)0.0;
                            corr_weight_nrm = (ScalarT)0.0;
                        }

                        const auto d = dst_p.col(corr.indexInFirst);
                        const auto n = dst_n.col(corr.indexInFirst);
                        const auto s = src_p.col(corr.indexInSecond);

                        const ScalarT cosa = std::cos(angle_curr);
                        const ScalarT sina = std::sin(angle_curr);

                        Eigen::Matrix<ScalarT,2,1> s_t(cosa*s[0] - sina*s[1] + trans_curr[0], sina*s[0] + cosa*s[1] + trans_curr[1]);

                        eq_ind = num_point_to_point_equations + i;

                        const ScalarT dot_val = (n[0]*(-sina*s[0] - cosa*s[1]) + n[1]*(cosa*s[0] - sina*s[1]));

                        for (size_t j = 0; j < ctrl_neighbors.size(); j++) {
                            const auto offset = 3*ctrl_neighbors[j].index;
                            weight = corr_weight_nrm*ctrl_neighbors[j].value;

                            // Point to plane
                            nz_ind = outer_ptr[eq_ind] + 3*j;
                            values[nz_ind] = dot_val*weight;
                            inner_ind[nz_ind++] = offset;
                            values[nz_ind] = n[0]*weight;
                            inner_ind[nz_ind++] = offset + 1;
                            values[nz_ind] = n[1]*weight;
                            inner_ind[nz_ind++] = offset + 2;
                        }

                        b[eq_ind] = n.dot(d - s_t)*corr_weight_sqrt;
                    }
                }

                // Regularization
#pragma omp for nowait
                for (size_t i = 0; i < regularization_neighborhoods.size(); i++) {
                    eq_ind = num_data_term_equations + reg_eq_ind[i];
                    nz_ind = outer_ptr[num_data_term_equations] + 2*reg_eq_ind[i];
                    const auto& neighbors = regularization_neighborhoods[i];

                    for (size_t j = 1; j < neighbors.size(); j++) {
                        auto s_offset = 3*neighbors[0].index;
                        auto n_offset = 3*neighbors[j].index;
                        weight = regularization_weight_sqrt*std::sqrt(reg_evaluator(neighbors[0].index, neighbors[j].index, neighbors[j].value));

                        if (n_offset < s_offset) std::swap(s_offset, n_offset);

                        diff = tforms_vec[s_offset + 0] - tforms_vec[n_offset + 0];
                        d_sqrt_huber_loss = weight*internal::sqrtHuberLossDerivative<ScalarT>(diff, huber_boundary);
                        values[nz_ind] = d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = s_offset;
                        values[nz_ind] = -d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = n_offset;
                        b[eq_ind++] = -weight*internal::sqrtHuberLoss<ScalarT>(diff, huber_boundary);

                        diff = tforms_vec[s_offset + 1] - tforms_vec[n_offset + 1];
                        d_sqrt_huber_loss = weight*internal::sqrtHuberLossDerivative<ScalarT>(diff, huber_boundary);
                        values[nz_ind] = d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = s_offset + 1;
                        values[nz_ind] = -d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = n_offset + 1;
                        b[eq_ind++] = -weight*internal::sqrtHuberLoss<ScalarT>(diff, huber_boundary);

                        diff = tforms_vec[s_offset + 2] - tforms_vec[n_offset + 2];
                        d_sqrt_huber_loss = weight*internal::sqrtHuberLossDerivative<ScalarT>(diff, huber_boundary);
                        values[nz_ind] = d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = s_offset + 2;
                        values[nz_ind] = -d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = n_offset + 2;
                        b[eq_ind++] = -weight*internal::sqrtHuberLoss<ScalarT>(diff, huber_boundary);
                    }
                }
            }

            // Solve linear system using CG
            AtA = At*At.transpose();
            Atb.noalias() = At*b;

            if (iter == 0) solver.analyzePattern(AtA);
            solver.factorize(AtA);
            delta = solver.solve(Atb);
            tforms_vec += delta;

            iter++;

            // Check for convergence
            max_delta_sq = (ScalarT)0.0;
#pragma omp parallel for private (curr_delta_sq) reduction (max: max_delta_sq)
            for (size_t i = 0; i < num_ctrl_points; i++) {
                curr_delta_sq = delta.template segment<3>(3*i).squaredNorm();
                if (curr_delta_sq > max_delta_sq) max_delta_sq = curr_delta_sq;
            }

            if (max_delta_sq < gn_conv_tol_sq) {
                has_converged = true;
                break;
            }
        }

        // Convert to output format
        transforms.resize(num_ctrl_points);
#pragma omp parallel for
        for (size_t i = 0; i < transforms.size(); i++) {
            transforms[i].linear().noalias() = Eigen::Rotation2D<ScalarT>(tforms_vec[3*i]).toRotationMatrix();
            transforms[i].translation() = tforms_vec.template segment<2>(3*i + 1);
        }

        return has_converged;
    }

    // Locally rigid sparse warp field, 3D
    template <class TransformT, class PointCorrSetT, class PlaneCorrSetT, class CtrlNeighborhoodSetT, class RegNeighborhoodSetT, class PointCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class PlaneCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class ControlWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class RegWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>>
    typename std::enable_if<int(TransformT::Mode) == int(Eigen::Isometry) && TransformT::Dim == 3,bool>::type
    estimateSparseWarpFieldCombinedMetric(const ConstVectorSetMatrixMap<typename TransformT::Scalar,3> &dst_p,
                                          const ConstVectorSetMatrixMap<typename TransformT::Scalar,3> &dst_n,
                                          const ConstVectorSetMatrixMap<typename TransformT::Scalar,3> &src_p,
                                          const PointCorrSetT &point_to_point_correspondences,
                                          typename TransformT::Scalar point_to_point_weight,
                                          const PlaneCorrSetT &point_to_plane_correspondences,
                                          typename TransformT::Scalar point_to_plane_weight,
                                          const CtrlNeighborhoodSetT &src_to_ctrl_neighborhoods,
                                          size_t num_ctrl_points,
                                          const RegNeighborhoodSetT &regularization_neighborhoods,
                                          typename TransformT::Scalar regularization_weight,
                                          TransformSet<TransformT> &transforms,
                                          typename TransformT::Scalar huber_boundary = (typename TransformT::Scalar)(1e-4),
                                          size_t max_gn_iter = 10,
                                          typename TransformT::Scalar gn_conv_tol = (typename TransformT::Scalar)1e-5,
                                          size_t max_cg_iter = 1000,
                                          typename TransformT::Scalar cg_conv_tol = (typename TransformT::Scalar)1e-5,
                                          const PointCorrWeightEvaluatorT &point_corr_evaluator = PointCorrWeightEvaluatorT(),
                                          const PlaneCorrWeightEvaluatorT &plane_corr_evaluator = PlaneCorrWeightEvaluatorT(),
                                          const ControlWeightEvaluatorT &control_evaluator = ControlWeightEvaluatorT(),
                                          const RegWeightEvaluatorT &reg_evaluator = RegWeightEvaluatorT())
    {
        typedef typename TransformT::Scalar ScalarT;

        const bool has_point_to_point_terms = !point_to_point_correspondences.empty() && (point_to_point_weight > (ScalarT)0.0);
        const bool has_point_to_plane_terms = !point_to_plane_correspondences.empty() && (point_to_plane_weight > (ScalarT)0.0);

        if (src_to_ctrl_neighborhoods.size() != src_p.cols() ||
            (!has_point_to_point_terms && !has_point_to_plane_terms) ||
            (has_point_to_plane_terms && dst_p.cols() != dst_n.cols()))
        {
            transforms.resize(num_ctrl_points);
            transforms.setIdentity();
            return false;
        }

        // Sort control nodes by index and compute total weight
        CtrlNeighborhoodSetT src_to_ctrl_sorted(src_to_ctrl_neighborhoods.size());
        std::vector<ScalarT> total_weight(src_to_ctrl_sorted.size());
        std::vector<char> has_data_term(src_to_ctrl_neighborhoods.size(), 0);
#pragma omp parallel shared (src_to_ctrl_sorted, total_weight, has_data_term)
        {
            if (has_point_to_point_terms) {
#pragma omp for nowait
                for (size_t i = 0; i < point_to_point_correspondences.size(); i++) {
                    has_data_term[point_to_point_correspondences[i].indexInSecond] = 1;
                }
            }

            if (has_point_to_plane_terms) {
#pragma omp for
                for (size_t i = 0; i < point_to_plane_correspondences.size(); i++) {
                    has_data_term[point_to_plane_correspondences[i].indexInSecond] = 1;
                }
            }

#pragma omp for schedule (dynamic)
            for (size_t i = 0; i < has_data_term.size(); i++) {
                if (has_data_term[i]) {
                    total_weight[i] = (ScalarT)0.0;
                    src_to_ctrl_sorted[i].resize(src_to_ctrl_neighborhoods[i].size());
                    for (size_t j = 0; j < src_to_ctrl_neighborhoods[i].size(); j++) {
                        src_to_ctrl_sorted[i][j].index = src_to_ctrl_neighborhoods[i][j].index;
                        src_to_ctrl_sorted[i][j].value = control_evaluator(i, src_to_ctrl_neighborhoods[i][j].index, src_to_ctrl_neighborhoods[i][j].value);
                        total_weight[i] += src_to_ctrl_sorted[i][j].value;
                    }
                    std::sort(src_to_ctrl_sorted[i].begin(), src_to_ctrl_sorted[i].end(), typename CtrlNeighborhoodSetT::value_type::value_type::IndexLessComparator());
                }
            }
        }

        // Get regularization equation count and indices
        std::vector<size_t> reg_eq_ind(regularization_neighborhoods.size());
        size_t num_reg_arcs = 0;
        if (!regularization_neighborhoods.empty()) {
            reg_eq_ind[0] = 0;
            num_reg_arcs = std::max((size_t)0, regularization_neighborhoods[0].size() - 1);
        }
        for (size_t i = 1; i < regularization_neighborhoods.size(); i++) {
            reg_eq_ind[i] = reg_eq_ind[i-1] + 6*std::max((size_t)0, regularization_neighborhoods[i-1].size() - 1);
            num_reg_arcs += std::max((size_t)0, regularization_neighborhoods[i].size() - 1);
        }

        // Compute number of equations and unknowns
        const size_t num_unknowns = 6*num_ctrl_points;
        const size_t num_point_to_point_equations = 3*has_point_to_point_terms*point_to_point_correspondences.size();
        const size_t num_point_to_plane_equations = has_point_to_plane_terms*point_to_plane_correspondences.size();
        const size_t num_data_term_equations = num_point_to_point_equations + num_point_to_plane_equations;
        const size_t num_regularization_equations = 6*num_reg_arcs;
        const size_t num_equations = num_data_term_equations + num_regularization_equations;

        // Jacobian
        Eigen::SparseMatrix<ScalarT> At(num_unknowns, num_equations);
        // Outer pointers
        typename Eigen::SparseMatrix<ScalarT>::StorageIndex * const outer_ptr = At.outerIndexPtr();
        outer_ptr[0] = 0;
        if (has_point_to_point_terms) {
            for (size_t i = 0; i < point_to_point_correspondences.size(); i++) {
                const size_t nnz_per_eq = 6*src_to_ctrl_sorted[point_to_point_correspondences[i].indexInSecond].size();
                outer_ptr[3*i + 1] = outer_ptr[3*i] + nnz_per_eq;
                outer_ptr[3*i + 2] = outer_ptr[3*i] + nnz_per_eq + nnz_per_eq;
                outer_ptr[3*i + 3] = outer_ptr[3*i] + nnz_per_eq + nnz_per_eq + nnz_per_eq;
            }
        }
        if (has_point_to_plane_terms) {
            for (size_t i = 0; i < point_to_plane_correspondences.size(); i++) {
                outer_ptr[num_point_to_point_equations + i + 1] = outer_ptr[num_point_to_point_equations + i] + 6*src_to_ctrl_sorted[point_to_plane_correspondences[i].indexInSecond].size();
            }
        }
#pragma omp parallel for
        for (size_t i = 1; i < num_regularization_equations + 1; i++) {
            outer_ptr[num_data_term_equations + i] = outer_ptr[num_data_term_equations] + 2*i;
        }
        At.reserve(outer_ptr[num_equations]);
        // Values
        ScalarT * const values = At.valuePtr();
        // Inner indices
        typename Eigen::SparseMatrix<ScalarT>::StorageIndex * const inner_ind = At.innerIndexPtr();

        // Vector of (negative) residuals
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> b(num_equations);

        // Vector of unknowns (Euler angles and translation offsets per control node)
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> tforms_vec(Eigen::Matrix<ScalarT,Eigen::Dynamic,1>::Zero(num_unknowns, 1));

        // Conjugate Gradient solver
//        Eigen::ConjugateGradient<Eigen::SparseMatrix<ScalarT>,Eigen::Lower|Eigen::Upper,Eigen::IncompleteCholesky<ScalarT,Eigen::Lower|Eigen::Upper>> solver;
//        Eigen::ConjugateGradient<Eigen::SparseMatrix<ScalarT>,Eigen::Lower|Eigen::Upper,Eigen::IdentityPreconditioner> solver;
        Eigen::ConjugateGradient<Eigen::SparseMatrix<ScalarT>,Eigen::Lower|Eigen::Upper,Eigen::DiagonalPreconditioner<ScalarT>> solver;
//        Eigen::ConjugateGradient<Eigen::SparseMatrix<ScalarT>,Eigen::Lower|Eigen::Upper,BlockDiagonalPreconditioner<ScalarT,6>> solver;
        solver.setMaxIterations(max_cg_iter);
        solver.setTolerance(cg_conv_tol);

        Eigen::SparseMatrix<ScalarT> AtA;
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> Atb;

        // Parameters
        const ScalarT point_to_point_weight_sqrt = std::sqrt(point_to_point_weight);
        const ScalarT point_to_plane_weight_sqrt = std::sqrt(point_to_plane_weight);
        const ScalarT regularization_weight_sqrt = std::sqrt(regularization_weight);
        const ScalarT gn_conv_tol_sq = gn_conv_tol*gn_conv_tol;

        // Temporaries
        Eigen::Matrix<ScalarT,3,3> rot_coeffs, d_rot_coeffs_da, d_rot_coeffs_db, d_rot_coeffs_dc;
        Eigen::Matrix<ScalarT,3,1> trans_s, d_rot_da_s, d_rot_db_s, d_rot_dc_s;
        Eigen::Matrix<ScalarT,3,1> angles_curr, trans_curr;
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> delta;
        ScalarT weight, corr_weight_sqrt, corr_weight_nrm, diff, d_sqrt_huber_loss, curr_delta_sq, max_delta_sq;
        size_t eq_ind, nz_ind;

        bool has_converged = false;
        size_t iter = 0;
        while (iter < max_gn_iter) {
#pragma omp parallel shared (At, b) private (eq_ind, nz_ind, weight, corr_weight_sqrt, corr_weight_nrm, diff, d_sqrt_huber_loss, angles_curr, trans_curr, rot_coeffs, d_rot_coeffs_da, d_rot_coeffs_db, d_rot_coeffs_dc, trans_s, d_rot_da_s, d_rot_db_s, d_rot_dc_s)
            {
                // Data term
                if (has_point_to_point_terms) {
#pragma omp for nowait
                    for (size_t i = 0; i < point_to_point_correspondences.size(); i++) {
                        const auto& corr = point_to_point_correspondences[i];
                        const auto& ctrl_neighbors = src_to_ctrl_sorted[corr.indexInSecond];

                        // Compute weighted influence from control nodes
                        angles_curr.setZero();
                        trans_curr.setZero();
                        for (size_t j = 0; j < ctrl_neighbors.size(); j++) {
                            const auto offset = 6*ctrl_neighbors[j].index;
                            angles_curr.noalias() += ctrl_neighbors[j].value*tforms_vec.template segment<3>(offset);
                            trans_curr.noalias() += ctrl_neighbors[j].value*tforms_vec.template segment<3>(offset + 3);
                        }
                        if (total_weight[corr.indexInSecond] != (ScalarT)0.0) {
                            weight = (ScalarT)(1.0)/total_weight[corr.indexInSecond];
                            angles_curr *= weight;
                            trans_curr *= weight;

                            corr_weight_sqrt = point_to_point_weight_sqrt*std::sqrt(point_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value));
                            corr_weight_nrm = corr_weight_sqrt/total_weight[corr.indexInSecond];
                        } else {
                            corr_weight_sqrt = (ScalarT)0.0;
                            corr_weight_nrm = (ScalarT)0.0;
                        }

                        const auto d = dst_p.col(corr.indexInFirst);
                        const auto s = src_p.col(corr.indexInSecond);

                        internal::computeRotationTerms(angles_curr[0], angles_curr[1], angles_curr[2], rot_coeffs, d_rot_coeffs_da, d_rot_coeffs_db, d_rot_coeffs_dc);

                        trans_s.noalias() = d - (rot_coeffs.transpose()*s + trans_curr);
                        d_rot_da_s.noalias() = d_rot_coeffs_da.transpose()*s;
                        d_rot_db_s.noalias() = d_rot_coeffs_db.transpose()*s;
                        d_rot_dc_s.noalias() = d_rot_coeffs_dc.transpose()*s;

                        eq_ind = 3*i;

                        for (size_t j = 0; j < ctrl_neighbors.size(); j++) {
                            const auto offset = 6*ctrl_neighbors[j].index;
                            weight = corr_weight_nrm*ctrl_neighbors[j].value;

                            nz_ind = outer_ptr[eq_ind] + 6*j;
                            values[nz_ind] = d_rot_da_s[0]*weight;
                            inner_ind[nz_ind++] = offset;
                            values[nz_ind] = d_rot_db_s[0]*weight;
                            inner_ind[nz_ind++] = offset + 1;
                            values[nz_ind] = d_rot_dc_s[0]*weight;
                            inner_ind[nz_ind++] = offset + 2;
                            values[nz_ind] = weight;
                            inner_ind[nz_ind++] = offset + 3;
                            values[nz_ind] = (ScalarT)0.0;
                            inner_ind[nz_ind++] = offset + 4;
                            values[nz_ind] = (ScalarT)0.0;
                            inner_ind[nz_ind++] = offset + 5;

                            nz_ind = outer_ptr[eq_ind + 1] + 6*j;
                            values[nz_ind] = d_rot_da_s[1]*weight;
                            inner_ind[nz_ind++] = offset;
                            values[nz_ind] = d_rot_db_s[1]*weight;
                            inner_ind[nz_ind++] = offset + 1;
                            values[nz_ind] = d_rot_dc_s[1]*weight;
                            inner_ind[nz_ind++] = offset + 2;
                            values[nz_ind] = (ScalarT)0.0;
                            inner_ind[nz_ind++] = offset + 3;
                            values[nz_ind] = weight;
                            inner_ind[nz_ind++] = offset + 4;
                            values[nz_ind] = (ScalarT)0.0;
                            inner_ind[nz_ind++] = offset + 5;

                            nz_ind = outer_ptr[eq_ind + 2] + 6*j;
                            values[nz_ind] = d_rot_da_s[2]*weight;
                            inner_ind[nz_ind++] = offset;
                            values[nz_ind] = d_rot_db_s[2]*weight;
                            inner_ind[nz_ind++] = offset + 1;
                            values[nz_ind] = d_rot_dc_s[2]*weight;
                            inner_ind[nz_ind++] = offset + 2;
                            values[nz_ind] = (ScalarT)0.0;
                            inner_ind[nz_ind++] = offset + 3;
                            values[nz_ind] = (ScalarT)0.0;
                            inner_ind[nz_ind++] = offset + 4;
                            values[nz_ind] = weight;
                            inner_ind[nz_ind++] = offset + 5;
                        }

                        b.template segment<3>(eq_ind) = trans_s*corr_weight_sqrt;
                    }
                }

                if (has_point_to_plane_terms) {
#pragma omp for nowait
                    for (size_t i = 0; i < point_to_plane_correspondences.size(); i++) {
                        const auto& corr = point_to_plane_correspondences[i];
                        const auto& ctrl_neighbors = src_to_ctrl_sorted[corr.indexInSecond];

                        // Compute weighted influence from control nodes
                        angles_curr.setZero();
                        trans_curr.setZero();
                        for (size_t j = 0; j < ctrl_neighbors.size(); j++) {
                            const auto offset = 6*ctrl_neighbors[j].index;
                            angles_curr.noalias() += ctrl_neighbors[j].value*tforms_vec.template segment<3>(offset);
                            trans_curr.noalias() += ctrl_neighbors[j].value*tforms_vec.template segment<3>(offset + 3);
                        }
                        if (total_weight[corr.indexInSecond] != (ScalarT)0.0) {
                            weight = (ScalarT)(1.0)/total_weight[corr.indexInSecond];
                            angles_curr *= weight;
                            trans_curr *= weight;

                            corr_weight_sqrt = point_to_plane_weight_sqrt*std::sqrt(plane_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value));
                            corr_weight_nrm = corr_weight_sqrt/total_weight[corr.indexInSecond];
                        } else {
                            corr_weight_sqrt = (ScalarT)0.0;
                            corr_weight_nrm = (ScalarT)0.0;
                        }

                        const auto d = dst_p.col(corr.indexInFirst);
                        const auto n = dst_n.col(corr.indexInFirst);
                        const auto s = src_p.col(corr.indexInSecond);

                        internal::computeRotationTerms(angles_curr[0], angles_curr[1], angles_curr[2], rot_coeffs, d_rot_coeffs_da, d_rot_coeffs_db, d_rot_coeffs_dc);

                        trans_s.noalias() = d - (rot_coeffs.transpose()*s + trans_curr);
                        d_rot_da_s.noalias() = d_rot_coeffs_da.transpose()*s;
                        d_rot_db_s.noalias() = d_rot_coeffs_db.transpose()*s;
                        d_rot_dc_s.noalias() = d_rot_coeffs_dc.transpose()*s;

                        eq_ind = num_point_to_point_equations + i;

                        const ScalarT dot1 = n.dot(d_rot_da_s);
                        const ScalarT dot2 = n.dot(d_rot_db_s);
                        const ScalarT dot3 = n.dot(d_rot_dc_s);

                        for (size_t j = 0; j < ctrl_neighbors.size(); j++) {
                            const auto offset = 6*ctrl_neighbors[j].index;
                            weight = corr_weight_nrm*ctrl_neighbors[j].value;

                            // Point to plane
                            nz_ind = outer_ptr[eq_ind] + 6*j;
                            values[nz_ind] = dot1*weight;
                            inner_ind[nz_ind++] = offset;
                            values[nz_ind] = dot2*weight;
                            inner_ind[nz_ind++] = offset + 1;
                            values[nz_ind] = dot3*weight;
                            inner_ind[nz_ind++] = offset + 2;
                            values[nz_ind] = n[0]*weight;
                            inner_ind[nz_ind++] = offset + 3;
                            values[nz_ind] = n[1]*weight;
                            inner_ind[nz_ind++] = offset + 4;
                            values[nz_ind] = n[2]*weight;
                            inner_ind[nz_ind++] = offset + 5;
                        }

                        b[eq_ind] = n.dot(trans_s)*corr_weight_sqrt;
                    }
                }

                // Regularization
#pragma omp for nowait
                for (size_t i = 0; i < regularization_neighborhoods.size(); i++) {
                    eq_ind = num_data_term_equations + reg_eq_ind[i];
                    nz_ind = outer_ptr[num_data_term_equations] + 2*reg_eq_ind[i];
                    const auto& neighbors = regularization_neighborhoods[i];

                    for (size_t j = 1; j < neighbors.size(); j++) {
                        auto s_offset = 6*neighbors[0].index;
                        auto n_offset = 6*neighbors[j].index;
                        weight = regularization_weight_sqrt*std::sqrt(reg_evaluator(neighbors[0].index, neighbors[j].index, neighbors[j].value));

                        if (n_offset < s_offset) std::swap(s_offset, n_offset);

                        diff = tforms_vec[s_offset + 0] - tforms_vec[n_offset + 0];
                        d_sqrt_huber_loss = weight*internal::sqrtHuberLossDerivative<ScalarT>(diff, huber_boundary);
                        values[nz_ind] = d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = s_offset;
                        values[nz_ind] = -d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = n_offset;
                        b[eq_ind++] = -weight*internal::sqrtHuberLoss<ScalarT>(diff, huber_boundary);

                        diff = tforms_vec[s_offset + 1] - tforms_vec[n_offset + 1];
                        d_sqrt_huber_loss = weight*internal::sqrtHuberLossDerivative<ScalarT>(diff, huber_boundary);
                        values[nz_ind] = d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = s_offset + 1;
                        values[nz_ind] = -d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = n_offset + 1;
                        b[eq_ind++] = -weight*internal::sqrtHuberLoss<ScalarT>(diff, huber_boundary);

                        diff = tforms_vec[s_offset + 2] - tforms_vec[n_offset + 2];
                        d_sqrt_huber_loss = weight*internal::sqrtHuberLossDerivative<ScalarT>(diff, huber_boundary);
                        values[nz_ind] = d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = s_offset + 2;
                        values[nz_ind] = -d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = n_offset + 2;
                        b[eq_ind++] = -weight*internal::sqrtHuberLoss<ScalarT>(diff, huber_boundary);

                        diff = tforms_vec[s_offset + 3] - tforms_vec[n_offset + 3];
                        d_sqrt_huber_loss = weight*internal::sqrtHuberLossDerivative<ScalarT>(diff, huber_boundary);
                        values[nz_ind] = d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = s_offset + 3;
                        values[nz_ind] = -d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = n_offset + 3;
                        b[eq_ind++] = -weight*internal::sqrtHuberLoss<ScalarT>(diff, huber_boundary);

                        diff = tforms_vec[s_offset + 4] - tforms_vec[n_offset + 4];
                        d_sqrt_huber_loss = weight*internal::sqrtHuberLossDerivative<ScalarT>(diff, huber_boundary);
                        values[nz_ind] = d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = s_offset + 4;
                        values[nz_ind] = -d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = n_offset + 4;
                        b[eq_ind++] = -weight*internal::sqrtHuberLoss<ScalarT>(diff, huber_boundary);

                        diff = tforms_vec[s_offset + 5] - tforms_vec[n_offset + 5];
                        d_sqrt_huber_loss = weight*internal::sqrtHuberLossDerivative<ScalarT>(diff, huber_boundary);
                        values[nz_ind] = d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = s_offset + 5;
                        values[nz_ind] = -d_sqrt_huber_loss;
                        inner_ind[nz_ind++] = n_offset + 5;
                        b[eq_ind++] = -weight*internal::sqrtHuberLoss<ScalarT>(diff, huber_boundary);
                    }
                }
            }

            // Solve linear system using CG
            AtA = At*At.transpose();
            Atb.noalias() = At*b;

            if (iter == 0) solver.analyzePattern(AtA);
            solver.factorize(AtA);
            delta = solver.solve(Atb);
            tforms_vec += delta;

            iter++;

            // Check for convergence
            max_delta_sq = (ScalarT)0.0;
#pragma omp parallel for private (curr_delta_sq) reduction (max: max_delta_sq)
            for (size_t i = 0; i < num_ctrl_points; i++) {
                curr_delta_sq = delta.template segment<6>(6*i).squaredNorm();
                if (curr_delta_sq > max_delta_sq) max_delta_sq = curr_delta_sq;
            }

            if (max_delta_sq < gn_conv_tol_sq) {
                has_converged = true;
                break;
            }
        }

        // Convert to output format
        transforms.resize(num_ctrl_points);
#pragma omp parallel for
        for (size_t i = 0; i < transforms.size(); i++) {
            transforms[i].linear().noalias() = (Eigen::AngleAxis<ScalarT>(tforms_vec[6*i + 2],Eigen::Matrix<ScalarT,3,1>::UnitZ()) *
                                                Eigen::AngleAxis<ScalarT>(tforms_vec[6*i + 1],Eigen::Matrix<ScalarT,3,1>::UnitY()) *
                                                Eigen::AngleAxis<ScalarT>(tforms_vec[6*i + 0],Eigen::Matrix<ScalarT,3,1>::UnitX())).matrix();
            transforms[i].linear() = transforms[i].rotation();
            transforms[i].translation() = tforms_vec.template segment<3>(6*i + 3);
        }

        return has_converged;
    }

    // Locally affine sparse warp field, general dimension
    template <class TransformT, class PointCorrSetT, class PlaneCorrSetT, class CtrlNeighborhoodSetT, class RegNeighborhoodSetT, class PointCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class PlaneCorrWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class ControlWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>, class RegWeightEvaluatorT = UnityWeightEvaluator<typename TransformT::Scalar,typename TransformT::Scalar>>
    typename std::enable_if<int(TransformT::Mode) == int(Eigen::Affine) || int(TransformT::Mode) == int(Eigen::AffineCompact),bool>::type
    estimateSparseWarpFieldCombinedMetric(const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_p,
                                          const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &dst_n,
                                          const ConstVectorSetMatrixMap<typename TransformT::Scalar,TransformT::Dim> &src_p,
                                          const PointCorrSetT &point_to_point_correspondences,
                                          typename TransformT::Scalar point_to_point_weight,
                                          const PlaneCorrSetT &point_to_plane_correspondences,
                                          typename TransformT::Scalar point_to_plane_weight,
                                          const CtrlNeighborhoodSetT &src_to_ctrl_neighborhoods,
                                          size_t num_ctrl_points,
                                          const RegNeighborhoodSetT &regularization_neighborhoods,
                                          typename TransformT::Scalar regularization_weight,
                                          TransformSet<TransformT> &transforms,
                                          typename TransformT::Scalar huber_boundary = (typename TransformT::Scalar)(1e-4),
                                          size_t max_gn_iter = 10,
                                          typename TransformT::Scalar gn_conv_tol = (typename TransformT::Scalar)1e-5,
                                          size_t max_cg_iter = 1000,
                                          typename TransformT::Scalar cg_conv_tol = (typename TransformT::Scalar)1e-5,
                                          const PointCorrWeightEvaluatorT &point_corr_evaluator = PointCorrWeightEvaluatorT(),
                                          const PlaneCorrWeightEvaluatorT &plane_corr_evaluator = PlaneCorrWeightEvaluatorT(),
                                          const ControlWeightEvaluatorT &control_evaluator = ControlWeightEvaluatorT(),
                                          const RegWeightEvaluatorT &reg_evaluator = RegWeightEvaluatorT())
    {
        typedef typename TransformT::Scalar ScalarT;
        enum {
            Dim = TransformT::Dim,
            NumUnknownsLocal = TransformT::Dim*(TransformT::Dim + 1),
            NumNonZerosPointToPoint = TransformT::Dim + 1,
            NumNonZerosPointToPlane = TransformT::Dim*(TransformT::Dim + 1)
        };

        const bool has_point_to_point_terms = !point_to_point_correspondences.empty() && (point_to_point_weight > (ScalarT)0.0);
        const bool has_point_to_plane_terms = !point_to_plane_correspondences.empty() && (point_to_plane_weight > (ScalarT)0.0);

        if (src_to_ctrl_neighborhoods.size() != src_p.cols() ||
            (!has_point_to_point_terms && !has_point_to_plane_terms) ||
            (has_point_to_plane_terms && dst_p.cols() != dst_n.cols()))
        {
            transforms.resize(num_ctrl_points);
            transforms.setIdentity();
            return false;
        }

        // Sort control nodes by index and compute total weight
        CtrlNeighborhoodSetT src_to_ctrl_sorted(src_to_ctrl_neighborhoods.size());
        std::vector<ScalarT> total_weight(src_to_ctrl_sorted.size());
        std::vector<char> has_data_term(src_to_ctrl_neighborhoods.size(), 0);
#pragma omp parallel shared (src_to_ctrl_sorted, total_weight, has_data_term)
        {
            if (has_point_to_point_terms) {
#pragma omp for nowait
                for (size_t i = 0; i < point_to_point_correspondences.size(); i++) {
                    has_data_term[point_to_point_correspondences[i].indexInSecond] = 1;
                }
            }

            if (has_point_to_plane_terms) {
#pragma omp for
                for (size_t i = 0; i < point_to_plane_correspondences.size(); i++) {
                    has_data_term[point_to_plane_correspondences[i].indexInSecond] = 1;
                }
            }

#pragma omp for schedule (dynamic)
            for (size_t i = 0; i < has_data_term.size(); i++) {
                if (has_data_term[i]) {
                    total_weight[i] = (ScalarT)0.0;
                    src_to_ctrl_sorted[i].resize(src_to_ctrl_neighborhoods[i].size());
                    for (size_t j = 0; j < src_to_ctrl_neighborhoods[i].size(); j++) {
                        src_to_ctrl_sorted[i][j].index = src_to_ctrl_neighborhoods[i][j].index;
                        src_to_ctrl_sorted[i][j].value = control_evaluator(i, src_to_ctrl_neighborhoods[i][j].index, src_to_ctrl_neighborhoods[i][j].value);
                        total_weight[i] += src_to_ctrl_sorted[i][j].value;
                    }
                    std::sort(src_to_ctrl_sorted[i].begin(), src_to_ctrl_sorted[i].end(), typename CtrlNeighborhoodSetT::value_type::value_type::IndexLessComparator());
                }
            }
        }

        // Get regularization equation count and indices
        std::vector<size_t> reg_eq_ind(regularization_neighborhoods.size());
        size_t num_reg_arcs = 0;
        if (!regularization_neighborhoods.empty()) {
            reg_eq_ind[0] = 0;
            num_reg_arcs = std::max((size_t)0, regularization_neighborhoods[0].size() - 1);
        }
        for (size_t i = 1; i < regularization_neighborhoods.size(); i++) {
            reg_eq_ind[i] = reg_eq_ind[i-1] + NumUnknownsLocal*std::max((size_t)0, regularization_neighborhoods[i-1].size() - 1);
            num_reg_arcs += std::max((size_t)0, regularization_neighborhoods[i].size() - 1);
        }

        // Compute number of equations and unknowns
        const size_t num_unknowns = NumUnknownsLocal*num_ctrl_points;
        const size_t num_point_to_point_equations = Dim*has_point_to_point_terms*point_to_point_correspondences.size();
        const size_t num_point_to_plane_equations = has_point_to_plane_terms*point_to_plane_correspondences.size();
        const size_t num_data_term_equations = num_point_to_point_equations + num_point_to_plane_equations;
        const size_t num_regularization_equations = NumUnknownsLocal*num_reg_arcs;
        const size_t num_equations = num_data_term_equations + num_regularization_equations;

        // Jacobian
        Eigen::SparseMatrix<ScalarT> At(num_unknowns, num_equations);
        // Outer pointers
        typename Eigen::SparseMatrix<ScalarT>::StorageIndex * const outer_ptr = At.outerIndexPtr();
        outer_ptr[0] = 0;
        if (has_point_to_point_terms) {
            for (size_t i = 0; i < point_to_point_correspondences.size(); i++) {
                const size_t nnz_per_eq = NumNonZerosPointToPoint*src_to_ctrl_sorted[point_to_point_correspondences[i].indexInSecond].size();
                for (size_t j = 0; j < Dim; j++) {
                    outer_ptr[Dim*i + j + 1] = outer_ptr[Dim*i + j] + nnz_per_eq;
                }
            }
        }
        if (has_point_to_plane_terms) {
            for (size_t i = 0; i < point_to_plane_correspondences.size(); i++) {
                outer_ptr[num_point_to_point_equations + i + 1] = outer_ptr[num_point_to_point_equations + i] + NumNonZerosPointToPlane*src_to_ctrl_sorted[point_to_plane_correspondences[i].indexInSecond].size();
            }
        }
#pragma omp parallel for
        for (size_t i = 1; i < num_regularization_equations + 1; i++) {
            outer_ptr[num_data_term_equations + i] = outer_ptr[num_data_term_equations] + 2*i;
        }
        At.reserve(outer_ptr[num_equations]);
        // Values
        ScalarT * const values = At.valuePtr();
        // Inner indices
        typename Eigen::SparseMatrix<ScalarT>::StorageIndex * const inner_ind = At.innerIndexPtr();

        // Vector of (negative) residuals
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> b(num_equations);

        // Vector of unknowns
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> tforms_vec(num_unknowns, 1);
#pragma omp parallel for
        for (size_t i = 0; i < num_ctrl_points; i++) {
            Eigen::Map<Eigen::Matrix<ScalarT,Dim,Dim,Eigen::RowMajor>>(tforms_vec.data() + i*NumUnknownsLocal, Dim, Dim).setIdentity();
            tforms_vec.template segment<Dim>(i*NumUnknownsLocal + Dim*Dim).setZero();
        }

        // Conjugate Gradient solver
//        Eigen::ConjugateGradient<Eigen::SparseMatrix<ScalarT>,Eigen::Lower|Eigen::Upper,Eigen::IncompleteCholesky<ScalarT,Eigen::Lower|Eigen::Upper>> solver;
//        Eigen::ConjugateGradient<Eigen::SparseMatrix<ScalarT>,Eigen::Lower|Eigen::Upper,Eigen::IdentityPreconditioner> solver;
        Eigen::ConjugateGradient<Eigen::SparseMatrix<ScalarT>,Eigen::Lower|Eigen::Upper,Eigen::DiagonalPreconditioner<ScalarT>> solver;
        solver.setMaxIterations(max_cg_iter);
        solver.setTolerance(cg_conv_tol);

        Eigen::SparseMatrix<ScalarT> AtA;
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> Atb;

        // Parameters
        const ScalarT point_to_point_weight_sqrt = std::sqrt(point_to_point_weight);
        const ScalarT point_to_plane_weight_sqrt = std::sqrt(point_to_plane_weight);
        const ScalarT regularization_weight_sqrt = std::sqrt(regularization_weight);
        const ScalarT gn_conv_tol_sq = gn_conv_tol*gn_conv_tol;

        // Temporaries
        Eigen::Matrix<ScalarT,Dim*Dim,1> linear_curr;
        Eigen::Matrix<ScalarT,Dim,1> trans_curr;
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> delta;
        ScalarT weight, corr_weight_sqrt, corr_weight_nrm, diff, d_sqrt_huber_loss, curr_delta_sq, max_delta_sq;
        size_t eq_ind, nz_ind;

        bool has_converged = false;
        size_t iter = 0;
        while (iter < max_gn_iter) {
#pragma omp parallel shared (At, b) private (eq_ind, nz_ind, weight, corr_weight_sqrt, corr_weight_nrm, diff, d_sqrt_huber_loss, linear_curr, trans_curr)
            {
                // Data term
                if (has_point_to_point_terms) {
#pragma omp for nowait
                    for (size_t i = 0; i < point_to_point_correspondences.size(); i++) {
                        const auto& corr = point_to_point_correspondences[i];
                        const auto& ctrl_neighbors = src_to_ctrl_sorted[corr.indexInSecond];

                        // Compute weighted influence from control nodes
                        linear_curr.setZero();
                        trans_curr.setZero();
                        for (size_t j = 0; j < ctrl_neighbors.size(); j++) {
                            const auto offset = NumUnknownsLocal*ctrl_neighbors[j].index;
                            linear_curr.noalias() += ctrl_neighbors[j].value*tforms_vec.template segment<Dim*Dim>(offset);
                            trans_curr.noalias() += ctrl_neighbors[j].value*tforms_vec.template segment<Dim>(offset + Dim*Dim);
                        }
                        if (total_weight[corr.indexInSecond] != (ScalarT)0.0) {
                            weight = (ScalarT)(1.0)/total_weight[corr.indexInSecond];
                            linear_curr *= weight;
                            trans_curr *= weight;

                            corr_weight_sqrt = point_to_point_weight_sqrt*std::sqrt(point_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value));
                            corr_weight_nrm = corr_weight_sqrt/total_weight[corr.indexInSecond];
                        } else {
                            corr_weight_sqrt = (ScalarT)0.0;
                            corr_weight_nrm = (ScalarT)0.0;
                        }

                        const auto d = dst_p.col(corr.indexInFirst);
                        const auto s = src_p.col(corr.indexInSecond);

                        Eigen::Matrix<ScalarT,Dim,1> s_t = Eigen::Map<Eigen::Matrix<ScalarT,Dim,Dim,Eigen::RowMajor>>(linear_curr.data(), Dim, Dim)*s + trans_curr;

                        eq_ind = Dim*i;

                        for (size_t j = 0; j < ctrl_neighbors.size(); j++) {
                            const auto offset = NumUnknownsLocal*ctrl_neighbors[j].index;
                            weight = corr_weight_nrm*ctrl_neighbors[j].value;

                            for (size_t eq = 0; eq < Dim; eq++) {
                                nz_ind = outer_ptr[eq_ind + eq] + NumNonZerosPointToPoint*j;
                                for (size_t nz = 0; nz < Dim; nz++) {
                                    values[nz_ind] = weight*s_t[nz];
                                    inner_ind[nz_ind++] = offset + Dim*eq + nz;
                                }
                                values[nz_ind] = weight;
                                inner_ind[nz_ind++] = offset + Dim*Dim + eq;
                            }
                        }

                        b.template segment<Dim>(eq_ind) = corr_weight_sqrt*(d - s_t);
                    }
                }

                if (has_point_to_plane_terms) {
#pragma omp for nowait
                    for (size_t i = 0; i < point_to_plane_correspondences.size(); i++) {
                        const auto& corr = point_to_plane_correspondences[i];
                        const auto& ctrl_neighbors = src_to_ctrl_sorted[corr.indexInSecond];

                        // Compute weighted influence from control nodes
                        linear_curr.setZero();
                        trans_curr.setZero();
                        for (size_t j = 0; j < ctrl_neighbors.size(); j++) {
                            const auto offset = NumUnknownsLocal*ctrl_neighbors[j].index;
                            linear_curr.noalias() += ctrl_neighbors[j].value*tforms_vec.template segment<Dim*Dim>(offset);
                            trans_curr.noalias() += ctrl_neighbors[j].value*tforms_vec.template segment<Dim>(offset + Dim*Dim);
                        }
                        if (total_weight[corr.indexInSecond] != (ScalarT)0.0) {
                            weight = (ScalarT)(1.0)/total_weight[corr.indexInSecond];
                            linear_curr *= weight;
                            trans_curr *= weight;

                            corr_weight_sqrt = point_to_plane_weight_sqrt*std::sqrt(plane_corr_evaluator(corr.indexInFirst, corr.indexInSecond, corr.value));
                            corr_weight_nrm = corr_weight_sqrt/total_weight[corr.indexInSecond];
                        } else {
                            corr_weight_sqrt = (ScalarT)0.0;
                            corr_weight_nrm = (ScalarT)0.0;
                        }

                        const auto d = dst_p.col(corr.indexInFirst);
                        const auto n = dst_n.col(corr.indexInFirst);
                        const auto s = src_p.col(corr.indexInSecond);

                        Eigen::Matrix<ScalarT,Dim,1> s_t = Eigen::Map<Eigen::Matrix<ScalarT,Dim,Dim,Eigen::RowMajor>>(linear_curr.data(), Dim, Dim)*s + trans_curr;

                        eq_ind = num_point_to_point_equations + i;

                        for (size_t j = 0; j < ctrl_neighbors.size(); j++) {
                            const auto offset = NumUnknownsLocal*ctrl_neighbors[j].index;
                            weight = corr_weight_nrm*ctrl_neighbors[j].value;

                            nz_ind = outer_ptr[eq_ind] + NumUnknownsLocal*j;

                            for (size_t block = 0; block < Dim; block++) {
                                for (size_t curr = 0; curr < Dim; curr++) {
                                    values[nz_ind] = weight*n[block]*s_t[curr];
                                    inner_ind[nz_ind++] = offset + Dim*block + curr;
                                }
                            }
                            for (size_t curr = 0; curr < Dim; curr++) {
                                values[nz_ind] = weight*n[curr];
                                inner_ind[nz_ind++] = offset + Dim*Dim + curr;
                            }
                        }

                        b[eq_ind] = (n.dot(d - s_t))*corr_weight_sqrt;
                    }
                }

                // Regularization
#pragma omp for nowait
                for (size_t i = 0; i < regularization_neighborhoods.size(); i++) {
                    eq_ind = num_data_term_equations + reg_eq_ind[i];
                    nz_ind = outer_ptr[num_data_term_equations] + 2*reg_eq_ind[i];
                    const auto& neighbors = regularization_neighborhoods[i];

                    for (size_t j = 1; j < neighbors.size(); j++) {
                        auto s_offset = NumUnknownsLocal*neighbors[0].index;
                        auto n_offset = NumUnknownsLocal*neighbors[j].index;
                        weight = regularization_weight_sqrt*std::sqrt(reg_evaluator(neighbors[0].index, neighbors[j].index, neighbors[j].value));

                        if (n_offset < s_offset) std::swap(s_offset, n_offset);

                        for (size_t eq = 0; eq < NumUnknownsLocal; eq++) {
                            diff = tforms_vec[s_offset + eq] - tforms_vec[n_offset + eq];
                            d_sqrt_huber_loss = weight*internal::sqrtHuberLossDerivative<ScalarT>(diff, huber_boundary);
                            values[nz_ind] = d_sqrt_huber_loss;
                            inner_ind[nz_ind++] = s_offset + eq;
                            values[nz_ind] = -d_sqrt_huber_loss;
                            inner_ind[nz_ind++] = n_offset + eq;
                            b[eq_ind++] = -weight*internal::sqrtHuberLoss<ScalarT>(diff, huber_boundary);
                        }
                    }
                }
            }

            // Solve linear system using CG
            AtA = At*At.transpose();
            Atb.noalias() = At*b;

            if (iter == 0) solver.analyzePattern(AtA);
            solver.factorize(AtA);
            delta = solver.solve(Atb);
            tforms_vec += delta;

            iter++;

            // Check for convergence
            max_delta_sq = (ScalarT)0.0;
#pragma omp parallel for private (curr_delta_sq) reduction (max: max_delta_sq)
            for (size_t i = 0; i < num_ctrl_points; i++) {
                curr_delta_sq = delta.template segment<NumUnknownsLocal>(NumUnknownsLocal*i).squaredNorm();
                if (curr_delta_sq > max_delta_sq) max_delta_sq = curr_delta_sq;
            }

            if (max_delta_sq < gn_conv_tol_sq) {
                has_converged = true;
                break;
            }
        }

        // Convert to output format
        transforms.resize(num_ctrl_points);
#pragma omp parallel for
        for (size_t i = 0; i < transforms.size(); i++) {
            transforms[i].linear().noalias() = Eigen::Map<Eigen::Matrix<ScalarT,Dim,Dim,Eigen::RowMajor>>(tforms_vec.data() + i*NumUnknownsLocal, Dim, Dim);
            transforms[i].translation() = tforms_vec.template segment<Dim>(NumUnknownsLocal*i + Dim*Dim);
        }

        return has_converged;
    }
}
