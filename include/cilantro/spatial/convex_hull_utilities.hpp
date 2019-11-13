#pragma once

#include <cilantro/3rd_party/libqhullcpp/Qhull.h>
#include <cilantro/3rd_party/libqhullcpp/QhullFacetSet.h>
#include <cilantro/3rd_party/libqhullcpp/QhullFacetList.h>
#include <cilantro/3rd_party/libqhullcpp/QhullVertexSet.h>
#include <cilantro/3rd_party/eigen_quadprog/eiquadprog.hpp>
#include <cilantro/core/principal_component_analysis.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim>
    bool checkLinearInequalityConstraintRedundancy(const Eigen::Ref<const HomogeneousVector<ScalarT,EigenDim>> &ineq_to_test,
                                                   const ConstHomogeneousVectorSetMatrixMap<ScalarT,EigenDim> &inequalities,
                                                   const Eigen::Ref<const Vector<ScalarT,EigenDim>> &feasible_point,
                                                   double dist_tol = std::numeric_limits<ScalarT>::epsilon())
    {
        const size_t dim = inequalities.rows() - 1;
        const size_t num_inequalities = inequalities.cols();
        if (num_inequalities == 0) return false;
        Eigen::MatrixXd ineq_data(inequalities.template cast<double>());
        Eigen::VectorXd ineq_test(ineq_to_test.template cast<double>());

        // Normalize input
        // Force unit length normals
        for (size_t i = 0; i < num_inequalities; i++) {
            ineq_data.col(i) /= ineq_data.col(i).head(dim).norm();
        }
        ineq_test /= ineq_test.head(dim).norm();

        // Center halfspaces around origin and then scale dimensions
        Eigen::VectorXd t_vec(-feasible_point.template cast<double>());
        ineq_data.row(dim) = ineq_data.row(dim) - t_vec.transpose()*ineq_data.topRows(dim);
        ineq_test(dim) = ineq_test(dim) - t_vec.dot(ineq_test.head(dim));
        double max_abs_dist = ineq_data.row(dim).cwiseAbs().maxCoeff();
        double scale = (max_abs_dist < dist_tol) ? 1.0 : 1.0/max_abs_dist;
        ineq_data.row(dim) *= scale;
        ineq_test(dim) *= scale;

        // Objective
        // 'Preconditioned' quadratic term
        const ScalarT tol_sq = dist_tol*dist_tol;
        Eigen::MatrixXd G(ineq_test*(ineq_test.transpose()));
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(G, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::VectorXd S(svd.singularValues());
        for (size_t i = 0; i < S.size(); i++) {
            if (S(i) < tol_sq) S(i) = tol_sq;
        }
        G = svd.matrixU()*(S.asDiagonal())*(svd.matrixV().transpose());

        // Linear term
        Eigen::VectorXd g0(-ineq_test);
        g0(dim) = 0.0;

        // Equality constraints
        Eigen::MatrixXd CE(Eigen::VectorXd::Zero(dim+1));
        CE(dim) = 1.0;
        Eigen::VectorXd ce0(1);
        ce0(0) = -1.0;

        // Inequality constraints
        Eigen::MatrixXd CI(-ineq_data);
        Eigen::VectorXd ci0(Eigen::VectorXd::Zero(num_inequalities));

        // Optimization
        Eigen::VectorXd x(dim+1);
        double val = solve_quadprog(G, g0, CE, ce0, CI, ci0, x);

//        Eigen::VectorXd sol((x.head(dim)/scale - t_vec));
//        std::cout << val << ", for: " << x.transpose() << " (" << sol.transpose() << ")" << std::endl;
//        Eigen::VectorXd ineq_test0(ineq_to_test.template cast<double>());
//        std::cout << "dist: " << (sol.dot(ineq_test0.head(dim)) + ineq_test0(dim)) << std::endl;

        if (std::isinf(val) || std::isnan(val) || !x.allFinite()) return false;

        return x.head(dim).dot(ineq_test.head(dim)) + ineq_test(dim) < -dist_tol;
    }

    template <typename ScalarT, ptrdiff_t EigenDim>
    bool findFeasiblePointInHalfspaceIntersection(const ConstHomogeneousVectorSetMatrixMap<ScalarT,EigenDim> &halfspaces,
                                                  Vector<ScalarT,EigenDim> &feasible_point,
                                                  double dist_tol = std::numeric_limits<ScalarT>::epsilon(),
                                                  bool force_strictly_interior = true)
    {
        const size_t dim = halfspaces.rows() - 1;
        const size_t num_halfspaces = halfspaces.cols();
        if (num_halfspaces == 0) {
            feasible_point.setZero(dim,1);
            return true;
        }
        Eigen::MatrixXd ineq_data(halfspaces.template cast<double>());

        // Normalize input
        // Force unit length normals
        for (size_t i = 0; i < num_halfspaces; i++) {
            ineq_data.col(i) /= ineq_data.col(i).head(dim).norm();
        }

        // Center halfspaces around origin and then scale dimensions
        Eigen::Matrix<double,EigenDim,1> t_vec((ineq_data.topRows(dim).array().rowwise()*ineq_data.row(dim).array().abs()).rowwise().mean());
        ineq_data.row(dim) = ineq_data.row(dim) - t_vec.transpose()*ineq_data.topRows(dim);
        const double max_abs_dist = ineq_data.row(dim).cwiseAbs().maxCoeff();
        const double scale = (max_abs_dist < dist_tol) ? 1.0 : 1.0/max_abs_dist;
        ineq_data.row(dim) *= scale;

        // Objective
        // 'Preconditioned' quadratic term
        const ScalarT tol_sq = dist_tol*dist_tol;
        Eigen::MatrixXd G(dim+2,dim+2);
        G.topLeftCorner(dim+1,dim+1) = ineq_data*(ineq_data.transpose());
        G.row(dim+1).setZero();
        G.col(dim+1).setZero();
        //G(dim+1,dim+1) = 1.0;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(G, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::VectorXd S(svd.singularValues());
        for (size_t i = 0; i < S.size(); i++) {
            if (S(i) < tol_sq) S(i) = 1.0;
        }
        //G = svd.matrixU()*(S.asDiagonal())*(svd.matrixV().transpose());
        G = dist_tol*svd.matrixU()*(S.asDiagonal())*(svd.matrixV().transpose());

        // Linear term
        Eigen::VectorXd g0(Eigen::VectorXd::Zero(dim+2));
        g0(dim+1) = -1.0;

        // Equality constraints
        Eigen::MatrixXd CE(Eigen::VectorXd::Zero(dim+2));
        CE(dim) = 1.0;
        Eigen::VectorXd ce0(1);
        ce0(0) = -1.0;

        // Inequality constraints
        Eigen::MatrixXd CI(dim+2,num_halfspaces+1);
        CI.topLeftCorner(dim+1,num_halfspaces) = -ineq_data;
        CI.row(dim+1).setConstant(-1.0);
        CI.col(num_halfspaces).setZero();
        CI(dim+1,num_halfspaces) = 1.0;
        Eigen::VectorXd ci0(Eigen::VectorXd::Zero(num_halfspaces+1));

        // Optimization
        Eigen::VectorXd x(dim+2);
        double val = solve_quadprog(G, g0, CE, ce0, CI, ci0, x);
        Eigen::Matrix<double,EigenDim,1> fp(x.head(dim));
        feasible_point = (fp/scale - t_vec).template cast<ScalarT>();

        //std::cout << val << ", for: " << x.transpose() << "   (" << feasible_point.transpose() << ")" << std::endl;

        if (std::isinf(val) || std::isnan(val) || !x.allFinite()) {
            feasible_point.setConstant(dim, 1, std::numeric_limits<ScalarT>::quiet_NaN());
            return false;
        }

        // Useful in case of unbounded intersections
        if (force_strictly_interior && x(dim+1) < dist_tol) {
            size_t num_additional = 0;
            std::vector<size_t> tight_ind(num_halfspaces);
            for (size_t i = 0; i < num_halfspaces; i++) {
                if (std::abs(ineq_data.col(i).dot(x.head(dim+1))) < dist_tol) {
                    tight_ind[num_additional++] = i;
                }
            }
            tight_ind.resize(num_additional);

            if (num_additional > 0) {
                const double offset = (double)(num_halfspaces - 1);
                HomogeneousVectorSet<double,EigenDim> halfspaces_tight(dim+1,num_halfspaces+num_additional);
                halfspaces_tight.leftCols(num_halfspaces) = ineq_data;
                num_additional = num_halfspaces;
                for (size_t i = 0; i < tight_ind.size(); i++) {
                    halfspaces_tight.col(num_additional) = -ineq_data.col(tight_ind[i]);
                    halfspaces_tight(dim,num_additional) -= offset;
                    num_additional++;
                }

                bool res = findFeasiblePointInHalfspaceIntersection<double,EigenDim>(halfspaces_tight, fp, dist_tol, false);
                if (!res) {
                    feasible_point.setConstant(dim, 1, std::numeric_limits<ScalarT>::quiet_NaN());
                    return false;
                }

                feasible_point = (fp/scale - t_vec).template cast<ScalarT>();
                res = ((feasible_point.transpose()*halfspaces.topRows(dim) + halfspaces.row(dim)).array() <= -dist_tol).all();
                if (!res) {
                    feasible_point.setConstant(dim, 1, std::numeric_limits<ScalarT>::quiet_NaN());
                    return false;
                }

                return true;
            }
        }

        return true;
    }

    template <typename ScalarT, ptrdiff_t EigenDim>
    bool evaluateHalfspaceIntersection(const ConstHomogeneousVectorSetMatrixMap<ScalarT,EigenDim> &halfspaces,
                                       HomogeneousVectorSet<ScalarT,EigenDim> &facet_halfspaces,
                                       VectorSet<ScalarT,EigenDim> &polytope_vertices,
                                       Vector<ScalarT,EigenDim> &interior_point,
                                       bool &is_bounded,
                                       double dist_tol = std::numeric_limits<ScalarT>::epsilon(),
                                       double merge_tol = 0.0)
    {
        const size_t dim = halfspaces.rows() - 1;
        size_t num_halfspaces = halfspaces.cols();

        // Set up variables
        HomogeneousVectorSet<double,EigenDim> hs_coeffs(halfspaces.template cast<double>());
        Eigen::Matrix<double,EigenDim,1> feasible_point(dim);

        // Force unit length normals
        for (size_t i = 0; i < num_halfspaces; i++) {
            hs_coeffs.col(i) /= hs_coeffs.col(i).head(dim).norm();
        }

        bool is_empty = !findFeasiblePointInHalfspaceIntersection<double,EigenDim>(hs_coeffs, feasible_point, dist_tol, true);
        interior_point = feasible_point.template cast<ScalarT>();

        // Check if intersection is empty
        if (is_empty) {
            facet_halfspaces.resize(dim+1, 2);
            facet_halfspaces.setZero();
            facet_halfspaces(0,0) = 1.0;
            facet_halfspaces(dim,0) = 1.0;
            facet_halfspaces(0,1) = -1.0;
            facet_halfspaces(dim,1) = 1.0;
            polytope_vertices.resize(dim, 0);
            is_bounded = true;

            return false;
        }

        Eigen::FullPivLU<Eigen::Matrix<double,EigenDim,Eigen::Dynamic>> lu(hs_coeffs.topRows(dim));
        lu.setThreshold(dist_tol);

        //std::cout << "rank: " << lu.rank() << std::endl;

        if (lu.rank() < dim) {
            Eigen::Map<HomogeneousVectorSet<double,EigenDim>> first_cols(hs_coeffs.data(), dim+1, num_halfspaces-1);

            facet_halfspaces.resize(dim+1,num_halfspaces);
            size_t nr = 0;
            for (size_t i = 0; i < num_halfspaces; i++) {
                hs_coeffs.col(i).swap(hs_coeffs.col(num_halfspaces-1));
                if (!checkLinearInequalityConstraintRedundancy<double,EigenDim>(hs_coeffs.col(num_halfspaces-1), first_cols, feasible_point, dist_tol)) {
                    facet_halfspaces.col(nr++) = hs_coeffs.col(num_halfspaces-1).template cast<ScalarT>();
                }
                hs_coeffs.col(i).swap(hs_coeffs.col(num_halfspaces-1));
            }
            facet_halfspaces.conservativeResize(Eigen::NoChange,nr);
            polytope_vertices.resize(dim, 0);
            is_bounded = false;

            return true;
        }

        is_bounded = true;

        // 'Precondition' qhull input...
//    if (std::abs(hs_coeffs.row(dim).maxCoeff() - hs_coeffs.row(dim).minCoeff()) < dist_tol) {
        hs_coeffs.conservativeResize(Eigen::NoChange, 2*num_halfspaces);
        hs_coeffs.rightCols(num_halfspaces) = hs_coeffs.leftCols(num_halfspaces);
        hs_coeffs.block(dim,num_halfspaces,1,num_halfspaces).array() -= 1.0;
        num_halfspaces *= 2;
//    }

        // Run qhull in halfspace mode
        std::vector<double> fpv(dim);
        Eigen::Matrix<double,EigenDim,1>::Map(fpv.data(),dim,1) = feasible_point;

        orgQhull::Qhull qh;
        qh.qh()->HALFspace = True;
        qh.qh()->PRINTprecision = False;
        //qh.qh()->JOGGLEmax = 0.0;
        qh.qh()->TRIangulate = False;
        qh.qh()->premerge_centrum = merge_tol;
        qh.setFeasiblePoint(orgQhull::Coordinates(fpv));
        qh.runQhull("", dim+1, num_halfspaces, hs_coeffs.data(), "");

        // Get polytope vertices from dual hull facets
        orgQhull::QhullFacetList facets = qh.facetList();
        size_t num_dual_facets = facets.size();

        VectorSet<double,EigenDim> vertices(dim, num_dual_facets);
        polytope_vertices.resize(dim, num_dual_facets);

        size_t ind = 0;
        for (auto fi = facets.begin(); fi != facets.end(); ++fi) {
            if (fi->hyperplane().offset() < 0.0) {
                Eigen::Matrix<double,EigenDim,1> normal(dim);
                size_t i = 0;
                for (auto hpi = fi->hyperplane().begin(); hpi != fi->hyperplane().end(); ++hpi) {
                    normal(i++) = *hpi;
                }
                vertices.col(ind) = feasible_point - normal/fi->hyperplane().offset();
                polytope_vertices.col(ind) = vertices.col(ind).template cast<ScalarT>();
                ind++;
            } else {
                is_bounded = false;
            }
        }
        vertices.conservativeResize(Eigen::NoChange, ind);
        polytope_vertices.conservativeResize(Eigen::NoChange, ind);

        // Get facet hyperplanes from dual hull vertices
        facet_halfspaces.resize(dim+1, qh.vertexList().size());
        ind = 0;
        for (auto vi = qh.vertexList().begin(); vi != qh.vertexList().end(); ++vi) {
            const HomogeneousVector<double,EigenDim>& hs(hs_coeffs.col(vi->point().id()));
            if (((hs.head(dim).transpose()*vertices).array() + hs(dim)).cwiseAbs().minCoeff() < dist_tol) {
                facet_halfspaces.col(ind++) = hs.template cast<ScalarT>();
            }
        }
        facet_halfspaces.conservativeResize(Eigen::NoChange, ind);

        return true;
    }

    template <typename ScalarT, ptrdiff_t EigenDim>
    bool halfspaceIntersectionFromVertices(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &vertices,
                                           VectorSet<ScalarT,EigenDim> &polytope_vertices,
                                           HomogeneousVectorSet<ScalarT,EigenDim> &facet_halfspaces,
                                           double &area, double &volume,
                                           bool require_full_dimension = true,
                                           double merge_tol = 0.0)
    {
        const size_t dim = vertices.rows();
        const size_t num_points = vertices.cols();

        if (num_points == 0) {
            facet_halfspaces.resize(dim+1, 2);
            facet_halfspaces.setZero();
            facet_halfspaces(0,0) = 1.0;
            facet_halfspaces(dim,0) = 1.0;
            facet_halfspaces(0,1) = -1.0;
            facet_halfspaces(dim,1) = 1.0;
            polytope_vertices.resize(dim, 0);
            area = 0.0;
            volume = 0.0;
            return false;
        }

        // Avoid unnecessary copy/cast if input data is double
        Eigen::Matrix<double,EigenDim,Eigen::Dynamic> data_holder(dim,0);
        Eigen::Map<Eigen::Matrix<double,EigenDim,Eigen::Dynamic>> vert_data(NULL, dim, 0);
        if (std::is_same<ScalarT, double>::value) {
            new (&vert_data) Eigen::Map<Eigen::Matrix<double,EigenDim,Eigen::Dynamic>>((double *)vertices.data(), dim, num_points);
        } else {
            data_holder = vertices.template cast<double>();
            new (&vert_data) Eigen::Map<Eigen::Matrix<double,EigenDim,Eigen::Dynamic>>(data_holder.data(), dim, num_points);
        }

        Eigen::Matrix<double,EigenDim,1> mu(vert_data.rowwise().mean());
        const size_t true_dim = Eigen::FullPivLU<Eigen::Matrix<double,EigenDim,Eigen::Dynamic>>(vert_data.colwise() - mu).rank();

        //std::cout << "TRUE DIMENSION: " << true_dim << std::endl << std::endl;

        if (require_full_dimension && true_dim < dim) {
            facet_halfspaces.resize(dim+1, 2);
            facet_halfspaces.setZero();
            facet_halfspaces(0,0) = 1.0;
            facet_halfspaces(dim,0) = 1.0;
            facet_halfspaces(0,1) = -1.0;
            facet_halfspaces(dim,1) = 1.0;
            polytope_vertices.resize(dim, 0);
            area = 0.0;
            volume = 0.0;
            return false;
        }

        if (true_dim < dim) {
            PrincipalComponentAnalysis<double,EigenDim> pca(vert_data);
            const Eigen::Matrix<double,EigenDim,1>& t_vec(pca.getDataMean());
            const Eigen::Matrix<double,EigenDim,EigenDim>& rot_mat(pca.getEigenVectors());

            Eigen::MatrixXd proj_vert((rot_mat.transpose()*(vert_data.colwise() - t_vec)).topRows(true_dim));

            HomogeneousVectorSet<double,EigenDim> facet_halfspaces_d(dim+1,0);
            size_t hs_ind = 0;

            // Populate polytope vertices amd add constraints for first true_dim dimensions
            if (true_dim > 1) {
                // Get vertices and constraints from qhull
                orgQhull::Qhull qh;
                qh.qh()->TRIangulate = False;
                qh.qh()->premerge_centrum = merge_tol;
                qh.qh()->PRINTprecision = False;
                qh.runQhull("", true_dim, num_points, proj_vert.data(), "");
                //qh.defineVertexNeighborFacets();
                orgQhull::QhullFacetList qh_facets = qh.facetList();

                // Populate polytope vertices
                size_t vert_ind = 0;
                polytope_vertices.resize(dim, qh.vertexCount());
                for (auto vi = qh.vertexList().begin(); vi != qh.vertexList().end(); ++vi) {
                    polytope_vertices.col(vert_ind++) = vertices.col(vi->point().id());
                }

                // Populate facet halfspaces
                facet_halfspaces_d.setZero(dim+1, qh.facetCount() + 2*(dim - true_dim));
                for (auto fi = qh_facets.begin(); fi != qh_facets.end(); ++fi) {
                    size_t i = 0;
                    for (auto hpi = fi->hyperplane().begin(); hpi != fi->hyperplane().end(); ++hpi) {
                        facet_halfspaces_d(i++,hs_ind) = *hpi;
                    }
                    facet_halfspaces_d(dim,hs_ind) = fi->hyperplane().offset();
                    hs_ind++;
                }

                area = (dim == true_dim+1) ? qh.volume() : 0.0;
                volume = 0.0;
            } else if (true_dim == 0) {
                // Handle special case (single point)
                polytope_vertices.resize(dim,1);
                polytope_vertices.col(0) = vertices.col(0);
                facet_halfspaces_d.setZero(dim+1,2*dim);
                area = 0.0;
                volume = 0.0;
            } else if (true_dim == 1) {
                // Handle special case (1D line, not handled by qhull)
                size_t ind_min, ind_max;
                const double min_val = proj_vert.row(0).minCoeff(&ind_min);
                const double max_val = proj_vert.row(0).maxCoeff(&ind_max);

                polytope_vertices.resize(dim,2);
                polytope_vertices.col(0) = vertices.col(ind_min);
                polytope_vertices.col(1) = vertices.col(ind_max);

                facet_halfspaces_d.setZero(dim+1,2*dim);
                facet_halfspaces_d(0,0) = -1.0;
                facet_halfspaces_d(dim,0) = min_val;
                facet_halfspaces_d(0,1) = 1.0;
                facet_halfspaces_d(dim,1) = -max_val;
                hs_ind = 2;

                area = (dim == 2) ? (max_val-min_val) : 0.0;
                volume = 0.0;
            }

            // Add (equality) constraints for remaining dimensions
            for (size_t curr_dim = true_dim; curr_dim < dim; curr_dim++) {
                facet_halfspaces_d(curr_dim,hs_ind++) = -1.0;
                facet_halfspaces_d(curr_dim,hs_ind++) = 1.0;
            }

            // Project back into output arguments
            typename std::conditional<EigenDim == Eigen::Dynamic, Eigen::Matrix<double,EigenDim,EigenDim>, Eigen::Matrix<double,EigenDim+1,EigenDim+1>>::type tform;
            tform.setZero(dim+1,dim+1);
            tform.block(0,0,dim,dim) = rot_mat;
            tform.block(0,dim,dim,1) = t_vec;
            tform(dim,dim) = 1.0;

            facet_halfspaces = (tform.inverse().transpose()*facet_halfspaces_d).template cast<ScalarT>();

            return true;
        }

        orgQhull::Qhull qh;
        qh.qh()->TRIangulate = False;
        qh.qh()->premerge_centrum = merge_tol;
        qh.qh()->PRINTprecision = False;
        qh.runQhull("", dim, num_points, vert_data.data(), "");
        //qh.defineVertexNeighborFacets();
        orgQhull::QhullFacetList qh_facets = qh.facetList();

        // Populate polytope vertices
        size_t k = 0;
        polytope_vertices.resize(dim, qh.vertexCount());
        for (auto vi = qh.vertexList().begin(); vi != qh.vertexList().end(); ++vi) {
            size_t i = 0;
            for (auto ci = vi->point().begin(); ci != vi->point().end(); ++ci) {
                polytope_vertices(i++,k) = (ScalarT)(*ci);
            }
            k++;
        }

        // Populate facet halfspaces
        k = 0;
        facet_halfspaces.resize(dim+1, qh.facetCount());
        for (auto fi = qh_facets.begin(); fi != qh_facets.end(); ++fi) {
            size_t i = 0;
            for (auto hpi = fi->hyperplane().begin(); hpi != fi->hyperplane().end(); ++hpi) {
                facet_halfspaces(i++,k) = (ScalarT)(*hpi);
            }
            facet_halfspaces(dim,k++) = (ScalarT)(fi->hyperplane().offset());
        }

        area = qh.area();
        volume = qh.volume();

        return true;
    }

    template <typename ScalarT, ptrdiff_t EigenDim>
    void computeConvexHullAreaAndVolume(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &vertices,
                                        double &area, double &volume,
                                        double merge_tol = 0.0)
    {
        const size_t dim = vertices.rows();
        const size_t num_points = vertices.cols();

        if (num_points == 0) {
            area = 0.0;
            volume = 0.0;
            return;
        }

        // Avoid unnecessary copy/cast if input data is double
        Eigen::Matrix<double,EigenDim,Eigen::Dynamic> data_holder(dim,0);
        Eigen::Map<Eigen::Matrix<double,EigenDim,Eigen::Dynamic>> vert_data(NULL, dim, 0);
        if (std::is_same<ScalarT, double>::value) {
            new (&vert_data) Eigen::Map<Eigen::Matrix<double,EigenDim,Eigen::Dynamic>>((double *)vertices.data(), dim, num_points);
        } else {
            data_holder = vertices.template cast<double>();
            new (&vert_data) Eigen::Map<Eigen::Matrix<double,EigenDim,Eigen::Dynamic>>(data_holder.data(), dim, num_points);
        }

        Eigen::Matrix<double,EigenDim,1> mu(vert_data.rowwise().mean());
        const size_t true_dim = Eigen::FullPivLU<Eigen::Matrix<double,EigenDim,Eigen::Dynamic>>(vert_data.colwise() - mu).rank();

        //std::cout << "TRUE DIMENSION: " << true_dim << std::endl << std::endl;

        if (true_dim < dim) {
            PrincipalComponentAnalysis<double,EigenDim> pca(vert_data);
            const Eigen::Matrix<double,EigenDim,1>& t_vec(pca.getDataMean());
            const Eigen::Matrix<double,EigenDim,EigenDim>& rot_mat(pca.getEigenVectors());

            Eigen::MatrixXd proj_vert((rot_mat.transpose()*(vert_data.colwise() - t_vec)).topRows(true_dim));

            // Populate polytope vertices amd add constraints for first true_dim dimensions
            if (true_dim > 1) {
                // Get vertices and constraints from qhull
                orgQhull::Qhull qh;
                qh.qh()->TRIangulate = False;
                qh.qh()->premerge_centrum = merge_tol;
                qh.qh()->PRINTprecision = False;
                qh.runQhull("", true_dim, num_points, proj_vert.data(), "");
                //qh.defineVertexNeighborFacets();

                area = (dim == true_dim+1) ? qh.volume() : 0.0;
                volume = 0.0;
            } else if (true_dim == 0) {
                // Handle special case (single point)
                area = 0.0;
                volume = 0.0;
            } else if (true_dim == 1) {
                // Handle special case (1D line, not handled by qhull)
                size_t ind_min, ind_max;
                const double min_val = proj_vert.row(0).minCoeff(&ind_min);
                const double max_val = proj_vert.row(0).maxCoeff(&ind_max);
                area = (dim == 2) ? (max_val-min_val) : 0.0;
                volume = 0.0;
            }

            return;
        }

        orgQhull::Qhull qh;
        qh.qh()->TRIangulate = False;
        qh.qh()->premerge_centrum = merge_tol;
        qh.qh()->PRINTprecision = False;
        qh.runQhull("", dim, num_points, vert_data.data(), "");
        //qh.defineVertexNeighborFacets();

        area = qh.area();
        volume = qh.volume();
    }

    template <typename ScalarT, ptrdiff_t EigenDim, typename IndexT = size_t>
    bool convexHullFromPoints(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                              VectorSet<ScalarT,EigenDim> &hull_points,
                              HomogeneousVectorSet<ScalarT,EigenDim> &halfspaces,
                              std::vector<std::vector<IndexT>> &facets,
                              std::vector<std::vector<IndexT>> &point_neighbor_facets,
                              std::vector<std::vector<IndexT>> &facet_neighbor_facets,
                              std::vector<IndexT> &hull_point_indices,
                              double &area, double &volume,
                              bool simplicial_facets = true,
                              double merge_tol = 0.0)
    {
        const size_t dim = points.rows();
        const size_t num_points = points.cols();

        if (num_points < dim+1) {
            halfspaces.resize(dim+1, 2);
            halfspaces.setZero();
            halfspaces(0,0) = 1.0;
            halfspaces(dim,0) = 1.0;
            halfspaces(0,1) = -1.0;
            halfspaces(dim,1) = 1.0;
            hull_points.resize(dim, 0);
            facets.clear();
            point_neighbor_facets.clear();
            facet_neighbor_facets.clear();
            hull_point_indices.clear();
            area = 0.0;
            volume = 0.0;
            return false;
        }

        // Avoid unnecessary copy/cast if input data is double
        Eigen::Matrix<double,EigenDim,Eigen::Dynamic> data_holder(dim,0);
        Eigen::Map<Eigen::Matrix<double,EigenDim,Eigen::Dynamic>> vert_data(NULL, dim, 0);
        if (std::is_same<ScalarT, double>::value) {
            new (&vert_data) Eigen::Map<Eigen::Matrix<double,EigenDim,Eigen::Dynamic>>((double *)points.data(), dim, num_points);
        } else {
            data_holder = points.template cast<double>();
            new (&vert_data) Eigen::Map<Eigen::Matrix<double,EigenDim,Eigen::Dynamic>>(data_holder.data(), dim, num_points);
        }

        Eigen::Matrix<double,EigenDim,1> mu(vert_data.rowwise().mean());
        if (Eigen::FullPivLU<Eigen::Matrix<double,EigenDim,Eigen::Dynamic>>(vert_data.colwise() - mu).rank() < dim) {
            halfspaces.resize(dim+1, 2);
            halfspaces.setZero();
            halfspaces(0,0) = 1.0;
            halfspaces(dim,0) = 1.0;
            halfspaces(0,1) = -1.0;
            halfspaces(dim,1) = 1.0;
            hull_points.resize(dim, 0);
            facets.clear();
            point_neighbor_facets.clear();
            facet_neighbor_facets.clear();
            hull_point_indices.clear();
            area = 0.0;
            volume = 0.0;
            return false;
        }

        orgQhull::Qhull qh;
        if (simplicial_facets) qh.qh()->TRIangulate = True;
        qh.qh()->premerge_centrum = merge_tol;
        qh.qh()->PRINTprecision = False;
        qh.runQhull("", dim, num_points, vert_data.data(), "");
        qh.defineVertexNeighborFacets();
        orgQhull::QhullFacetList qh_facets = qh.facetList();

        // Establish mapping between hull vertex ids and hull points indices
        IndexT max_id = 0;
        for (auto vi = qh.vertexList().begin(); vi != qh.vertexList().end(); ++vi)
            if (max_id < vi->id()) max_id = vi->id();
        std::vector<IndexT> vid_to_ptidx(max_id + 1);
        IndexT k = 0;
        for (auto vi = qh.vertexList().begin(); vi != qh.vertexList().end(); ++vi)
            vid_to_ptidx[vi->id()] = k++;

        // Establish mapping between face ids and face indices
        max_id = 0;
        for (auto fi = qh_facets.begin(); fi != qh_facets.end(); ++fi)
            if (max_id < fi->id()) max_id = fi->id();
        std::vector<IndexT> fid_to_fidx(max_id + 1);
        k = 0;
        for (auto fi = qh_facets.begin(); fi != qh_facets.end(); ++fi)
            fid_to_fidx[fi->id()] = k++;

        // Populate hull points and their indices in the input cloud
        k = 0;
        hull_points.resize(dim, qh.vertexCount());
        point_neighbor_facets.resize(qh.vertexCount());
        hull_point_indices.resize(qh.vertexCount());
        for (auto vi = qh.vertexList().begin(); vi != qh.vertexList().end(); ++vi) {
            IndexT i = 0;
            for (auto ci = vi->point().begin(); ci != vi->point().end(); ++ci) {
                hull_points(i++,k) = (ScalarT)(*ci);
            }

            i = 0;
            point_neighbor_facets[k].resize(vi->neighborFacets().size());
            for (auto fi = vi->neighborFacets().begin(); fi != vi->neighborFacets().end(); ++fi) {
                point_neighbor_facets[k][i++] = fid_to_fidx[(*fi).id()];
            }

            hull_point_indices[k] = vi->point().id();
            k++;
        }

        // Populate halfspaces and faces (indices in the hull cloud)
        k = 0;
        halfspaces.resize(dim+1, qh.facetCount());
        facets.resize(qh_facets.size());
        facet_neighbor_facets.resize(qh_facets.size());
        for (auto fi = qh_facets.begin(); fi != qh_facets.end(); ++fi) {
            IndexT i = 0;
            for (auto hpi = fi->hyperplane().begin(); hpi != fi->hyperplane().end(); ++hpi) {
                halfspaces(i++,k) = (ScalarT)(*hpi);
            }
            halfspaces(dim,k) = (ScalarT)(fi->hyperplane().offset());

            facets[k].resize(fi->vertices().size());
            if (fi->isTopOrient()) {
                i = facets[k].size() - 1;
                for (auto vi = fi->vertices().begin(); vi != fi->vertices().end(); ++vi) {
                    facets[k][i--] = vid_to_ptidx[(*vi).id()];
                }
            } else {
                i = 0;
                for (auto vi = fi->vertices().begin(); vi != fi->vertices().end(); ++vi) {
                    facets[k][i++] = vid_to_ptidx[(*vi).id()];
                }
            }

            i = 0;
            facet_neighbor_facets[k].resize(fi->neighborFacets().size());
            for (auto nfi = fi->neighborFacets().begin(); nfi != fi->neighborFacets().end(); ++nfi) {
                facet_neighbor_facets[k][i++] = fid_to_fidx[(*nfi).id()];
            }

            k++;
        }

        area = qh.area();
        volume = qh.volume();

        return true;
    }
}
