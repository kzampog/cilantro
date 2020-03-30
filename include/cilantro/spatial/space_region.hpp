#pragma once

#include <cilantro/spatial/convex_polytope.hpp>

namespace cilantro {
    template <typename ScalarT, ptrdiff_t EigenDim, typename IndexT = size_t>
    class SpaceRegion {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ScalarT Scalar;
        typedef IndexT Index;

        enum { Dimension = EigenDim };

        typedef typename std::conditional<EigenDim != Eigen::Dynamic && sizeof(Eigen::Matrix<ptrdiff_t,EigenDim,1>) % 16 == 0,
                std::vector<ConvexPolytope<ScalarT,EigenDim,IndexT>,Eigen::aligned_allocator<ConvexPolytope<ScalarT,EigenDim,IndexT>>>,
                std::vector<ConvexPolytope<ScalarT,EigenDim,IndexT>>>::type ConvexPolytopeVector;

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        SpaceRegion()
                : dim_(EigenDim)
        {}

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim == Eigen::Dynamic>::type>
        SpaceRegion(size_t dim = 2)
                : dim_(dim)
        {}

        SpaceRegion(const ConvexPolytopeVector &polytopes)
                : dim_((EigenDim != Eigen::Dynamic) ? EigenDim : ((polytopes.empty()) ? 2 : polytopes[0].getSpaceDimension())),
                  polytopes_(polytopes)
        {}

        SpaceRegion(const ConvexPolytope<ScalarT,EigenDim,IndexT> &polytope)
                : dim_(polytope.getSpaceDimension()), polytopes_(ConvexPolytopeVector(1, polytope))
        {}

        // Directly construct single-polytope space region
        template<class... PolytopeArgs>
        SpaceRegion(PolytopeArgs... args) {
            polytopes_.emplace_back(args...);
            dim_ = polytopes_.back().getSpaceDimension();
        }

        ~SpaceRegion() {}

        SpaceRegion unionWith(const SpaceRegion &sr) const {
            ConvexPolytopeVector res_polytopes(polytopes_);
            res_polytopes.insert(res_polytopes.end(), sr.polytopes_.begin(), sr.polytopes_.end());
            return SpaceRegion(std::move(res_polytopes));
        }

        SpaceRegion intersectionWith(const SpaceRegion &sr,
                                     bool compute_topology = false,
                                     bool simplicial_facets = false,
                                     double merge_tol = 0.0,
                                     double dist_tol = std::numeric_limits<ScalarT>::epsilon()) const
        {
            ConvexPolytopeVector res_polytopes;
            for (size_t i = 0; i < polytopes_.size(); i++) {
                for (size_t j = 0; j < sr.polytopes_.size(); j++) {
                    ConvexPolytope<ScalarT,EigenDim,IndexT> poly_tmp(polytopes_[i].intersectionWith(sr.polytopes_[j], compute_topology, simplicial_facets, merge_tol, dist_tol));
                    if (!poly_tmp.isEmpty()) {
                        res_polytopes.emplace_back(std::move(poly_tmp));
                    }
                }
            }
            return SpaceRegion(std::move(res_polytopes));
        }

        // Inefficient
        template <ptrdiff_t Dim = EigenDim>
        typename std::enable_if<Dim != Eigen::Dynamic, SpaceRegion>::type complement(bool compute_topology = false,
                                                                                     bool simplicial_facets = false,
                                                                                     double merge_tol = 0.0,
                                                                                     double dist_tol = std::numeric_limits<ScalarT>::epsilon()) const
        {
            std::vector<HomogeneousVectorSet<ScalarT,EigenDim>> tuples;
            tuples.emplace_back(EigenDim+1,0);
            for (size_t p = 0; p < polytopes_.size(); p++) {
                const HomogeneousVectorSet<ScalarT,EigenDim>& hs(polytopes_[p].getFacetHyperplanes());
                std::vector<HomogeneousVectorSet<ScalarT,EigenDim>> tuples_new;
                for (size_t t = 0; t < tuples.size(); t++) {
                    HomogeneousVectorSet<ScalarT,EigenDim> tup_curr(EigenDim+1, tuples[t].cols()+1);
                    tup_curr.leftCols(tuples[t].cols()) = tuples[t];
                    for (size_t h = 0; h < hs.cols(); h++) {
                        tup_curr.col(tup_curr.cols()-1) = -hs.col(h);
                        tuples_new.emplace_back(tup_curr);
                    }
                }
                tuples = std::move(tuples_new);
            }

            ConvexPolytopeVector res_polytopes;
            for (size_t t = 0; t < tuples.size(); t++) {
                ConvexPolytope<ScalarT,EigenDim,IndexT> poly_tmp(tuples[t], compute_topology, simplicial_facets, merge_tol, dist_tol);
                if (!poly_tmp.isEmpty()) {
                    res_polytopes.emplace_back(std::move(poly_tmp));
                }
            }
            return SpaceRegion(std::move(res_polytopes));
        }

        // Inefficient
        template <ptrdiff_t Dim = EigenDim>
        typename std::enable_if<Dim == Eigen::Dynamic, SpaceRegion>::type complement(bool compute_topology = false,
                                                                                     bool simplicial_facets = false,
                                                                                     double merge_tol = 0.0,
                                                                                     double dist_tol = std::numeric_limits<ScalarT>::epsilon()) const
        {
            std::vector<HomogeneousVectorSet<ScalarT,EigenDim>> tuples;
            tuples.emplace_back(dim_+1,0);
            for (size_t p = 0; p < polytopes_.size(); p++) {
                const HomogeneousVectorSet<ScalarT,EigenDim>& hs(polytopes_[p].getFacetHyperplanes());
                std::vector<HomogeneousVectorSet<ScalarT,EigenDim>> tuples_new;
                for (size_t t = 0; t < tuples.size(); t++) {
                    HomogeneousVectorSet<ScalarT,EigenDim> tup_curr(dim_+1, tuples[t].cols()+1);
                    tup_curr.leftCols(tuples[t].cols()) = tuples[t];
                    for (size_t h = 0; h < hs.cols(); h++) {
                        tup_curr.col(tup_curr.cols()-1) = -hs.col(h);
                        tuples_new.emplace_back(tup_curr);
                    }
                }
                tuples = std::move(tuples_new);
            }

            ConvexPolytopeVector res_polytopes;
            for (size_t t = 0; t < tuples.size(); t++) {
                ConvexPolytope<ScalarT,EigenDim,IndexT> poly_tmp(tuples[t], dim_, compute_topology, simplicial_facets, merge_tol, dist_tol);
                if (!poly_tmp.isEmpty()) {
                    res_polytopes.emplace_back(std::move(poly_tmp));
                }
            }
            return SpaceRegion(std::move(res_polytopes));
        }

        inline SpaceRegion relativeComplement(const SpaceRegion &sr,
                                              bool compute_topology = false,
                                              bool simplicial_facets = false,
                                              double merge_tol = 0.0,
                                              double dist_tol = std::numeric_limits<ScalarT>::epsilon()) const
        {
            return intersectionWith(sr.complement(compute_topology, simplicial_facets, merge_tol, dist_tol),
                                    compute_topology, simplicial_facets, merge_tol, dist_tol);
        }

        inline size_t getSpaceDimension() const { return dim_; }

        bool isEmpty() const {
            for (size_t i = 0; i < polytopes_.size(); i++) {
                if (!polytopes_[i].isEmpty()) return false;
            }
            return true;
        }

        bool isBounded() const {
            for (size_t i = 0; i < polytopes_.size(); i++) {
                if (!polytopes_[i].isBounded()) return false;
            }
            return true;
        }

        // Inefficient
        template <ptrdiff_t Dim = EigenDim>
        typename std::enable_if<Dim != Eigen::Dynamic, double>::type getVolume(double merge_tol = 0.0,
                                                                               double dist_tol = std::numeric_limits<ScalarT>::epsilon()) const
        {
            if (!isBounded()) return std::numeric_limits<double>::infinity();

            ConvexPolytopeVector subsets;
            subsets.emplace_back(HomogeneousVectorSet<ScalarT,EigenDim>(EigenDim+1, 0));

            std::vector<size_t> subset_sizes;
            subset_sizes.emplace_back(0);
            double volume = 0.0;
            for (size_t i = 0; i < polytopes_.size(); i++) {
                ConvexPolytopeVector subsets_tmp(subsets);
                std::vector<size_t> subset_sizes_tmp(subset_sizes);
                for (size_t j = 0; j < subsets_tmp.size(); j++) {
                    subsets_tmp[j] = subsets_tmp[j].intersectionWith(polytopes_[i], false, false, merge_tol, dist_tol);
                    subset_sizes_tmp[j]++;
                    volume += (2.0*(subset_sizes_tmp[j]%2) - 1.0)*subsets_tmp[j].getVolume();
                }
                std::move(std::begin(subsets_tmp), std::end(subsets_tmp), std::back_inserter(subsets));
                std::move(std::begin(subset_sizes_tmp), std::end(subset_sizes_tmp), std::back_inserter(subset_sizes));
            }

            return volume;
        }

        // Inefficient
        template <ptrdiff_t Dim = EigenDim>
        typename std::enable_if<Dim == Eigen::Dynamic, double>::type getVolume(double merge_tol = 0.0,
                                                                               double dist_tol = std::numeric_limits<ScalarT>::epsilon()) const
        {
            if (!isBounded()) return std::numeric_limits<double>::infinity();

            ConvexPolytopeVector subsets;
            subsets.emplace_back(HomogeneousVectorSet<ScalarT,EigenDim>(dim_+1, 0), dim_);

            std::vector<size_t> subset_sizes;
            subset_sizes.emplace_back(0);
            double volume = 0.0;
            for (size_t i = 0; i < polytopes_.size(); i++) {
                ConvexPolytopeVector subsets_tmp(subsets);
                std::vector<size_t> subset_sizes_tmp(subset_sizes);
                for (size_t j = 0; j < subsets_tmp.size(); j++) {
                    subsets_tmp[j] = subsets_tmp[j].intersectionWith(polytopes_[i], false, false, merge_tol, dist_tol);
                    subset_sizes_tmp[j]++;
                    volume += (2.0*(subset_sizes_tmp[j]%2) - 1.0)*subsets_tmp[j].getVolume();
                }
                std::move(std::begin(subsets_tmp), std::end(subsets_tmp), std::back_inserter(subsets));
                std::move(std::begin(subset_sizes_tmp), std::end(subset_sizes_tmp), std::back_inserter(subset_sizes));
            }

            return volume;
        }

        inline const ConvexPolytopeVector& getConvexPolytopes() const { return polytopes_; }

        Vector<ScalarT,EigenDim> getInteriorPoint() const {
            for (size_t i = 0; i < polytopes_.size(); i++) {
                if (!polytopes_[i].isEmpty()) return polytopes_[i].getInteriorPoint();
            }
            return Vector<ScalarT,EigenDim>::Constant(dim_, 1, std::numeric_limits<ScalarT>::quiet_NaN());
        }

        bool containsPoint(const Eigen::Ref<const Vector<ScalarT,EigenDim>> &point, ScalarT offset = 0.0) const {
            for (size_t i = 0; i < polytopes_.size(); i++) {
                if (polytopes_[i].containsPoint(point, offset)) return true;
            }
            return false;
        }

        Eigen::Matrix<bool,1,Eigen::Dynamic> getInteriorPointsIndexMask(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                                                        ScalarT offset = 0.0) const
        {
            Eigen::Matrix<bool,1,Eigen::Dynamic> mask(1,points.cols());
            for (size_t i = 0; i < points.cols(); i++) {
                mask(i) = containsPoint(points.col(i), offset);
            }
            return mask;
        }

        template <typename IdxT = IndexT>
        std::vector<IdxT> getInteriorPointIndices(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &points,
                                                  ScalarT offset = 0.0) const
        {
            std::vector<IdxT> indices;
            indices.reserve(points.cols());
            for (size_t i = 0; i < points.cols(); i++) {
                if (containsPoint(points.col(i), offset)) indices.emplace_back(i);
            }
            return indices;
        }

        SpaceRegion& transform(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,EigenDim>> &rotation,
                               const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,1>> &translation)
        {
            for (size_t i = 0; i < polytopes_.size(); i++) {
                polytopes_[i].transform(rotation, translation);
            }
            return *this;
        }

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim != Eigen::Dynamic>::type>
        SpaceRegion& transform(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim+1,EigenDim+1>> &tform) {
            for (size_t i = 0; i < polytopes_.size(); i++) {
                polytopes_[i].transform(tform);
            }
            return *this;
        }

        template <ptrdiff_t Dim = EigenDim, class = typename std::enable_if<Dim == Eigen::Dynamic>::type>
        SpaceRegion& transform(const Eigen::Ref<const Eigen::Matrix<ScalarT,EigenDim,EigenDim>> &tform) {
            for (size_t i = 0; i < polytopes_.size(); i++) {
                polytopes_[i].transform(tform);
            }
            return *this;
        }

        inline SpaceRegion& transform(const RigidTransform<ScalarT,EigenDim> &tform) {
            for (size_t i = 0; i < polytopes_.size(); i++) {
                polytopes_[i].transform(tform);
            }
            return *this;
        }

        template <class TransformT>
        inline SpaceRegion transformed(const TransformT &tform) const {
            SpaceRegion res = *this;
            res.transform(tform);
            return res;
        }

        template <class RotationT, class TranslationT>
        inline SpaceRegion transformed(const RotationT &rot, const TranslationT trans) const {
            SpaceRegion res = *this;
            res.transform(rot, trans);
            return res;
        }

    protected:
        size_t dim_;
        ConvexPolytopeVector polytopes_;
    };

    typedef SpaceRegion<float,2> SpaceRegion2f;
    typedef SpaceRegion<double,2> SpaceRegion2d;
    typedef SpaceRegion<float,3> SpaceRegion3f;
    typedef SpaceRegion<double,3> SpaceRegion3d;
    typedef SpaceRegion<float,Eigen::Dynamic> SpaceRegionXf;
    typedef SpaceRegion<float,Eigen::Dynamic> SpaceRegionXd;
}
