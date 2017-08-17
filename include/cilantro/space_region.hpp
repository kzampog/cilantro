#pragma once

#include <cilantro/convex_polytope.hpp>

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
class SpaceRegion {
public:
    SpaceRegion() : polytopes_(std::vector<ConvexPolytope<InputScalarT,OutputScalarT,EigenDim> >(0)) {}
    SpaceRegion(const ConvexPolytope<InputScalarT,OutputScalarT,EigenDim> &polytope) : polytopes_(std::vector<ConvexPolytope<InputScalarT,OutputScalarT,EigenDim> >(1,polytope)) {}
    SpaceRegion(const std::vector<ConvexPolytope<InputScalarT,OutputScalarT,EigenDim> > &polytopes) : polytopes_(polytopes) {}

    ~SpaceRegion() {}

    SpaceRegion unionWith(const SpaceRegion &sr) const {
        std::vector<ConvexPolytope<InputScalarT,OutputScalarT,EigenDim> > polytopes(polytopes_);
        polytopes.insert(polytopes.end(), sr.polytopes_.begin(), sr.polytopes_.end());
        return SpaceRegion(polytopes);
    }

    SpaceRegion intesectionWith(const SpaceRegion &sr) const {
//        std::vector<ConvexPolytope<InputScalarT,OutputScalarT,EigenDim> > polytopes(polytopes_);
//        polytopes.insert(polytopes.end(), sr.polytopes_.begin(), sr.polytopes_.end());
//        return SpaceRegion(polytopes);
    }

    SpaceRegion complement(const SpaceRegion &sr) const {
//        std::vector<ConvexPolytope<InputScalarT,OutputScalarT,EigenDim> > polytopes(polytopes_);
//        polytopes.insert(polytopes.end(), sr.polytopes_.begin(), sr.polytopes_.end());
//        return SpaceRegion(polytopes);
    }


    inline bool isEmpty() const {
        for (size_t i = 0; i < polytopes_.size(); i++) {
            if (!polytopes_[i].isEmpty()) return false;
        }
        return true;
    }

    inline bool isBounded() const {
        for (size_t i = 0; i < polytopes_.size(); i++) {
            if (!polytopes_[i].isBounded()) return false;
        }
        return true;
    }

    inline const Eigen::Matrix<OutputScalarT,EigenDim,1>& getInteriorPoint() const {
        for (size_t i = 0; i < polytopes_.size(); i++) {
            if (!polytopes_[i].isEmpty()) return polytopes_[i].getInteriorPoint();
        }
        return nan_point_;
    }

    inline bool containsPoint(const Eigen::Matrix<OutputScalarT,EigenDim,1> &point) const {
        for (size_t i = 0; i < polytopes_.size(); i++) {
            if (polytopes_[i].containsPoint(point)) return true;
        }
        return false;
    }

    inline Eigen::Matrix<bool,1,Eigen::Dynamic> getInteriorPointsIndexMask(const std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &points, OutputScalarT offset = 0.0) const {
        Eigen::Matrix<bool,1,Eigen::Dynamic> mask(Eigen::Matrix<bool,1,Eigen::Dynamic>::Zero(1,points.size()));
        for (size_t i = 0; i < polytopes_.size(); i++) {
            mask = mask.cwiseMax(polytopes_[i].getInteriorPointsIndexMask(points, offset));
        }
        return mask;
    }

    std::vector<size_t> getInteriorPointIndices(const std::vector<Eigen::Matrix<OutputScalarT,EigenDim,1> > &points, OutputScalarT offset = 0.0) const {
        Eigen::Matrix<bool,1,Eigen::Dynamic> mask(getInteriorPointsIndexMask(points, offset));
        std::vector<size_t> indices;
        indices.reserve(points.size());
        for (size_t i = 0; i < points.size(); i++) {
            if (mask[i]) indices.emplace_back(i);
        }
        return indices;
    }

private:
    std::vector<ConvexPolytope<InputScalarT,OutputScalarT,EigenDim> > polytopes_;

    static Eigen::Matrix<OutputScalarT,EigenDim,1> nan_point_;
};

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
Eigen::Matrix<OutputScalarT,EigenDim,1> SpaceRegion<InputScalarT,OutputScalarT,EigenDim>::nan_point_ = Eigen::Matrix<OutputScalarT,EigenDim,1>::Constant(std::numeric_limits<OutputScalarT>::quiet_NaN());
