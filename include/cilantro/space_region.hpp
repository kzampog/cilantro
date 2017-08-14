#pragma once

#include <cilantro/convex_polytope.hpp>

template <typename InputScalarT, typename OutputScalarT, ptrdiff_t EigenDim>
class SpaceRegion {
public:
    SpaceRegion() {}
    ~SpaceRegion() {}


private:
    std::vector<ConvexPolytope<InputScalarT,OutputScalarT,EigenDim> > polytopes_;

};
