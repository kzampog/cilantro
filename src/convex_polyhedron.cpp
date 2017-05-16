#include <cilantro/convex_polyhedron.hpp>

#include <iostream>

void test() {
    orgQhull::RboxPoints rbox("4");
    orgQhull::Qhull q(rbox, "");
    orgQhull::QhullFacetList facets= q.facetList();
    std::cout << facets;
}
