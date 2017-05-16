#pragma once

#include <libqhullcpp/RboxPoints.h>
#include <libqhullcpp/QhullError.h>
#include <libqhullcpp/QhullQh.h>
#include <libqhullcpp/QhullFacetList.h>
#include <libqhullcpp/QhullLinkedList.h>
#include <libqhullcpp/Qhull.h>

#include <cilantro/point_cloud.hpp>

void VtoH(const std::vector<Eigen::Vector3f> &points);

void HtoV(const std::vector<Eigen::Vector4f> &faces);
