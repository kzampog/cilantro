//
// Created by kganguly on 5/4/17.
//

#ifndef SISYPHUS_PLY_READER_H
#define SISYPHUS_PLY_READER_H

#include "tinyply/tinyply.h"
#include "sisyphus/point_cloud.hpp"
#include <fstream>

class PlyReader {
public:
    void read_ply_file(const std::string &filename, PointCloud &poClo);
};

#endif //SISYPHUS_PLY_READER_H
