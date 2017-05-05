#include <iostream>
#include <sisyphus/ply_io.hpp>

int main(int argc, char **argv) {
    PointCloud cloud;

    readPointCloudFromPLYFile(argv[1], cloud);
    std::cout << cloud.points.size() << " vertices read" << std::endl;

    writePointCloudToPLYFile(argv[2], cloud);

    return 0;
}
