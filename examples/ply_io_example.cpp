#include <sisyphus/point_cloud.hpp>
#include <sisyphus/ply_io.h>

int main(int argc, char **argv) {
    PointCloud pointCloud;

    PlyIO::read_ply_file(argv[1], pointCloud);
    std::cout << pointCloud.num_points << std::endl;

    PlyIO::write_ply_file(argv[2], pointCloud);
    return 0;
}