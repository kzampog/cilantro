#include <sisyphus/point_cloud.hpp>
#include <sisyphus/ply_reader.h>

int main(int argc, char **argv) {
    PointCloud pointCloud;

    PlyReader::read_ply_file(argv[1], pointCloud);

    std::cout << pointCloud.num_points << std::endl;
    return 0;
}