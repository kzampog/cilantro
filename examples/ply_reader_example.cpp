#include <sisyphus/point_cloud.hpp>
#include <sisyphus/ply_reader.h>

int main(int argc, char **argv) {
    PointCloud pointCloud;
    PlyReader::read_ply_file("test_clouds/kustas.ply", pointCloud);
    return 0;
}