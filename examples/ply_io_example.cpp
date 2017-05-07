#include <sisyphus/ply_io.hpp>
#include <iostream>

int main(int argc, char **argv) {
    PointCloud cloud;

    readPointCloudFromPLYFile(argv[1], cloud);

    std::cout << cloud.points.size() << " points read" << std::endl;
    std::cout << cloud.normals.size() << " normals read" << std::endl;
    std::cout << cloud.colors.size() << " colors read" << std::endl;

    std::vector<size_t> ind;
    for (size_t i = 0; i < cloud.points.size(); i += 50) {
        ind.push_back(i);
    }
    PointCloud pc(cloud, ind);

    writePointCloudToPLYFile(argv[2], pc);

    std::cout << pc.points.size() << " points written" << std::endl;
    std::cout << pc.normals.size() << " normals written" << std::endl;
    std::cout << pc.colors.size() << " colors written" << std::endl;

    return 0;
}
