#include <iostream>
#include <pangolin/utils/file_utils.h>
#include <cilantro/ply_io.hpp>


int main(int argc, char **argv) {
    PointCloud cloud;

    // Get file paths
    std::string fileInPath, fileOutPath;
    if (argc < 3) {
        std::cout << "No input path provided. Using default test cloud location." << std::endl;
        fileInPath = "../examples/test_clouds/kustas.ply";
        fileOutPath = "./kustas_copy.ply";
    } else {
        fileInPath = std::string(argv[1]);
        fileOutPath = std::string(argv[2]);
    }

    fileInPath = pangolin::PathExpand(fileInPath);
    fileOutPath = pangolin::PathExpand(fileOutPath);

    // Read ply file
    std::cout << "Reading file from " << fileInPath << std::endl;
    if (!pangolin::FileExists(fileInPath)) {
        std::cout << "Input file " << fileInPath << " doesn't exist." << std::endl;
        return -1;
    }

    readPointCloudFromPLYFile(fileInPath, cloud);
    std::cout << cloud.points.size() << " points read" << std::endl;
    std::cout << cloud.normals.size() << " normals read" << std::endl;
    std::cout << cloud.colors.size() << " colors read" << std::endl;

    // Modify pointcloud
    std::vector<size_t> ind;
    for (size_t i = 0; i < cloud.points.size(); i += 50) {
        ind.push_back(i);
    }
    PointCloud pc(cloud, ind);

    // Write ply file
    std::cout << "Writing pointcloud to " << fileOutPath << std::endl;
    if (!pangolin::FileExists(pangolin::PathParent(fileOutPath))) {
        std::cout << "Parent directory of output file " << fileOutPath << " doesn't exist." << std::endl;
        return -1;
    }

    writePointCloudToPLYFile(fileOutPath, pc);
    std::cout << pc.points.size() << " points written" << std::endl;
    std::cout << pc.normals.size() << " normals written" << std::endl;
    std::cout << pc.colors.size() << " colors written" << std::endl;

    return 0;
}
