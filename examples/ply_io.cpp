#include <iostream>
#include <pangolin/utils/file_utils.h>
#include <cilantro/utilities/point_cloud.hpp>

int main(int argc, char **argv) {
    cilantro::PointCloud3f cloud;

    // Get file paths
    std::string fileInPath, fileOutPath;
    if (argc < 3) {
        std::cout << "No input path provided. Using default test cloud location." << std::endl;
        fileInPath = "../examples/test_clouds/test.ply";
        fileOutPath = "./test_copy.ply";
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

    cloud.fromPLYFile(fileInPath);
    std::cout << cloud.points.cols() << " points read" << std::endl;
    std::cout << cloud.normals.cols() << " normals read" << std::endl;
    std::cout << cloud.colors.cols() << " colors read" << std::endl;

    // Modify pointcloud
    std::vector<size_t> ind;
    for (size_t i = 0; i < cloud.points.cols(); i += 50) {
        ind.push_back(i);
    }
    cilantro::PointCloud3f pc(cloud, ind);

    // Write ply file
    std::cout << "Writing pointcloud to " << fileOutPath << std::endl;
    if (!pangolin::FileExists(pangolin::PathParent(fileOutPath))) {
        std::cout << "Parent directory of output file " << fileOutPath << " doesn't exist." << std::endl;
        return -1;
    }

    pc.toPLYFile(fileOutPath);
    std::cout << pc.points.cols() << " points written" << std::endl;
    std::cout << pc.normals.cols() << " normals written" << std::endl;
    std::cout << pc.colors.cols() << " colors written" << std::endl;

    return 0;
}
