#include <cilantro/kmeans.hpp>
#include <cilantro/ply_io.hpp>
#include <cilantro/visualizer.hpp>
#include <cilantro/voxel_grid.hpp>

#include <iostream>
#include <ctime>

int main(int argc, char ** argv) {

    PointCloud cloud;
    readPointCloudFromPLYFile(argv[1], cloud);

//    cloud = VoxelGrid(cloud, 0.005).getDownsampledCloud();

    KMeans3D kmc(cloud.points);

    size_t k = 100;

    clock_t begin, end;
    double elapsed_time;
    begin = clock();

    kmc.cluster(k, 100, 0.001);

    end = clock();
    elapsed_time = 1000.0*double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "Clustering time: " << elapsed_time << std::endl;

    std::cout << "Performed iterations: " << kmc.getPerformedIterationsCount() << std::endl;

    const std::vector<std::vector<size_t> >& cpi(kmc.getClusterPointIndices());

    size_t mins = cloud.size(), maxs = 0;
    for (size_t i = 0; i < cpi.size(); i++) {
        if (cpi[i].size() < mins) mins = cpi[i].size();
        if (cpi[i].size() > maxs) maxs = cpi[i].size();
    }

    std::cout << "Cluster size range is: [" << mins << "," << maxs << "]" << std::endl;

    Visualizer viz("win", "disp");

    std::vector<Eigen::Vector3f> color_map(k);
    for (size_t i = 0; i < k; i++) {
        color_map[i] = Eigen::Vector3f::Random().array().abs();
    }

    const std::vector<size_t>& idx_map(kmc.getClusterIndexMap());

    std::vector<Eigen::Vector3f> cols(idx_map.size());
    for (size_t i = 0; i < cols.size(); i++) {
        cols[i] = color_map[idx_map[i]];
    }

    viz.addPointCloud("cloud", cloud);
    viz.addPointCloudColors("cloud", cols);

    while (!viz.wasStopped()) {
        viz.spinOnce();
    }

//    for (size_t i = 0; i < idx_map.size(); i++) {
//        std::cout << idx_map[i] << " ";
//    }
//    std::cout << std::endl;

//    std::cout << kmc.getClusterCentroidsMatrixMap() << std::endl;

    return 0;
}
