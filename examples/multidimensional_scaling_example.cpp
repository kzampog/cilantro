#include <cilantro/kd_tree.hpp>
#include <cilantro/multidimensional_scaling.hpp>
#include <cilantro/point_cloud.hpp>
#include <cilantro/visualizer.hpp>

int main(int argc, char ** argv) {
// TODO

//    cilantro::PointCloud3D cloud;
//    cloud.points.resize(Eigen::NoChange, 1700);
//    for (size_t i = 0; i < 1500; i++) {
//        cloud.points.col(i).setRandom().normalize();
//    }
//    for (size_t i = 1500; i < 1700; i++) {
//        cloud.points.col(i).setRandom().normalize();
//        cloud.points.col(i) *= 0.3f;
//    }
//    cloud.points.row(2).array() += 4.0f;
//
//    Eigen::MatrixXf data(cloud.size(),cloud.size());
//    for (size_t i = 0; i < cloud.size(); i++) {
//        for (size_t j = 0; j < cloud.size(); j++) {
//            data(i,j) = (cloud.points.col(i) - cloud.points.col(j)).norm();
//        }
//    }
//
//    std::cout << "Number of points: " << cloud.size() << std::endl;
//
//    size_t max_dim = 4;
//
//    auto start = std::chrono::high_resolution_clock::now();
//
//    cilantro::MultidimensionalScaling<float,3> mds(data);
//
//    auto end = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double, std::milli> elapsed = end - start;
//    std::cout << "Elapsed time: " << elapsed.count() << "ms" << std::endl;
//
//    // Create a new colored cloud
//    cilantro::PointCloud3D reproj(mds.getEmbeddedPoints());
//
//    // Visualize result
//    pangolin::CreateWindowAndBind("MultidimensionalScaling demo",1280,480);
//    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual).AddDisplay(pangolin::Display("disp1")).AddDisplay(pangolin::Display("disp2"));
//
//    cilantro::Visualizer viz1("MultidimensionalScaling demo", "disp1");
//    viz1.addPointCloud("cloud", cloud, cilantro::RenderingProperties().setPointSize(5.0f));
//
//    cilantro::Visualizer viz2("MultidimensionalScaling demo", "disp2");
//    viz2.addPointCloud("cloud_seg", reproj, cilantro::RenderingProperties().setPointSize(5.0f));
//
//    while (!viz1.wasStopped() && !viz2.wasStopped()) {
//        viz1.clearRenderArea();
//        viz1.render();
//        viz2.render();
//        pangolin::FinishFrame();
//    }

    return 0;
}
