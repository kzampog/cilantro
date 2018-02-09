#include <cilantro/multidimensional_scaling.hpp>
#include <cilantro/visualizer.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(int argc, char ** argv) {
    size_t num_points = 1000;
    cilantro::VectorSet<float,3> points(3,num_points);
    for (size_t i = 0; i < num_points; i++) {
        points(0,i) = std::cos((2.0f*M_PI*i)/num_points);
        points(1,i) = std::sin((2.0f*M_PI*i)/num_points);
        points(2,i) = 0.1f*std::sin(10.0f*(2.0f*M_PI*i)/num_points);
    }

    cilantro::VectorSet<float,1> values = points.row(2);
    cilantro::VectorSet<float,3> colors = cilantro::colormap(values);

    Eigen::MatrixXf distances(num_points, num_points);
    for (size_t i = 0; i < points.cols(); i++) {
        for (size_t j = 0; j < points.cols(); j++) {
            distances(i,j) = (points.col(i) - points.col(j)).norm();
        }
    }

    std::cout << "Number of points: " << points.cols() << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

//    cilantro::MultidimensionalScaling<float,2> mds(distances);
    cilantro::MultidimensionalScaling<float> mds(distances, 3, true);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << "ms" << std::endl;
    std::cout << "Embedding dimension: " << mds.getEmbeddedPoints().rows() << std::endl;

    size_t dim = mds.getEmbeddedPoints().rows();
//    std::cout << mds.getComputedEigenValues().transpose() << std::endl;

    // Create a new cloud by reprojecting the embedded points back to 3D
    cilantro::VectorSet<float,3> points_reproj(3, num_points);
    points_reproj.topRows(dim) = mds.getEmbeddedPoints();
    points_reproj.bottomRows(3-dim).setZero();

    // Visualize result
    pangolin::CreateWindowAndBind("MultidimensionalScaling demo",1280,480);
    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual).AddDisplay(pangolin::Display("disp1")).AddDisplay(pangolin::Display("disp2"));

    cilantro::Visualizer viz1("MultidimensionalScaling demo", "disp1");
    viz1.addPointCloud("cloud", points, cilantro::RenderingProperties().setPointSize(3.0f));
    viz1.addPointCloudColors("cloud", colors);

    cilantro::Visualizer viz2("MultidimensionalScaling demo", "disp2");
    viz2.addPointCloud("cloud_reproj", points_reproj, cilantro::RenderingProperties().setPointSize(3.0f));
    viz2.addPointCloudColors("cloud_reproj", colors);

    // Move camera
    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.25f*M_PI, Eigen::Vector3f::UnitX());
    Eigen::Vector3f t(0.0f, 2.0f, -2.0f);
    Eigen::Matrix4f cam_pose = Eigen::Matrix4f::Identity();
    cam_pose.topLeftCorner(3,3) = rot;
    cam_pose.topRightCorner(3,1) = t;
    viz1.setCameraPose(cam_pose);
    viz2.setCameraPose(cam_pose);
    viz1.setDefaultCameraPose(cam_pose);
    viz2.setDefaultCameraPose(cam_pose);

    while (!viz1.wasStopped() && !viz2.wasStopped()) {
        viz1.clearRenderArea();
        viz1.render();
        viz2.render();
        pangolin::FinishFrame();
    }

    return 0;
}
