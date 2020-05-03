#include <cilantro/utilities/multidimensional_scaling.hpp>
#include <cilantro/visualization.hpp>
#include <cilantro/utilities/timer.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void generate_input_data(cilantro::VectorSet3f &original_points,
                         cilantro::VectorSet<float,1> &values,
                         Eigen::MatrixXf &dist_sq)
{
    size_t num_points = 1000;
    original_points.resize(3, num_points);
    for (size_t i = 0; i < num_points; i++) {
        original_points(0,i) = std::cos((2.0f*M_PI*i)/num_points);
        original_points(1,i) = std::sin((2.0f*M_PI*i)/num_points);
        original_points(2,i) = 0.1f*std::sin(10.0f*(2.0f*M_PI*i)/num_points);
    }

    values = original_points.row(2);

    dist_sq.resize(num_points, num_points);
    for (size_t i = 0; i < original_points.cols(); i++) {
        for (size_t j = 0; j < original_points.cols(); j++) {
            dist_sq(i,j) = (original_points.col(i) - original_points.col(j)).squaredNorm();
        }
    }
}

int main(int argc, char ** argv) {
    // Prepare input data
    cilantro::VectorSet3f points;
    cilantro::VectorSet<float,1> values;
    Eigen::MatrixXf distances_sq;
    generate_input_data(points, values, distances_sq);

    std::cout << "Number of points: " << distances_sq.rows() << std::endl;

    cilantro::Timer timer;
    timer.start();

    cilantro::MultidimensionalScaling<float,2> mds(distances_sq);
    // cilantro::MultidimensionalScaling<float> mds(distances_sq, 3, true);

    timer.stop();
    std::cout << "Elapsed time: " << timer.getElapsedTime() << "ms" << std::endl;
    std::cout << "Embedding dimension: " << mds.getEmbeddedPoints().rows() << std::endl;

    size_t dim = mds.getEmbeddedPoints().rows();
//    std::cout << mds.getComputedEigenValues().transpose() << std::endl;

    // Create a new cloud by re-embedding the embedded points back to 3D
    cilantro::VectorSet<float,3> points_reproj(3, distances_sq.rows());
    points_reproj.topRows(dim) = mds.getEmbeddedPoints();
    points_reproj.bottomRows(3-dim).setZero();

    // Visualize result
    pangolin::CreateWindowAndBind("MultidimensionalScaling demo",1280,480);
    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual).AddDisplay(pangolin::Display("disp1")).AddDisplay(pangolin::Display("disp2"));

    cilantro::Visualizer viz1("MultidimensionalScaling demo", "disp1");
    viz1.addObject<cilantro::PointCloudRenderable>("cloud", points, cilantro::RenderingProperties().setPointSize(3.0f))
            ->setPointValues(values);

    cilantro::Visualizer viz2("MultidimensionalScaling demo", "disp2");
    viz2.addObject<cilantro::PointCloudRenderable>("cloud_reproj", points_reproj, cilantro::RenderingProperties().setPointSize(3.0f))
            ->setPointValues(values);

    // Move camera
    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.25f*M_PI, Eigen::Vector3f::UnitX());
    Eigen::Vector3f t(0.0f, 2.0f, -2.0f);
    Eigen::Matrix4f cam_pose = Eigen::Matrix4f::Identity();
    cam_pose.topLeftCorner(3,3) = rot;
    cam_pose.topRightCorner(3,1) = t;
    viz1.setCameraPose(cam_pose);
    viz1.setDefaultCameraPose(cam_pose);

    // Keep viewpoints in sync
    viz2.setRenderState(viz1.getRenderState());

    while (!viz1.wasStopped() && !viz2.wasStopped()) {
        viz1.clearRenderArea();
        viz1.render();
        viz2.render();
        pangolin::FinishFrame();
    }

    return 0;
}
