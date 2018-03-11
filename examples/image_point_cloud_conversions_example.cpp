#include <cilantro/image_point_cloud_conversions.hpp>
#include <cilantro/visualizer.hpp>
#include <cilantro/image_viewer.hpp>

int main(int argc, char ** argv) {
    // Intrinsics
    Eigen::Matrix3f K;
    K << 528, 0, 320, 0, 528, 240, 0, 0, 1;

//    std::string uri = "files://[/home/kzampog/Desktop/rgbd_sequences/dok_demo/rgb_*.png,/home/kzampog/Desktop/rgbd_sequences/dok_demo/depth_*.png]";
    std::string uri = "openni2:[img1=rgb,img2=depth_reg,coloursync=true,closerange=true,holefilter=true]//";

    std::unique_ptr<pangolin::VideoInterface> dok = pangolin::OpenVideo(uri);
    size_t w = 640, h = 480;
    unsigned char* img = new unsigned char[dok->SizeBytes()];

    pangolin::Image<Eigen::Matrix<unsigned char,3,1> > rgb_img((Eigen::Matrix<unsigned char,3,1> *)img, w, h, w*sizeof(Eigen::Matrix<unsigned char,3,1>));
    pangolin::Image<unsigned short> depth_img((unsigned short *)(img+3*w*h), w, h, w*sizeof(unsigned short));

    cilantro::PointCloud3f cloud;

    std::string win_name = "Image/point cloud conversions demo";
    pangolin::CreateWindowAndBind(win_name, 1280, 960);
    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual)
            .AddDisplay(pangolin::Display("disp1")).AddDisplay(pangolin::Display("disp2")).AddDisplay(pangolin::Display("disp3"));

    cilantro::ImageViewer rgbv(win_name, "disp1");
    cilantro::ImageViewer depthv(win_name, "disp2");
    cilantro::Visualizer pcdv(win_name, "disp3");

    while (!pcdv.wasStopped() && !rgbv.wasStopped() && !depthv.wasStopped()) {
        dok->GrabNext(img, true);
        RGBDImagesToPointCloud(rgb_img, depth_img, K, cloud, false);

        pcdv.addPointCloud("cloud", cloud);
        rgbv.setImage(rgb_img.ptr, w, h, "RGB24");
        depthv.setImage(depth_img.ptr, w, h, "GRAY16LE");

        pcdv.clearRenderArea();
        pcdv.render();
        rgbv.render();
        depthv.render();
        pangolin::FinishFrame();
    }

    delete[] img;

    return 0;
}
