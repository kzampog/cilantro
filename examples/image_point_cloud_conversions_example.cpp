#include <cilantro/point_cloud.hpp>
#include <cilantro/image_viewer.hpp>
#include <cilantro/visualizer.hpp>
#include <cilantro/common_renderables.hpp>

int main(int argc, char ** argv) {
    // Intrinsics
    Eigen::Matrix3f K;
    K << 528, 0, 320, 0, 528, 240, 0, 0, 1;

//    std::string uri = "files://[/home/kzampog/Desktop/rgbd_sequences/dok_demo/rgb_*.png,/home/kzampog/Desktop/rgbd_sequences/dok_demo/depth_*.png]";
    std::string uri = "openni2:[img1=rgb,img2=depth_reg,coloursync=true,closerange=true,holefilter=true]//";

    std::unique_ptr<pangolin::VideoInterface> dok = pangolin::OpenVideo(uri);
    size_t w = 640, h = 480;
    unsigned char* img = new unsigned char[dok->SizeBytes()];

    pangolin::Image<unsigned char> rgb_img(img, w, h, 3*w*sizeof(unsigned char));
    pangolin::Image<unsigned short> depth_img((unsigned short *)(img+3*w*h), w, h, w*sizeof(unsigned short));

    cilantro::PointCloud3f cloud;
    pangolin::ManagedImage<float> depthf_img(w, h);

    std::string win_name = "Image/point cloud conversions demo";
    pangolin::CreateWindowAndBind(win_name, 1280, 960);
    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual)
            .AddDisplay(pangolin::Display("disp1")).AddDisplay(pangolin::Display("disp2"))
            .AddDisplay(pangolin::Display("disp3")).AddDisplay(pangolin::Display("disp4"));

    cilantro::ImageViewer rgbv(win_name, "disp1");
    cilantro::ImageViewer depthv(win_name, "disp2");
    cilantro::Visualizer pcdv(win_name, "disp3");
    cilantro::ImageViewer depthfv(win_name, "disp4");

    while (!pcdv.wasStopped() && !rgbv.wasStopped() && !depthv.wasStopped()) {
        dok->GrabNext(img, true);

        // Get point cloud from RGBD image pair
        cloud.template fromRGBDImages<unsigned short>(rgb_img.ptr, depth_img.ptr, w, h, K, false, cilantro::DepthValueConverter<unsigned short,float>(1000.0f));
//        cilantro::RGBDImagesToPointsColors<unsigned short,float>(rgb_img.ptr, depth_img.ptr, w, h, K, cloud.points, cloud.colors, false, cilantro::DepthValueConverter<unsigned short,float>(1000.0f));

        // Get a depth map back from the point cloud
        cilantro::pointsToDepthImage<float,float>(cloud.points, K, depthf_img.ptr, w, h, cilantro::DepthValueConverter<float,float>(1.0f));

        rgbv.setImage(rgb_img.ptr, w, h, "RGB24");
        depthv.setImage(depth_img.ptr, w, h, "GRAY16LE");
        pcdv.addObject<cilantro::PointCloudRenderable>("cloud", cloud);
        depthfv.setImage(depthf_img.ptr, w, h, "GRAY32F");

        pcdv.clearRenderArea();
        rgbv.render();
        depthv.render();
        pcdv.render();
        depthfv.render();
        pangolin::FinishFrame();
    }

    delete[] img;

    return 0;
}
