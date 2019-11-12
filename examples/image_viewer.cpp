#include <cilantro/visualization/image_viewer.hpp>

int main( int argc, char* argv[] )
{
//    std::string uri = "files://[/home/kzampog/Desktop/rgbd_sequences/dok_demo/rgb_*.png,/home/kzampog/Desktop/rgbd_sequences/dok_demo/depth_*.png]";
    std::string uri = "openni2:[img1=rgb,img2=depth_reg,coloursync=true,closerange=true,holefilter=true]//";

    std::unique_ptr<pangolin::VideoInterface> dok = pangolin::OpenVideo(uri);

    pangolin::CreateWindowAndBind("ImageViewer demo",1280,480);
    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual).AddDisplay(pangolin::Display("disp1")).AddDisplay(pangolin::Display("disp2"));

    cilantro::ImageViewer rgbv("ImageViewer demo", "disp1");
    cilantro::ImageViewer depthv("ImageViewer demo", "disp2");

    size_t w = 640, h = 480;
    unsigned char* img = new unsigned char[dok->SizeBytes()];
    while (dok->GrabNext(img, true) && !rgbv.wasStopped() && !depthv.wasStopped()) {
        rgbv.clearRenderArea().setImage(img, w, h, "RGB24").render();
        depthv.setImage(img + 3*w*h, w, h, "GRAY16LE").render();
        pangolin::FinishFrame();
    }

    delete[] img;

    return 0;
}
