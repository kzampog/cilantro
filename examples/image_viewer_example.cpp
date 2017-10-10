#include <cilantro/image_viewer.hpp>

int main( int argc, char* argv[] )
{
//    std::string uri = "files://[/home/kzampog/Desktop/rgbd_sequences/dok_demo/rgb_*.png,/home/kzampog/Desktop/rgbd_sequences/dok_demo/depth_*.png]";
    std::string uri = "openni2:[img1=rgb,img2=depth_reg,coloursync=true,closerange=true,holefilter=true]//";

    std::unique_ptr<pangolin::VideoInterface> dok = pangolin::OpenVideo(uri);

    ImageViewer rgbv("RGB", "disp1");
    ImageViewer depthv("DEPTH", "disp2");

    size_t w = 640, h = 480;
    unsigned char* img = new unsigned char[dok->SizeBytes()];
    while (dok->GrabNext(img, true) && !rgbv.wasStopped() && !depthv.wasStopped()) {
        rgbv.setImage(img, w, h, "RGB24").spinOnce();
        depthv.setImage(img + 3*w*h, w, h, "GRAY16LE").spinOnce();
    }

    delete[] img;

    return 0;
}
