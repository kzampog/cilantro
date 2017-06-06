#include <cilantro/image_viewer.hpp>
#include <cilantro/visualizer.hpp>

#include <ctime>

int main( int argc, char* argv[] )
{
    Eigen::Matrix3f K;
    K << 528, 0, 320, 0, 528, 240, 0, 0, 1;

//    std::string uri = "files://[/home/kzampog/Desktop/rgbd_sequences/dok_demo/rgb_*.png,/home/kzampog/Desktop/rgbd_sequences/dok_demo/depth_*.png]";
    std::string uri = "openni2:[img1=rgb,img2=depth_reg,coloursync=true,closerange=true,holefilter=true]//";

    std::unique_ptr<pangolin::VideoInterface> dok = pangolin::OpenVideo(uri);

    ImageViewer rgbv("RGB", "disp1");
    ImageViewer depthv("DEPTH", "disp2");
    Visualizer viz("CLOUD", "disp3");

    size_t w = 640, h = 480;
    unsigned char* img = new unsigned char[dok->SizeBytes()];
    while (dok->GrabNext(img, true)) {

        clock_t begin, end;
        double build_time;
        begin = clock();

        pangolin::Image<Eigen::Matrix<unsigned char,3,1> > rgb_img((Eigen::Matrix<unsigned char,3,1> *)img, w, h, w*sizeof(Eigen::Matrix<unsigned char,3,1>));
        pangolin::Image<unsigned short> depth_img((unsigned short *)(img + 3*w*h), w, h, w*sizeof(unsigned short));

        PointCloud cloud;
        cloud.points.reserve(w*h);
        cloud.colors.reserve(w*h);
        for (size_t row = 0; row < h; row++) {
            for (size_t col = 0; col < w; col++) {
                float d = depth_img(col,row)/1000.0f;
                if (d > 0.0f) {
                    cloud.points.push_back(Eigen::Vector3f((col - K(0,2)) * d / K(0,0), (row - K(1,2)) * d / K(1,1), d));
                    cloud.colors.push_back(rgb_img(col,row).cast<float>()/255.0f);
                }
            }
        }

        end = clock();
        build_time = 1000.0*double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "Cloud generation: " << build_time << std::endl;

        rgbv.setImage(img, w, h, "RGB24").spinOnce();
        depthv.setImage(img + 3*w*h, w, h, "GRAY16LE").spinOnce();
        viz.addPointCloud("cloud", cloud).spinOnce();
    }

    delete[] img;

    return 0;
}
