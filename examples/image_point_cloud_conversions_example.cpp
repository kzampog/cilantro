#include <cilantro/image_point_cloud_conversions.hpp>
#include <cilantro/visualizer.hpp>
#include <cilantro/image_viewer.hpp>

#include <cilantro/ply_io.hpp>

#include <ctime>

int main( int argc, char* argv[] )
{
//    std::string uri = "files://[/home/kzampog/Desktop/rgbd_sequences/dok_demo/rgb_*.png,/home/kzampog/Desktop/rgbd_sequences/dok_demo/depth_*.png]";
    std::string uri = "openni2:[img1=rgb,img2=depth_reg,coloursync=true,closerange=true,holefilter=true]//";

    std::unique_ptr<pangolin::VideoInterface> dok = pangolin::OpenVideo(uri);

    Visualizer pcdv("CLOUD", "disp");
    ImageViewer rgbv("RGB", "disp");
    ImageViewer depthv("DEPTH", "disp");

    Eigen::Matrix3f K;
    K << 528, 0, 320, 0, 528, 240, 0, 0, 1;

    size_t w = 640, h = 480;
    PointCloud cloud;
    unsigned char* img = new unsigned char[dok->SizeBytes()];
    while (dok->GrabNext(img, true)) {
        pangolin::Image<Eigen::Matrix<unsigned char,3,1> > rgb_img((Eigen::Matrix<unsigned char,3,1> *)img, w, h, w*sizeof(Eigen::Matrix<unsigned char,3,1>));
        pangolin::Image<unsigned short> depth_img((unsigned short *)(img+3*w*h), w, h, w*sizeof(unsigned short));

        pangolin::ManagedImage<Eigen::Matrix<unsigned char,3,1> > rgb_img2(w,h);
        pangolin::ManagedImage<unsigned short> depth_img2(w,h);

        clock_t begin, end;
        double build_time;
        begin = clock();

        RGBDImagesToPointCloud(rgb_img, depth_img, K, cloud, false);
//        depthImageToPointCloud(depth_img, K, cloud, false);

//        Eigen::Matrix3f R;
//        R.setIdentity();
//        R(0,0) = -1.0f;
//        R(1,1) = -1.0f;
//        Eigen::Vector3f t(0,0,0);

//        pointCloudToRGBDImages(cloud, K, R, t, rgb_img2, depth_img2);
        pointCloudToRGBDImages(cloud, K, rgb_img2, depth_img2);
//        pointCloudToDepthImage(cloud, K, depth_img2);

        end = clock();
        build_time = 1000.0*double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "Images to cloud and back: " << build_time << std::endl;

        pcdv.addPointCloud("cloud", cloud).spinOnce();
        rgbv.setImage(rgb_img2.ptr, w, h, "RGB24").spinOnce();
        depthv.setImage(depth_img2.ptr, w, h, "GRAY16LE").spinOnce();
    }

    delete[] img;

//    Eigen::Matrix3f K;
//    K << 528, 0, 320, 0, 528, 240, 0, 0, 1;
//    size_t w = 640, h = 480;
//
//    PointCloud cloud;
//    readPointCloudFromPLYFile(argv[1], cloud);
//
//    pangolin::ManagedImage<size_t> idx_map(w, h);
//    pointCloudToIndexMap(cloud, K, idx_map);
//
//    std::vector<size_t> indices;
//    for (size_t y = 0; y < h; y++) {
//        for (size_t x = 0; x < w; x++) {
//            if (idx_map(x,y) != std::numeric_limits<std::size_t>::max()) indices.push_back(idx_map(x,y));
//        }
//    }
//
//    std::cout << indices.size() << std::endl;
//
//    PointCloud cloud2(cloud, indices);
//    Visualizer viz("CLOUD", "disp");
//    viz.addPointCloud("cloud2", cloud2);
//
//    while (!viz.wasStopped()) {
//        viz.spinOnce();
//    }

    return 0;
}
