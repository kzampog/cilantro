#include <cilantro/point_cloud.hpp>
#include <cilantro/image_viewer.hpp>
#include <cilantro/visualizer.hpp>
#include <cilantro/common_renderables.hpp>

void color_toggle_callback(cilantro::Visualizer &viz, cilantro::RenderingProperties &rp) {
    if (rp.pointColor == cilantro::RenderingProperties::noColor) {
        rp.setPointColor(0.8f, 0.8f, 0.8f);
    } else {
        rp.setPointColor(cilantro::RenderingProperties::noColor);
    }
    viz.setRenderingProperties("cloud", rp);
}

int main(int argc, char ** argv) {
    // Intrinsics
    Eigen::Matrix3f K;
    K << 525, 0, 319.5, 0, 525, 239.5, 0, 0, 1;

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

    cilantro::RenderingProperties rp;
    pcdv.registerKeyboardCallback('c', std::bind(color_toggle_callback, std::ref(pcdv), std::ref(rp)));
    rp.setUseLighting(false);

    std::cout << "Press 'l' to toggle lighting" << std::endl;
    std::cout << "Press 'c' to toggle color" << std::endl;
    std::cout << "Press 'n' to toggle rendering of normals" << std::endl;

    while (!pcdv.wasStopped() && !rgbv.wasStopped() && !depthv.wasStopped()) {
        dok->GrabNext(img, true);

        // Get point cloud from RGBD image pair
        cilantro::DepthValueConverter<unsigned short,float> dc1(1000.0f);
//        cloud.fromDepthImage(depth_img.ptr, dc1, w, h, K, false, true);
//        cloud.fromRGBDImages(rgb_img.ptr, depth_img.ptr, dc1, w, h, K, false, true);
//        cilantro::RGBDImagesToPointsColors(rgb_img.ptr, depth_img.ptr, dc1, w, h, K, cloud.points, cloud.colors, false);
//        cilantro::depthImageToPointsNormals(depth_img.ptr, dc1, w, h, K, cloud.points, cloud.normals, false);
        cilantro::RGBDImagesToPointsNormalsColors(rgb_img.ptr, depth_img.ptr, dc1, w, h, K, cloud.points, cloud.normals, cloud.colors, false);

        // Get a depth map back from the point cloud
        cilantro::RigidTransform3f cam_pose;
        pcdv.getCameraPose(cam_pose);
        cilantro::DepthValueConverter<float,float> dc2(1.0f);
        cilantro::pointsToDepthImage<decltype(dc2)>(cloud.points, cam_pose, K, dc2, depthf_img.ptr, w, h);

        rgbv.setImage(rgb_img.ptr, w, h, "RGB24");
        depthv.setImage(depth_img.ptr, w, h, "GRAY16LE");
        pcdv.addObject<cilantro::PointCloudRenderable>("cloud", cloud, rp);
        depthfv.setImage(depthf_img.ptr, w, h, "GRAY32F");

        pcdv.clearRenderArea();
        rgbv.render();
        depthv.render();
        pcdv.render();
        depthfv.render();
        pangolin::FinishFrame();

        // Keep rendering properties on update
        rp = pcdv.getRenderingProperties("cloud");
    }

    delete[] img;

    return 0;
}
