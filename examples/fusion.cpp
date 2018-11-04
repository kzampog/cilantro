#include <cilantro/point_cloud.hpp>
#include <cilantro/image_viewer.hpp>
#include <cilantro/visualizer.hpp>
#include <cilantro/common_renderables.hpp>
#include <cilantro/icp_common_instances.hpp>

void color_toggle_callback(cilantro::Visualizer &viz, cilantro::RenderingProperties &rp) {
    if (rp.pointColor == cilantro::RenderingProperties::noColor) {
        rp.setPointColor(0.8f, 0.8f, 0.8f);
    } else {
        rp.setPointColor(cilantro::RenderingProperties::noColor);
    }
    viz.setRenderingProperties("model", rp);
}

void capture_callback(bool &capture) {
    capture = true;
}

void clear_callback(cilantro::PointCloud3f &cloud) {
    cloud.clear();
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

    cilantro::DepthValueConverter<unsigned short,float> dc(1000.0f);

    std::string win_name = "Fusion demo";
    pangolin::CreateWindowAndBind(win_name, 2*w, h);
    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual)
            .AddDisplay(pangolin::Display("disp1")).AddDisplay(pangolin::Display("disp2"));

    cilantro::Visualizer pcdv(win_name, "disp1");
    cilantro::ImageViewer rgbv(win_name, "disp2");

    cilantro::PointCloud3f model, frame;
    bool capture = false;
    pcdv.registerKeyboardCallback('a', std::bind(capture_callback, std::ref(capture)));
    pcdv.registerKeyboardCallback('d', std::bind(clear_callback, std::ref(model)));

    cilantro::RenderingProperties rp;
    pcdv.registerKeyboardCallback('c', std::bind(color_toggle_callback, std::ref(pcdv), std::ref(rp)));
    rp.setUseLighting(false);

    cilantro::RigidTransformation3f cam_pose(cilantro::RigidTransformation3f::Identity());

    float max_depth = 1.1f;
    float fusion_weight = 0.1f;
    float fusion_dist_thresh = 0.025f;
    float factor = -0.5f/(150*150);

    std::cout << "Press 'a' to initialize model/fuse new view" << std::endl;
    std::cout << "Press 'd' to reinitialize process" << std::endl;
    std::cout << "Press 'c' to toggle model color" << std::endl;

    // Main loop
    while (!pangolin::ShouldQuit()) {
        dok->GrabNext(img, true);
        frame.fromRGBDImages(rgb_img.ptr, depth_img.ptr, dc, w, h, K, false, true);

        // Localize
        if (!model.isEmpty()) {
//            cilantro::SimpleCombinedMetricRigidProjectiveICP3f icp(frame.points, frame.normals, model.points);
//            icp.correspondenceSearchEngine().setMaxDistance(0.1f*0.1f);
//            icp.setInitialTransformation(cam_pose.inverse()).setConvergenceTolerance(5e-4f);
//            icp.setMaxNumberOfIterations(6).setMaxNumberOfOptimizationStepIterations(1);
//            cam_pose = icp.estimateTransformation().getTransformation().inverse();
            cilantro::SimpleCombinedMetricRigidProjectiveICP3f icp(model.points, model.normals, frame.points);
            icp.correspondenceSearchEngine().setProjectionExtrinsicMatrix(cam_pose).setMaxDistance(0.1f*0.1f);
            icp.setInitialTransformation(cam_pose).setConvergenceTolerance(5e-4f);
            icp.setMaxNumberOfIterations(6).setMaxNumberOfOptimizationStepIterations(1);
            cam_pose = icp.estimateTransformation().getTransformation();
        }

        // Map
        if (capture) {
            capture = false;

            if (model.isEmpty()) {
                model.fromRGBDImages(rgb_img.ptr, depth_img.ptr, dc, w, h, K, false, true);
                std::vector<size_t> remove_ind;
                remove_ind.reserve(model.size());
                for (size_t i = 0; i < model.size(); i++) {
                    if (model.points(2,i) > max_depth) remove_ind.emplace_back(i);
                }
                model.remove(remove_ind);
                cam_pose.setIdentity();
            } else {
                cilantro::PointCloud3f frame_t(frame.transformed(cam_pose));
                cilantro::PointCloud3f model_t(model.transformed(cam_pose.inverse()));

                // Project cloud to index image maps
                pangolin::ManagedImage<size_t> model_index_map(w, h);
                cilantro::pointsToIndexMap<float>(model_t.points, K, model_index_map.ptr, w, h);
                pangolin::ManagedImage<size_t> frame_index_map(w, h);
                cilantro::pointsToIndexMap<float>(frame.points, K, frame_index_map.ptr, w, h);

                cilantro::PointCloud3f to_append;
                to_append.points.resize(Eigen::NoChange, w*h);
                to_append.normals.resize(Eigen::NoChange, w*h);
                to_append.colors.resize(Eigen::NoChange, w*h);

                size_t append_count = 0;
                std::vector<size_t> remove_ind;
                remove_ind.reserve(w*h);
                const size_t empty = std::numeric_limits<std::size_t>::max();
#pragma omp parallel for
                for (size_t y = 0; y < frame_index_map.h; y++) {
                    for (size_t x = 0; x < frame_index_map.w; x++) {
                        const size_t frame_pt_ind = frame_index_map(x,y);
                        const size_t model_pt_ind = model_index_map(x,y);

                        if (frame_pt_ind == empty) {
                            // Nothing to do
                            continue;
                        }

                        const float frame_depth = frame.points(2,frame_pt_ind);
                        const float model_depth = (model_pt_ind != empty) ? model_t.points(2,model_pt_ind) : 0.0f;

                        if (model_pt_ind != empty && frame_depth > model_depth + fusion_dist_thresh) {
                            // Remove points in free space
#pragma omp critical
                            remove_ind.emplace_back(model_pt_ind);
                        }

                        if ((model_pt_ind == empty || frame_depth < model_depth - fusion_dist_thresh) && frame.points(2,frame_pt_ind) < max_depth)
                        {
                            // Augment model
#pragma omp critical
                            {
                                to_append.points.col(append_count) = frame_t.points.col(frame_pt_ind);
                                to_append.normals.col(append_count) = frame_t.normals.col(frame_pt_ind);
                                to_append.colors.col(append_count) = frame_t.colors.col(frame_pt_ind);
                                append_count++;
                            }
                        }

                        if (model_pt_ind != empty && std::abs(model_depth - frame_depth) < fusion_dist_thresh) {
                            // Fuse
                            const float weight = fusion_weight*std::exp(factor*((x - K(0,2))*(x - K(0,2)) + (y - K(1,2))*(y - K(1,2))));
                            const float weight_compl = 1.0f - weight;
#pragma omp critical
                            {
                                model.points.col(model_pt_ind) = weight_compl*model.points.col(model_pt_ind) + weight*frame_t.points.col(frame_pt_ind);
                                model.normals.col(model_pt_ind) = (weight_compl*model.normals.col(model_pt_ind) + weight*frame_t.normals.col(frame_pt_ind)).normalized();
                                model.colors.col(model_pt_ind) = weight_compl*model.colors.col(model_pt_ind) + weight*frame_t.colors.col(frame_pt_ind);
                            }
                        }
                    }
                }
                model.remove(remove_ind);
                to_append.points.conservativeResize(Eigen::NoChange, append_count);
                to_append.normals.conservativeResize(Eigen::NoChange, append_count);
                to_append.colors.conservativeResize(Eigen::NoChange, append_count);
                model.append(to_append);
            }
        }

        // Visualization
        rgbv.setImage(rgb_img.ptr, w, h, "RGB24");
        pcdv.addObject<cilantro::PointCloudRenderable>("model", model, rp);
        pcdv.addObject<cilantro::CameraFrustumRenderable>("cam", w, h, K, cam_pose.matrix(), 0.1f, cilantro::RenderingProperties().setLineWidth(2.0f).setLineColor(1.0f,1.0f,0.0f));
//        pcdv.addObject<cilantro::PointCloudRenderable>("frame", frame.transformed(cam_pose), cilantro::RenderingProperties().setOpacity(0.2f).setPointColor(0.8f, 0.8f, 0.8f).setUseLighting(false));
//        pcdv.setCameraPose(cam_pose);

        pcdv.clearRenderArea();
        rgbv.render();
        pcdv.render();
        pangolin::FinishFrame();

        // Keep model rendering properties on update
        rp = pcdv.getRenderingProperties("model");
    }

    if (argc > 1) model.toPLYFile(argv[1], true);

    delete[] img;

    return 0;
}
