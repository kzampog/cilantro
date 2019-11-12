#include <cilantro/utilities/point_cloud.hpp>
#include <cilantro/visualization.hpp>
#include <cilantro/registration/icp_common_instances.hpp>

template <typename T>
void vec_remove(std::vector<T> &vec, const std::vector<size_t> &indices) {
    if (indices.empty()) return;

    std::set<size_t> indices_set(indices.begin(), indices.end());
    if (indices_set.size() >= vec.size()) {
        vec.clear();
        return;
    }

    size_t valid_ind = vec.size() - 1;
    while (indices_set.find(valid_ind) != indices_set.end()) {
        valid_ind--;
    }

    auto ind_it = indices_set.begin();
    while (ind_it != indices_set.end() && *ind_it < valid_ind) {
        std::swap(vec[*ind_it], vec[valid_ind]);
        valid_ind--;
        while (*ind_it < valid_ind && indices_set.find(valid_ind) != indices_set.end()) {
            valid_ind--;
        }
        ++ind_it;
    }

    vec.resize(valid_ind + 1);
}

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

void clear_callback(cilantro::PointCloud3f &cloud, std::vector<float> &confidence) {
    cloud.clear();
    confidence.clear();
}

void cleanup_callback(cilantro::PointCloud3f &model, std::vector<float> &confidence, float conf_thresh) {
    std::vector<size_t> remove_ind;
    for (size_t i = 0; i < confidence.size(); i++) {
        if (confidence[i] < conf_thresh) remove_ind.emplace_back(i);
    }
    model.remove(remove_ind);
    vec_remove(confidence, remove_ind);
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

    std::string win_name = "Fusion demo";
    pangolin::CreateWindowAndBind(win_name, 2*w, h);
    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual)
            .AddDisplay(pangolin::Display("disp1")).AddDisplay(pangolin::Display("disp2"));

    cilantro::Visualizer pcdv(win_name, "disp1");
    cilantro::ImageViewer rgbv(win_name, "disp2");

    cilantro::PointCloud3f model, frame;
    std::vector<float> confidence;
    cilantro::RigidTransform3f cam_pose(cilantro::RigidTransform3f::Identity());
    bool capture = false;
    cilantro::RenderingProperties rp;
    rp.setPointColor(0.8f, 0.8f, 0.8f);

    // Parameters
    float max_depth = 1.8f;
    float fusion_dist_thresh = 0.01f;
    float occlusion_dist_thresh = 0.025f;
    float radial_factor = -0.5f/(120*120);
    float confidence_thresh = 3.0f;

    pcdv.registerKeyboardCallback('a', std::bind(capture_callback, std::ref(capture)));
    pcdv.registerKeyboardCallback('s', std::bind(cleanup_callback, std::ref(model), std::ref(confidence), confidence_thresh));
    pcdv.registerKeyboardCallback('d', std::bind(clear_callback, std::ref(model), std::ref(confidence)));
    pcdv.registerKeyboardCallback('c', std::bind(color_toggle_callback, std::ref(pcdv), std::ref(rp)));

    cilantro::TruncatedDepthValueConverter<unsigned short,float> dc(1000.0f, max_depth);

    if (argc < 2) std::cout << "Note: no output PLY file path provided" << std::endl;
    std::cout << "Highlight the left viewport and:" << std::endl;
    std::cout << "\tPress 'a' to initialize model/fuse new view (keep pressed for continuous fusion)" << std::endl;
    std::cout << "\tPress 's' to remove unstable points" << std::endl;
    std::cout << "\tPress 'd' to reinitialize process" << std::endl;
    std::cout << "\tPress 'c' to toggle model color" << std::endl;
    std::cout << "\tPress 'l' to toggle lighting" << std::endl;

    size_t num_fused = 0;

    // Main loop
    while (!pangolin::ShouldQuit()) {
        dok->GrabNext(img, true);
        frame.fromRGBDImages(rgb_img.ptr, depth_img.ptr, dc, w, h, K, false, true);

        // Localize
        if (!model.isEmpty()) {
            cilantro::SimpleCombinedMetricRigidProjectiveICP3f icp(model.points, model.normals, frame.points, frame.normals);
            icp.correspondenceSearchEngine().setProjectionExtrinsicMatrix(cam_pose).setMaxDistance(0.1f*0.1f)
                    .setProjectionImageWidth(w).setProjectionImageHeight(h).setProjectionIntrinsicMatrix(K);
            icp.setInitialTransform(cam_pose).setConvergenceTolerance(5e-4f);
            icp.setMaxNumberOfIterations(6).setMaxNumberOfOptimizationStepIterations(1);
            cam_pose = icp.estimate().getTransform();
        } else {
            cam_pose.setIdentity();
            num_fused = 0;
        }

        // Map
        if (capture) {
            capture = false;

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
            std::vector<float> to_append_confidence;
            to_append_confidence.reserve(w*h);
            std::vector<size_t> remove_ind;
            remove_ind.reserve(w*h);
            const size_t empty = std::numeric_limits<std::size_t>::max();

            for (size_t y = 1; y < frame_index_map.h - 1; y++) {
                for (size_t x = 1; x < frame_index_map.w - 1; x++) {
                    const size_t frame_pt_ind = frame_index_map(x,y);
                    const size_t model_pt_ind = model_index_map(x,y);

                    if (frame_pt_ind == empty) {
                        // Nothing to do
                        continue;
                    }

                    const float frame_depth = frame.points(2,frame_pt_ind);
                    const float model_depth = (model_pt_ind != empty) ? model_t.points(2,model_pt_ind) : 0.0f;
                    const float radial_weight = std::exp(radial_factor*((x - K(0,2))*(x - K(0,2)) + (y - K(1,2))*(y - K(1,2))));

                    if (model_pt_ind != empty &&
                        std::abs(model_depth - frame_depth) < fusion_dist_thresh &&
                        std::acos(std::min(1.0f, std::max(-1.0f, model_t.normals.col(model_pt_ind).dot(frame.normals.col(frame_pt_ind))))) < 75.0f*M_PI/180.0f)
                    {
                        // Fuse
                        const float weight = radial_weight/(radial_weight + confidence[model_pt_ind]);
                        const float weight_compl = 1.0f - weight;
                        model.points.col(model_pt_ind) = weight_compl*model.points.col(model_pt_ind) + weight*frame_t.points.col(frame_pt_ind);
                        model.normals.col(model_pt_ind) = (weight_compl*model.normals.col(model_pt_ind) + weight*frame_t.normals.col(frame_pt_ind)).normalized();
                        model.colors.col(model_pt_ind) = weight_compl*model.colors.col(model_pt_ind) + weight*frame_t.colors.col(frame_pt_ind);
                        confidence[model_pt_ind] += weight;
                    } else if ((model_pt_ind == empty &&
                                model_index_map(x-1,y) == empty && model_index_map(x+1,y) == empty &&
                                model_index_map(x,y-1) == empty && model_index_map(x,y+1) == empty) ||
                               (model_pt_ind != empty &&
                                std::acos(std::min(1.0f, std::max(-1.0f, model_t.normals.col(model_pt_ind).dot(frame.normals.col(frame_pt_ind))))) > 105.0f*M_PI/180.0f))
                    {
                        // Augment model
                        to_append.points.col(append_count) = frame_t.points.col(frame_pt_ind);
                        to_append.normals.col(append_count) = frame_t.normals.col(frame_pt_ind);
                        to_append.colors.col(append_count) = frame_t.colors.col(frame_pt_ind);
                        to_append_confidence.emplace_back(radial_weight);
                        append_count++;
                    } else if (model_pt_ind != empty && frame_depth > model_depth + occlusion_dist_thresh &&
                               std::acos(std::min(1.0f, std::max(-1.0f, -model_t.points.col(model_pt_ind).normalized().dot(model_t.normals.col(model_pt_ind))))) < 45.0f*M_PI/180.0f)
                    {
                        // Remove points in free space
                        remove_ind.emplace_back(model_pt_ind);
                    }
                }
            }
            model.remove(remove_ind);
            vec_remove(confidence, remove_ind);
            to_append.points.conservativeResize(Eigen::NoChange, append_count);
            to_append.normals.conservativeResize(Eigen::NoChange, append_count);
            to_append.colors.conservativeResize(Eigen::NoChange, append_count);
            model.append(to_append);
            confidence.insert(confidence.end(), to_append_confidence.begin(), to_append_confidence.end());

            num_fused++;
        }

        // Visualization
        rgbv.setImage(rgb_img.ptr, w, h, "RGB24");
//        pcdv.addObject<cilantro::PointCloudRenderable>("model", model, rp)->setPointValues(confidence);
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

    std::cout << "Fused " << num_fused << " frames" << std::endl;

    if (argc > 1) {
        std::cout << "Removing unstable points" << std::endl;
        cleanup_callback(model, confidence, confidence_thresh);
        std::cout << "Saving model to \'" << argv[1] << "\'" << std::endl;
        model.toPLYFile(argv[1], true);
    }

    delete[] img;

    return 0;
}
