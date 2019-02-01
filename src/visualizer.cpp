#include <cilantro/visualizer.hpp>

#ifdef HAVE_PANGOLIN
namespace cilantro {
    Visualizer::Visualizer() {
        init_("Window", "Display");
    }

    Visualizer::Visualizer(const std::string &window_name, const std::string &display_name) {
        init_(window_name, display_name);
    }

    Visualizer::~Visualizer() {
        gl_context_->MakeCurrent();
    }

    RenderingProperties Visualizer::getRenderingProperties(const std::string &name) const {
        auto it = renderables_.find(name);
        if (it == renderables_.end()) return RenderingProperties();
        return it->second.first->getRenderingProperties();
    }

    Visualizer& Visualizer::setRenderingProperties(const std::string &name, const RenderingProperties &rp) {
        auto it = renderables_.find(name);
        if (it == renderables_.end()) return *this;
        it->second.first->setRenderingProperties(rp);
        return *this;
    }

    bool Visualizer::getVisibility(const std::string &name) const {
        auto it = renderables_.find(name);
        if (it == renderables_.end()) return false;
        return it->second.first->getVisibility();
    }

    Visualizer& Visualizer::setVisibility(const std::string &name, bool visible) {
        auto it = renderables_.find(name);
        if (it == renderables_.end()) return *this;
        it->second.first->setVisibility(visible);
        return *this;
    }

    Visualizer& Visualizer::toggleVisibility(const std::string &name) {
        auto it = renderables_.find(name);
        if (it == renderables_.end()) return *this;
        it->second.first->toggleVisibility();
        return *this;
    }

    Visualizer& Visualizer::clear() {
        gl_context_->MakeCurrent();
        renderables_.clear();
        return *this;
    }

    Visualizer& Visualizer::remove(const std::string &name) {
        gl_context_->MakeCurrent();
        renderables_.erase(name);
        return *this;
    }

    Visualizer& Visualizer::clearRenderArea() {
        gl_context_->MakeCurrent();
        display_->Activate(*gl_render_state_);
        glClearColor(clear_color_(0), clear_color_(1), clear_color_(2), 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        return *this;
    }

    Visualizer& Visualizer::render() {
        // Set render sequence
        pangolin::OpenGlMatrix mv = gl_render_state_->GetModelViewMatrix();
        Eigen::Matrix3f R;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R(i,j) = mv(i,j);
            }
        }
        Eigen::Vector3f t(mv(0,3), mv(1,3), mv(2,3));

        std::vector<std::pair<std::tuple<bool,bool,float>,ManagedRenderable*>> visible_objects;
        visible_objects.reserve(renderables_.size());
        for (auto it = renderables_.begin(); it != renderables_.end(); ++it) {
            ManagedRenderable* obj = &(it->second);
            if (obj->first->getVisibility()) {
                visible_objects.emplace_back(std::make_tuple(!obj->first->getDrawLast(), obj->first->getRenderingProperties().opacity == 1.0f, (R*(obj->first->getCentroid()) + t).squaredNorm()), obj);
            }
        }

        std::sort(visible_objects.begin(), visible_objects.end(), RenderPriorityComparator_());

        gl_context_->MakeCurrent();
        display_->Activate(*gl_render_state_);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        for (size_t i = 0; i < visible_objects.size(); i++) {
            visible_objects[i].second->first->updateGPUBuffersAndRender(*(visible_objects[i].second->second));
        }

        if (video_record_on_render_ && video_recorder_) {
            video_record_on_render_ = false;
            recordVideoFrame();
            video_record_on_render_ = true;
        }

        return *this;
    }

    Visualizer& Visualizer::finishFrame() {
        gl_context_->MakeCurrent();
        pangolin::FinishFrame();
        return *this;
    }

    Visualizer& Visualizer::spinOnce() {
        clearRenderArea();
        render();
        finishFrame();
        return *this;
    }

    Visualizer& Visualizer::spin() {
        while (!gl_context_->quit) {
            spinOnce();
        }
        return *this;
    }

    std::vector<std::string> Visualizer::getObjectNames() const {
        std::vector<std::string> res(renderables_.size());
        size_t k = 0;
        for (auto it = renderables_.begin(); it != renderables_.end(); ++it) {
            res[k++] = it->first;
        }
        return res;
    }

    Visualizer& Visualizer::setPerspectiveProjectionMatrix(size_t w, size_t h,
                                                           pangolin::GLprecision fu, pangolin::GLprecision fv,
                                                           pangolin::GLprecision u0, pangolin::GLprecision v0,
                                                           pangolin::GLprecision zNear, pangolin::GLprecision zFar)
    {
        display_->SetAspect(-(double)w/((double)h));
        input_handler_->SetPerspectiveProjectionMatrix(pangolin::ProjectionMatrix((int)w, (int)h, fu, fv, u0, v0, zNear, zFar));
        return *this;
    }

    Visualizer& Visualizer::setPerspectiveProjectionMatrix(size_t w, size_t h,
                                                           const Eigen::Matrix3f &intrinsics,
                                                           pangolin::GLprecision zNear, pangolin::GLprecision zFar)
    {
        display_->SetAspect(-(double)w/((double)h));
        input_handler_->SetPerspectiveProjectionMatrix(pangolin::ProjectionMatrix((int)w, (int)h, intrinsics(0,0), intrinsics(1,1), intrinsics(0,2), intrinsics(1,2), zNear, zFar));
        return *this;
    }

    Visualizer& Visualizer::setOrthographicProjectionMatrix(pangolin::GLprecision left, pangolin::GLprecision right,
                                                            pangolin::GLprecision bottom, pangolin::GLprecision top,
                                                            pangolin::GLprecision near, pangolin::GLprecision far)
    {
        input_handler_->SetOrthographicProjectionMatrix(left, right, bottom, top, near, far);
        return *this;
    }

    Visualizer& Visualizer::setOrthographicProjectionMatrix(pangolin::GLprecision height,
                                                            pangolin::GLprecision near, pangolin::GLprecision far)
    {
        double aspect = std::abs(display_->aspect);
        input_handler_->SetOrthographicProjectionMatrix(-height*aspect/2.0, height*aspect/2.0, -height/2.0, height/2.0, near, far);
        return *this;
    }

    const Visualizer& Visualizer::getCameraPose(Eigen::Ref<Eigen::Matrix4f> pose) const {
        pose = cam_axes_rot_*pangolin::ToEigen<float>(gl_render_state_->GetModelViewMatrix());
        Eigen::Matrix3f rot = pose.topLeftCorner(3,3).transpose();
        pose.topLeftCorner(3,3) = rot;
        pose.topRightCorner(3,1) = -rot*pose.topRightCorner(3,1);
        return *this;
    }

    const Visualizer& Visualizer::getCameraPose(Eigen::Vector3f &position,
                                                Eigen::Vector3f &look_at,
                                                Eigen::Vector3f &up_direction) const
    {
        Eigen::Matrix4f pose;
        getCameraPose(pose);
        position = pose.topRightCorner(3,1);
        look_at = position + pose.block<3,1>(0,2);
        up_direction = -pose.block<3,1>(0,1);
        return *this;
    }

    Visualizer& Visualizer::setCameraPose(const Eigen::Vector3f &position,
                                          const Eigen::Vector3f &look_at,
                                          const Eigen::Vector3f &up_direction)
    {
        gl_render_state_->SetModelViewMatrix(pangolin::ModelViewLookAt(position(0), position(1), position(2), look_at(0), look_at(1), look_at(2), up_direction(0), up_direction(1), up_direction(2)));
        return *this;
    }

    Visualizer& Visualizer::setCameraPose(float pos_x, float pos_y, float pos_z,
                                          float look_at_x, float look_at_y, float look_at_z,
                                          float up_dir_x, float up_dir_y, float up_dir_z)
    {
        gl_render_state_->SetModelViewMatrix(pangolin::ModelViewLookAt(pos_x, pos_y, pos_z, look_at_x, look_at_y, look_at_z, up_dir_x, up_dir_y, up_dir_z));
        return *this;
    }

    Visualizer& Visualizer::setCameraPose(const Eigen::Ref<const Eigen::Matrix4f> &pose) {
        Eigen::Matrix3f rot = pose.topLeftCorner(3,3).transpose();
        Eigen::Matrix4f model_view;
        model_view.topLeftCorner(3,3) = rot;
        model_view.topRightCorner(3,1) = -rot*pose.topRightCorner(3,1);
        model_view.bottomLeftCorner(1,3).setZero();
        model_view(3,3) = 1.0f;
        model_view = cam_axes_rot_*model_view;
        gl_render_state_->SetModelViewMatrix(model_view);
        return *this;
    }

    const Visualizer& Visualizer::getDefaultCameraPose(Eigen::Ref<Eigen::Matrix4f> pose) const {
        pose = cam_axes_rot_*pangolin::ToEigen<float>(input_handler_->default_model_view);
        Eigen::Matrix3f rot = pose.topLeftCorner(3,3).transpose();
        pose.topLeftCorner(3,3) = rot;
        pose.topRightCorner(3,1) = -rot*pose.topRightCorner(3,1);
        return *this;
    }

    const Visualizer& Visualizer::getDefaultCameraPose(Eigen::Vector3f &position,
                                                       Eigen::Vector3f &look_at,
                                                       Eigen::Vector3f &up_direction) const
    {
        Eigen::Matrix4f pose;
        getDefaultCameraPose(pose);
        position = pose.topRightCorner(3,1);
        look_at = position + pose.block<3,1>(0,2);
        up_direction = -pose.block<3,1>(0,1);
        return *this;
    }

    Visualizer& Visualizer::setDefaultCameraPose(const Eigen::Vector3f &position,
                                                 const Eigen::Vector3f &look_at,
                                                 const Eigen::Vector3f &up_direction)
    {
        input_handler_->default_model_view = pangolin::ModelViewLookAt(position(0), position(1), position(2), look_at(0), look_at(1), look_at(2), up_direction(0), up_direction(1), up_direction(2));
        return *this;
    }

    Visualizer& Visualizer::setDefaultCameraPose(float pos_x, float pos_y, float pos_z,
                                                 float look_at_x, float look_at_y, float look_at_z,
                                                 float up_dir_x, float up_dir_y, float up_dir_z)
    {
        input_handler_->default_model_view = pangolin::ModelViewLookAt(pos_x, pos_y, pos_z, look_at_x, look_at_y, look_at_z, up_dir_x, up_dir_y, up_dir_z);
        return *this;
    }

    Visualizer& Visualizer::setDefaultCameraPose(const Eigen::Ref<const Eigen::Matrix4f> &pose) {
        Eigen::Matrix3f rot = pose.topLeftCorner(3,3).transpose();
        Eigen::Matrix4f model_view;
        model_view.topLeftCorner(3,3) = rot;
        model_view.topRightCorner(3,1) = -rot*pose.topRightCorner(3,1);
        model_view.bottomLeftCorner(1,3).setZero();
        model_view(3,3) = 1.0f;
        model_view = cam_axes_rot_*model_view;
        input_handler_->default_model_view = model_view;
        return *this;
    }

    Visualizer& Visualizer::registerKeyboardCallback(unsigned char key, std::function<void(void)> func) {
        input_handler_->key_callback_map[key] = std::move(func);
        return *this;
    }

    pangolin::TypedImage Visualizer::getRenderImage(float scale, bool rgba) {
        gl_context_->MakeCurrent();

        const pangolin::Viewport orig = display_->v;

        display_->v.l = 0;
        display_->v.b = 0;
        display_->v.w = (int)(display_->v.w * scale);
        display_->v.h = (int)(display_->v.h * scale);

        const int w = display_->v.w;
        const int h = display_->v.h;

        // Create FBO
        pangolin::GlTexture color(w,h);
        pangolin::GlRenderBuffer depth(w,h);
        pangolin::GlFramebuffer fbo(color, depth);

        // Render into FBO
        fbo.Bind();
        display_->Activate();
        clearRenderArea();
        render();
        glFlush();

        if (rgba) {
            const pangolin::PixelFormat fmt = pangolin::PixelFormatFromString("RGBA32");
            pangolin::TypedImage buffer(w, h, fmt);
//        glReadBuffer(GL_BACK);
            glPixelStorei(GL_PACK_ALIGNMENT, 1); // TODO: Avoid this?
            glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buffer.ptr);
            // Flip y
            pangolin::TypedImage image(w, h, fmt);
            for(size_t y_out = 0; y_out < image.h; ++y_out) {
                const size_t y_in = (buffer.h-1) - y_out;
                std::memcpy(image.RowPtr((int)y_out), buffer.RowPtr((int)y_in), 4*buffer.w);
            }
            // unbind FBO
            fbo.Unbind();
            // restore viewport
            display_->v = orig;

            return image;
        } else {
            const pangolin::PixelFormat fmt = pangolin::PixelFormatFromString("RGB24");
            pangolin::TypedImage buffer(w, h, fmt);
//        glReadBuffer(GL_BACK);
            glPixelStorei(GL_PACK_ALIGNMENT, 1); // TODO: Avoid this?
            glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, buffer.ptr);
            // Flip y
            pangolin::TypedImage image(w, h, fmt);
            for(size_t y_out = 0; y_out < image.h; ++y_out) {
                const size_t y_in = (buffer.h-1) - y_out;
                std::memcpy(image.RowPtr((int)y_out), buffer.RowPtr((int)y_in), 3*buffer.w);
            }
            // unbind FBO
            fbo.Unbind();
            // restore viewport
            display_->v = orig;

            return image;
        }
    }

    Visualizer& Visualizer::saveRenderAsImage(const std::string &file_name, float scale, float quality, bool rgba) {
        pangolin::TypedImage image(getRenderImage(scale, rgba));
        SaveImage(image, image.fmt, file_name, true, quality);
        return *this;
    }

    Visualizer& Visualizer::startVideoRecording(const std::string &uri, size_t fps,
                                                bool record_on_render, float scale, bool rgba)
    {
        std::string output_uri;
        if (uri.find(':') == std::string::npos) {
            // Treat as filename
            output_uri = "ffmpeg:[fps=" + std::to_string(fps) + "]//" + uri;
        } else {
            output_uri = uri;
        }

        video_fps_ = fps;
        video_record_on_render_ = record_on_render;
        video_scale_ = scale;
        video_rgba_ = rgba;
        video_recorder_.reset(new pangolin::VideoOutput(output_uri));

        pangolin::PixelFormat fmt(pangolin::PixelFormatFromString((rgba) ? "RGBA32" : "RGB24"));
        const int w = (int)(display_->v.w * scale);
        const int h = (int)(display_->v.h * scale);
        size_t pitch = (size_t)((rgba) ? 4 : 3)*w;

        video_recorder_->SetStreams(std::vector<pangolin::StreamInfo>(1, pangolin::StreamInfo(fmt, w, h, pitch)));

        return *this;
    }

    Visualizer& Visualizer::recordVideoFrame() {
        if (video_recorder_) {
            pangolin::TypedImage img(getRenderImage(video_scale_, video_rgba_));
            // Flip y
            pangolin::TypedImage img_flipped(img.w, img.h, img.fmt);
            for(size_t y_out = 0; y_out < img_flipped.h; ++y_out) {
                const size_t y_in = (img.h-1) - y_out;
                std::memcpy(img_flipped.RowPtr((int)y_out), img.RowPtr((int)y_in), 3*img.w);
            }
            video_recorder_->WriteStreams(img_flipped.ptr);
        }
        return *this;
    }

    Visualizer& Visualizer::stopVideoRecording() {
        if (video_recorder_) {
            video_recorder_->Close();
            video_recorder_.reset();
        }
        video_record_on_render_ = false;
        return *this;
    }

    void Visualizer::init_(const std::string &window_name, const std::string &display_name) {
        gl_context_ = pangolin::FindContext(window_name);
        if (!gl_context_) {
            pangolin::CreateWindowAndBind(window_name);
            gl_context_ = pangolin::FindContext(window_name);
        }
        gl_context_->MakeCurrent();

        // Pangolin searches internally for existing named displays
        display_ = &(pangolin::Display(display_name));
        display_->SetAspect(-4.0f/3.0f);

        gl_render_state_.reset(new pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(640, 480, 525, 525, 319.5, 239.5, 0.1, 100), pangolin::ModelViewLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0)));
        input_handler_.reset(new VisualizerHandler(this));
        display_->SetHandler(input_handler_.get());

        clear_color_ = Eigen::Vector3f(0.7f, 0.7f, 1.0f);
        cam_axes_rot_.setIdentity();
        cam_axes_rot_(1,1) = -1.0f;
        cam_axes_rot_(2,2) = -1.0f;
        video_record_on_render_ = false;
    }
}
#endif
