#pragma once

#include <cilantro/config.hpp>

#ifdef HAVE_PANGOLIN
#include <utility>
#include <pangolin/display/display_internal.h>
#include <cilantro/visualization/renderable.hpp>
#include <cilantro/visualization/visualizer_handler.hpp>
#include <cilantro/core/space_transformations.hpp>

namespace cilantro {
    class Visualizer {
        friend class VisualizerHandler;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Visualizer();

        Visualizer(const std::string &window_name, const std::string &display_name);

        ~Visualizer();

        template <class RenderableT>
        std::shared_ptr<RenderableT> addObject(const std::string &name, const std::shared_ptr<RenderableT> &obj_ptr) {
            gl_context_->MakeCurrent();
            renderables_[name] = ManagedRenderable(std::static_pointer_cast<Renderable>(obj_ptr), std::shared_ptr<typename RenderableT::GPUBuffers>(new typename RenderableT::GPUBuffers()));
            obj_ptr->resetGPUBufferStatus();
            return obj_ptr;
        }

        template <class RenderableT, class... Args>
        std::shared_ptr<RenderableT> addObject(const std::string &name, Args&&... args) {
            gl_context_->MakeCurrent();
            std::shared_ptr<RenderableT> obj_ptr(new RenderableT(std::forward<Args>(args)...));
            renderables_[name] = ManagedRenderable(std::static_pointer_cast<Renderable>(obj_ptr), std::shared_ptr<typename RenderableT::GPUBuffers>(new typename RenderableT::GPUBuffers()));
            return obj_ptr;
        }

        template <class RenderableT = Renderable>
        std::shared_ptr<RenderableT> getObject(const std::string &name) {
            auto it = renderables_.find(name);
            if (it == renderables_.end()) return std::shared_ptr<RenderableT>();
            return std::dynamic_pointer_cast<RenderableT>(it->second.first);
        }

        RenderingProperties getRenderingProperties(const std::string &name) const;

        Visualizer& setRenderingProperties(const std::string &name, const RenderingProperties &rp);

        bool getVisibility(const std::string &name) const;

        Visualizer& setVisibility(const std::string &name, bool visible);

        Visualizer& toggleVisibility(const std::string &name);

        Visualizer& clear();

        Visualizer& remove(const std::string &name);

        Visualizer& clearRenderArea();

        Visualizer& render();

        Visualizer& finishFrame();

        Visualizer& spinOnce();

        Visualizer& spin();

        inline bool wasStopped() const { return gl_context_->quit; }

        std::vector<std::string> getObjectNames() const;

        inline Eigen::Vector3f getClearColor() const { return clear_color_; }

        inline Visualizer& setClearColor(const Eigen::Vector3f &color) { clear_color_ = color; return *this; }

        inline Visualizer& setClearColor(float r, float g, float b) { clear_color_ << r, g, b; return *this; }

        Visualizer& setPerspectiveProjectionMatrix(size_t w, size_t h,
                                                   pangolin::GLprecision fu, pangolin::GLprecision fv,
                                                   pangolin::GLprecision u0, pangolin::GLprecision v0,
                                                   pangolin::GLprecision zNear, pangolin::GLprecision zFar);

        Visualizer& setPerspectiveProjectionMatrix(size_t w, size_t h,
                                                   const Eigen::Matrix3f &intrinsics,
                                                   pangolin::GLprecision zNear, pangolin::GLprecision zFar);

        Visualizer& setOrthographicProjectionMatrix(pangolin::GLprecision left, pangolin::GLprecision right,
                                                    pangolin::GLprecision bottom, pangolin::GLprecision top,
                                                    pangolin::GLprecision near, pangolin::GLprecision far);

        Visualizer& setOrthographicProjectionMatrix(pangolin::GLprecision height,
                                                    pangolin::GLprecision near, pangolin::GLprecision far);

        inline Visualizer& enablePerspectiveProjection() {
            input_handler_->EnablePerspectiveProjection();
            return *this;
        }

        inline Visualizer& enableOrthographicProjection() {
            input_handler_->EnableOrthographicProjection();
            return *this;
        }

        inline Visualizer& toggleProjectionMode() { input_handler_->ToggleProjectionMode(); return *this; }

        const Visualizer& getCameraPose(Eigen::Ref<Eigen::Matrix4f> pose) const;

        inline const Visualizer& getCameraPose(RigidTransform3f &pose) const {
            return getCameraPose(pose.matrix());
        }

        const Visualizer& getCameraPose(Eigen::Vector3f &position,
                                        Eigen::Vector3f &look_at,
                                        Eigen::Vector3f &up_direction) const;

        Visualizer& setCameraPose(const Eigen::Vector3f &position,
                                  const Eigen::Vector3f &look_at,
                                  const Eigen::Vector3f &up_direction);

        Visualizer& setCameraPose(float pos_x, float pos_y, float pos_z,
                                  float look_at_x, float look_at_y, float look_at_z,
                                  float up_dir_x, float up_dir_y, float up_dir_z);

        Visualizer& setCameraPose(const Eigen::Ref<const Eigen::Matrix4f> &pose);

        inline Visualizer& setCameraPose(const RigidTransform3f &pose) {
            return setCameraPose(pose.matrix());
        }

        const Visualizer& getDefaultCameraPose(Eigen::Ref<Eigen::Matrix4f> pose) const;

        inline const Visualizer& getDefaultCameraPose(RigidTransform3f &pose) const {
            return getDefaultCameraPose(pose.matrix());
        }

        const Visualizer& getDefaultCameraPose(Eigen::Vector3f &position,
                                               Eigen::Vector3f &look_at,
                                               Eigen::Vector3f &up_direction) const;

        Visualizer& setDefaultCameraPose(const Eigen::Vector3f &position,
                                         const Eigen::Vector3f &look_at,
                                         const Eigen::Vector3f &up_direction);

        Visualizer& setDefaultCameraPose(float pos_x, float pos_y, float pos_z,
                                         float look_at_x, float look_at_y, float look_at_z,
                                         float up_dir_x, float up_dir_y, float up_dir_z);

        Visualizer& setDefaultCameraPose(const Eigen::Ref<const Eigen::Matrix4f> &pose);

        inline Visualizer& setDefaultCameraPose(const RigidTransform3f &pose) {
            return setDefaultCameraPose(pose.matrix());
        }

        Visualizer& registerKeyboardCallback(unsigned char key, std::function<void(void)> func);

        pangolin::TypedImage getRenderImage(float scale = 1.0f, bool rgba = false);

        Visualizer& saveRenderAsImage(const std::string &file_name, float scale, float quality, bool rgba = false);

        Visualizer& startVideoRecording(const std::string &uri, size_t fps,
                                        bool record_on_render = false, float scale = 1.0f, bool rgba = false);

        Visualizer& recordVideoFrame();

        Visualizer& stopVideoRecording();

        inline bool isRecording() const { return !!video_recorder_; }

        inline pangolin::PangolinGl* getGLContext() const { return gl_context_; }

        inline pangolin::View* getDisplay() const { return display_; }

        inline const std::shared_ptr<pangolin::OpenGlRenderState>& getRenderState() const { return gl_render_state_; }

        inline Visualizer& setRenderState(const std::shared_ptr<pangolin::OpenGlRenderState>& render_state) {
            gl_render_state_ = render_state;
            return *this;
        }

        inline VisualizerHandler* getInputHandler() const { return input_handler_.get(); }

    private:
        pangolin::PangolinGl *gl_context_;
        pangolin::View *display_;

        std::shared_ptr<pangolin::OpenGlRenderState> gl_render_state_;
        std::shared_ptr<VisualizerHandler> input_handler_;
        std::shared_ptr<pangolin::VideoOutput> video_recorder_;
        size_t video_fps_;
        float video_scale_;
        bool video_rgba_;
        bool video_record_on_render_;
        Eigen::Vector3f clear_color_;
        Eigen::Matrix4f cam_axes_rot_;

        typedef std::pair<std::shared_ptr<Renderable>,std::shared_ptr<GPUBufferObjects>> ManagedRenderable;

        std::map<std::string,ManagedRenderable> renderables_;

        void init_(const std::string &window_name, const std::string &display_name);

        struct RenderPriorityComparator_ {
            inline bool operator()(const std::pair<std::tuple<bool,bool,float>,ManagedRenderable*> &o1,
                                   const std::pair<std::tuple<bool,bool,float>,ManagedRenderable*> &o2) const
            {
                return o1.first > o2.first;
            }
        };
    };
}
#endif
