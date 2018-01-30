#pragma once

#include <cilantro/renderables.hpp>
#include <cilantro/visualizer_handler.hpp>
#include <cilantro/type_traits.hpp>
#include <pangolin/display/display_internal.h>

namespace cilantro {
    class Visualizer {
        friend class VisualizerHandler;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Visualizer();
        Visualizer(const std::string &window_name, const std::string &display_name);
        ~Visualizer();

        template <class CloudT, class = typename std::enable_if<has_points<CloudT>::value && has_normals<CloudT>::value && has_colors<CloudT>::value>::type>
        Visualizer& addPointCloud(const std::string &name, const CloudT &cloud, const RenderingProperties &rp = RenderingProperties()) {
            gl_context_->MakeCurrent();
            if (cloud.points.cols() == 0) {
                renderables_.erase(name);
                return *this;
            }
            renderables_[name] = std::shared_ptr<PointCloudRenderable>(new PointCloudRenderable);
            PointCloudRenderable *obj_ptr = (PointCloudRenderable *)renderables_[name].get();
            // Copy fields
            obj_ptr->points = cloud.points;
            if (cloud.normals.cols() == cloud.points.cols()) obj_ptr->normals = cloud.normals;
            if (cloud.colors.cols() ==  cloud.points.cols()) obj_ptr->colors = cloud.colors;
            obj_ptr->centroid = cloud.points.rowwise().mean();
            // Update buffers
            ((Renderable *)obj_ptr)->applyRenderingProperties(rp);

            return *this;
        }
        Visualizer& addPointCloud(const std::string &name, const ConstVectorSetMatrixMap<float,3> &points, const RenderingProperties &rp = RenderingProperties());
        Visualizer& addPointCloudNormals(const std::string &name, const ConstVectorSetMatrixMap<float,3> &normals);
        Visualizer& addPointCloudColors(const std::string &name, const ConstVectorSetMatrixMap<float,3> &colors);
        Visualizer& addPointCloudValues(const std::string &name, const ConstVectorSetMatrixMap<float,1> &point_values);

        Visualizer& addPointCorrespondences(const std::string &name, const ConstVectorSetMatrixMap<float,3> &points_src, const ConstVectorSetMatrixMap<float,3> &points_dst, const RenderingProperties &rp = RenderingProperties());
        template <class CloudT, class = typename std::enable_if<has_points<CloudT>::value>::type>
        Visualizer& addPointCorrespondences(const std::string &name, const CloudT &cloud_src, const CloudT &cloud_dst, const RenderingProperties &rp = RenderingProperties()) {
            return addPointCorrespondences(name, cloud_src.points, cloud_dst.points, rp);
        }

        Visualizer& addCoordinateFrame(const std::string &name, const Eigen::Matrix4f &tf = Eigen::Matrix4f::Identity(), float scale = 1.0f, const RenderingProperties &rp = RenderingProperties());

        Visualizer& addCameraFrustum(const std::string &name, size_t width, size_t height, const Eigen::Matrix3f &intrinsics, const Eigen::Matrix4f &pose = Eigen::Matrix4f::Identity(), float scale = 1.0f, const RenderingProperties &rp = RenderingProperties());

        template <class CloudT, class = typename std::enable_if<has_points<CloudT>::value && has_normals<CloudT>::value && has_colors<CloudT>::value>::type>
        Visualizer& addTriangleMesh(const std::string &name, const CloudT &cloud, const std::vector<std::vector<size_t>> &faces, const RenderingProperties &rp = RenderingProperties()) {
            gl_context_->MakeCurrent();
            if (cloud.points.cols() == 0 || faces.empty()) {
                renderables_.erase(name);
                return *this;
            }
            renderables_[name] = std::shared_ptr<TriangleMeshRenderable>(new TriangleMeshRenderable);
            TriangleMeshRenderable *obj_ptr = (TriangleMeshRenderable *)renderables_[name].get();
            obj_ptr->vertices = cloud.points;
            obj_ptr->faces = faces;
            if (cloud.normals.cols() == cloud.points.cols()) obj_ptr->vertexNormals = cloud.normals;
            if (cloud.colors.cols() ==  cloud.points.cols()) obj_ptr->vertexColors = cloud.colors;
            obj_ptr->faceNormals.resize(faces.size());
            for (size_t i = 0; i < faces.size(); i++) {
                const Eigen::Vector3f& pt0(obj_ptr->vertices.col(faces[i][0]));
                const Eigen::Vector3f& pt1(obj_ptr->vertices.col(faces[i][1]));
                const Eigen::Vector3f& pt2(obj_ptr->vertices.col(faces[i][2]));
                obj_ptr->faceNormals.col(i) = ((pt1-pt0).cross(pt2-pt0)).normalized();
            }
            ((Renderable *)obj_ptr)->applyRenderingProperties(rp);

            return *this;
        }
        Visualizer& addTriangleMesh(const std::string &name, const ConstVectorSetMatrixMap<float,3> &vertices, const std::vector<std::vector<size_t>> &faces, const RenderingProperties &rp = RenderingProperties());
        Visualizer& addTriangleMeshVertexNormals(const std::string &name, const ConstVectorSetMatrixMap<float,3> &vertex_normals);
        Visualizer& addTriangleMeshFaceNormals(const std::string &name, const ConstVectorSetMatrixMap<float,3> &face_normals);
        Visualizer& addTriangleMeshVertexColors(const std::string &name, const ConstVectorSetMatrixMap<float,3> &vertex_colors);
        Visualizer& addTriangleMeshFaceColors(const std::string &name, const ConstVectorSetMatrixMap<float,3> &face_colors);
        Visualizer& addTriangleMeshVertexValues(const std::string &name, const ConstVectorSetMatrixMap<float,1> &vertex_values);
        Visualizer& addTriangleMeshFaceValues(const std::string &name, const ConstVectorSetMatrixMap<float,1> &face_values);

        Visualizer& addText(const std::string &name, const std::string &text, const Eigen::Vector3f &position, const RenderingProperties &rp = RenderingProperties());
        Visualizer& addText(const std::string &name, const std::string &text, float x, float y, float z, const RenderingProperties &rp = RenderingProperties());

        inline Visualizer& clear() { gl_context_->MakeCurrent(); renderables_.clear(); return *this; }
        inline Visualizer& remove(const std::string &name) { gl_context_->MakeCurrent(); renderables_.erase(name); return *this; }

        Visualizer& clearRenderArea();
        Visualizer& render();
        inline Visualizer& finishFrame() { gl_context_->MakeCurrent(); pangolin::FinishFrame(); return *this; }
        inline Visualizer& spinOnce() { clearRenderArea(); render(); finishFrame(); return *this; }

        inline bool wasStopped() const { return gl_context_->quit; }

        bool getVisibilityStatus(const std::string &name) const;
        Visualizer& setVisibilityStatus(const std::string &name, bool visible);
        inline Visualizer& toggleVisibilityStatus(const std::string &name) { setVisibilityStatus(name, !getVisibilityStatus(name)); return *this; }

        RenderingProperties getRenderingProperties(const std::string &name) const;
        Visualizer& setRenderingProperties(const std::string &name, const RenderingProperties &rp);

        std::vector<std::string> getObjectNames() const;

        inline Eigen::Vector3f getClearColor() const { return clear_color_; }
        inline Visualizer& setClearColor(const Eigen::Vector3f &color) { clear_color_ = color; return *this; }
        inline Visualizer& setClearColor(float r, float g, float b) { clear_color_ = Eigen::Vector3f(r,g,b); return *this; }

        Visualizer& setPerspectiveProjectionMatrix(size_t w, size_t h, pangolin::GLprecision fu, pangolin::GLprecision fv, pangolin::GLprecision u0, pangolin::GLprecision v0, pangolin::GLprecision zNear, pangolin::GLprecision zFar);
        Visualizer& setPerspectiveProjectionMatrix(size_t w, size_t h, const Eigen::Matrix3f &intrinsics, pangolin::GLprecision zNear, pangolin::GLprecision zFar);

        Visualizer& setOrthographicProjectionMatrix(pangolin::GLprecision left, pangolin::GLprecision right, pangolin::GLprecision bottom, pangolin::GLprecision top, pangolin::GLprecision near, pangolin::GLprecision far);
        Visualizer& setOrthographicProjectionMatrix(pangolin::GLprecision height, pangolin::GLprecision near, pangolin::GLprecision far);

        inline Visualizer& enablePerspectiveProjection() { input_handler_->EnablePerspectiveProjection(); return *this; }
        inline Visualizer& enableOrthographicProjection() { input_handler_->EnableOrthographicProjection(); return *this; }
        inline Visualizer& toggleProjectionMode() { input_handler_->ToggleProjectionMode(); return *this; }

        Eigen::Matrix4f getCameraPose() const;
        const Visualizer& getCameraPose(Eigen::Vector3f &position, Eigen::Vector3f &look_at, Eigen::Vector3f &up_direction) const;
        Visualizer& setCameraPose(const Eigen::Vector3f &position, const Eigen::Vector3f &look_at, const Eigen::Vector3f &up_direction);
        Visualizer& setCameraPose(float pos_x, float pos_y, float pos_z, float look_at_x, float look_at_y, float look_at_z, float up_dir_x, float up_dir_y, float up_dir_z);
        Visualizer& setCameraPose(const Eigen::Matrix4f &pose);

        Eigen::Matrix4f getDefaultCameraPose() const;
        const Visualizer& getDefaultCameraPose(Eigen::Vector3f &position, Eigen::Vector3f &look_at, Eigen::Vector3f &up_direction) const;
        Visualizer& setDefaultCameraPose(const Eigen::Vector3f &position, const Eigen::Vector3f &look_at, const Eigen::Vector3f &up_direction);
        Visualizer& setDefaultCameraPose(float pos_x, float pos_y, float pos_z, float look_at_x, float look_at_y, float look_at_z, float up_dir_x, float up_dir_y, float up_dir_z);
        Visualizer& setDefaultCameraPose(const Eigen::Matrix4f &pose);

        Visualizer& registerKeyboardCallback(unsigned char key, std::function<void(void)> func);

        pangolin::TypedImage getRenderImage(float scale = 1.0f, bool rgba = false);
        Visualizer& saveRenderAsImage(const std::string &file_name, float scale, float quality, bool rgba = false);

        Visualizer& startVideoRecording(const std::string &uri, size_t fps, bool record_on_render = false, float scale = 1.0f, bool rgba = false);
        Visualizer& recordVideoFrame();
        Visualizer& stopVideoRecording();
        inline bool isRecording() const { return !!video_recorder_; }

        inline pangolin::PangolinGl* getPangolinGLContext() const { return gl_context_; }
        inline pangolin::View* getPangolinDisplay() const { return display_; }

        inline pangolin::OpenGlRenderState* getPangolinRenderState() const { return gl_render_state_.get(); }
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

        std::map<std::string, std::shared_ptr<Renderable>> renderables_;

        void init_(const std::string &window_name, const std::string &display_name);

        static inline bool render_priority_comparator_(const std::pair<std::tuple<bool,bool,float>, Renderable*> &o1, const std::pair<std::tuple<bool,bool,float>, Renderable*> &o2) {
            return o1.first > o2.first;
        }
    };
}
