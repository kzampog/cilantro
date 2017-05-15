#pragma once

#include <cilantro/point_cloud.hpp>
#include <pangolin/pangolin.h>
#include <pangolin/display/display_internal.h>

class Visualizer {
public:
    struct RenderingProperties {
        inline RenderingProperties() : drawingColor(Eigen::Vector3f(1.0f, 1.0f, 1.0f)),
                                       pointSize(5.0f),
                                       lineWidth(1.0f),
                                       opacity(1.0f),
                                       normalLength(0.05f),
                                       correspondencesFraction(0.5),
                                       overrideColors(false)
        {}
        inline ~RenderingProperties() {}

        Eigen::Vector3f drawingColor;
        float pointSize;
        float lineWidth;
        float opacity;
        float normalLength;
        float correspondencesFraction;
        bool overrideColors;

        inline RenderingProperties& setDrawingColor(const Eigen::Vector3f &color) { drawingColor = color; return *this; }
        inline RenderingProperties& setDrawingColor(float r, float g, float b) { drawingColor = Eigen::Vector3f(r,g,b); return *this; }
        inline RenderingProperties& setPointSize(float sz) { pointSize = sz; return *this; }
        inline RenderingProperties& setLineWidth(float lw) { lineWidth = lw; return *this; }
        inline RenderingProperties& setOpacity(float op) { opacity = op; return *this; }
        inline RenderingProperties& setNormalLength(float nl) { normalLength = nl; return *this; }
        inline RenderingProperties& setCorrespondencesFraction(float cf) { correspondencesFraction = cf; return *this; }
        inline RenderingProperties& setOverrideColors(bool oc) { overrideColors = oc; return *this; }
    };

    Visualizer(const std::string & window_name, const std::string &display_name);
    inline ~Visualizer() {}

    void addPointCloud(const std::string &name, const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &colors, const RenderingProperties &rp = RenderingProperties());
    void addPointCloud(const std::string &name, const std::vector<Eigen::Vector3f> &points, const RenderingProperties &rp = RenderingProperties());
    void addPointCloud(const std::string &name, const PointCloud &cloud, const RenderingProperties &rp = RenderingProperties());

    void addPointCloudNormals(const std::string &name, const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &normals, const RenderingProperties &rp = RenderingProperties());
    void addPointCloudNormals(const std::string &name, const PointCloud &cloud, const RenderingProperties &rp = RenderingProperties());

    void addPointCorrespondences(const std::string &name, const std::vector<Eigen::Vector3f> &points_src, const std::vector<Eigen::Vector3f> &points_dst, const RenderingProperties &rp = RenderingProperties());
    void addPointCorrespondences(const std::string &name, const PointCloud &cloud_src, const PointCloud &cloud_dst, const RenderingProperties &rp = RenderingProperties());

    void addCoordinateSystem(const std::string &name, float scale = 1.0f, const Eigen::Matrix4f &tf = Eigen::Matrix4f::Identity(), const RenderingProperties &rp = RenderingProperties());

    inline void clear() { renderables_.clear(); }
    inline void remove(const std::string &name) { renderables_.erase(name); }

    void render(const std::string &obj_name) const;
    void render() const;

    inline void spinOnce() const { render(); pangolin::FinishFrame(); }
    inline bool wasStopped() const { return gl_context_->quit; }

    bool getVisibility(const std::string &name) const;
    void setVisibility(const std::string &name, bool visible);
    inline void toggleVisibility(const std::string &name) { setVisibility(name, !getVisibility(name)); }

    RenderingProperties getRenderingProperties(const std::string &name) const;
    void setRenderingProperties(const std::string &name, const RenderingProperties &rp);

    std::vector<std::string> getObjectNames() const;

    inline Eigen::Vector3f getClearColor() const { return clear_color_; }
    inline void setClearColor(const Eigen::Vector3f &color) { clear_color_ = color; }
    inline void setClearColor(float r, float g, float b) { clear_color_ = Eigen::Vector3f(r,g,b); }

    void setProjectionMatrix(int w, int h, pangolin::GLprecision fu, pangolin::GLprecision fv, pangolin::GLprecision u0, pangolin::GLprecision v0, pangolin::GLprecision zNear, pangolin::GLprecision zFar);

    void registerKeyboardCallback(const std::vector<int> &keys, std::function<void(Visualizer&,int,void*)> func, void *cookie);

private:
    struct Renderable_ {
        Renderable_() : visible(true), position(Eigen::Vector3f::Zero()) {}
        bool visible;
        Eigen::Vector3f position;                       // For render priority...
        RenderingProperties renderingProperties;
        virtual void applyRenderingProperties() = 0;    // Updates GPU buffers
        inline void applyRenderingProperties(const RenderingProperties &rp) { renderingProperties = rp; applyRenderingProperties(); }
        virtual void render() = 0;
    };

    struct PointsRenderable_ : public Renderable_
    {
        std::vector<Eigen::Vector3f> points;
        std::vector<Eigen::Vector3f> colors;
        pangolin::GlBuffer pointsBuffer;
        pangolin::GlBuffer colorsBuffer;
        void applyRenderingProperties();
        void render();
    };

    struct NormalsRenderable_ : public Renderable_
    {
        std::vector<Eigen::Vector3f> points;
        std::vector<Eigen::Vector3f> normals;
        pangolin::GlBuffer lineEndPointsBuffer;
        void applyRenderingProperties();
        void render();
    };

    struct CorrespondencesRenderable_ : public Renderable_
    {
        std::vector<Eigen::Vector3f> srcPoints;
        std::vector<Eigen::Vector3f> dstPoints;
        pangolin::GlBuffer lineEndPointsBuffer;
        void applyRenderingProperties();
        void render();
    };

    struct AxisRenderable_ : public Renderable_
    {
        float scale;
        Eigen::Matrix4f transform;
        void applyRenderingProperties();
        void render();
    };

    pangolin::PangolinGl *gl_context_;
    pangolin::View *display_;

    std::unique_ptr<pangolin::OpenGlRenderState> gl_render_state_;
    std::unique_ptr<pangolin::Handler3D> input_handler_;
    pangolin::OpenGlMatrix initial_model_view_;

    Eigen::Vector3f clear_color_;

    std::map<std::string, std::unique_ptr<Renderable_> > renderables_;

    static void point_size_callback_(Visualizer &viz, int key);
    static void reset_view_callback_(Visualizer &viz);

    struct {
        bool operator()(const std::pair<Visualizer::Renderable_*, float> &p1, const std::pair<Visualizer::Renderable_*, float> &p2) const {
            if (p1.first->renderingProperties.opacity == 1.0f && p2.first->renderingProperties.opacity < 1.0f) {
                return true;
            } else {
                return p1.second > p2.second;
            }
        }
    } render_priority_comparator_;
};
