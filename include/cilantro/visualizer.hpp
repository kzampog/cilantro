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

    inline void clear() { renderables_.clear(); }
    inline void remove(const std::string &name) { renderables_.erase(name); }

    void render(const std::string &obj_name);
    void render();
    void spinOnce();
    inline bool wasStopped() { return gl_context_->quit; }

    RenderingProperties getRenderingProperties(const std::string &name);
    void setRenderingProperties(const std::string &name, const RenderingProperties &rp);

    std::vector<std::string> getObjectNames();

    inline Eigen::Vector3f getClearColor() const { return clear_color_; }
    inline void setClearColor(const Eigen::Vector3f &color) { clear_color_ = color; }
    inline void setClearColor(float r, float g, float b) { clear_color_ = Eigen::Vector3f(r,g,b); }

    void setProjectionMatrix(int w, int h, pangolin::GLprecision fu, pangolin::GLprecision fv, pangolin::GLprecision u0, pangolin::GLprecision v0, pangolin::GLprecision zNear, pangolin::GLprecision zFar);

    void registerKeyboardCallback(const std::vector<int> &keys, std::function<void(Visualizer&,int,void*)> func, void *cookie);

private:
    struct Renderable_ {
        RenderingProperties renderingProperties;
        virtual void applyRenderingProperties() = 0;    // Updates GPU buffers
        inline void applyRenderingProperties(const RenderingProperties &rp) { renderingProperties = rp; applyRenderingProperties(); }
        virtual void render() = 0;
    };

    struct PointsRenderable_ : public Renderable_
    {
        pangolin::GlBuffer pointsBuffer;
        pangolin::GlBuffer colorsBuffer;
        std::vector<Eigen::Vector3f> points;
        std::vector<Eigen::Vector3f> colors;
        void applyRenderingProperties();
        void render();
    };

    struct NormalsRenderable_ : public Renderable_
    {
        pangolin::GlBuffer lineEndPointsBuffer;
        std::vector<Eigen::Vector3f> points;
        std::vector<Eigen::Vector3f> normals;
        void applyRenderingProperties();
        void render();
    };

    struct CorrespondencesRenderable_ : public Renderable_
    {
        pangolin::GlBuffer lineEndPointsBuffer;
        std::vector<Eigen::Vector3f> points_src;
        std::vector<Eigen::Vector3f> points_dst;
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
};
