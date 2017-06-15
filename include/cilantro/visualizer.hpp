#pragma once

#include <cilantro/point_cloud.hpp>
#include <cilantro/colormap.hpp>
#include <pangolin/pangolin.h>
#include <pangolin/display/display_internal.h>

class Visualizer {
public:
    struct RenderingProperties {
        inline RenderingProperties() : drawingColor(no_color_),
                                       pointSize(2.0f),
                                       lineWidth(1.0f),
                                       opacity(1.0f),
                                       normalLength(0.05f),
                                       correspondencesFraction(1.0),
                                       drawWireframe(false),
                                       useFaceNormals(true),
                                       useFaceColors(false),
                                       minScalarValue(std::numeric_limits<float>::quiet_NaN()),
                                       maxScalarValue(std::numeric_limits<float>::quiet_NaN()),
                                       colormapType(ColormapType::JET)
        {}
        inline ~RenderingProperties() {}

        Eigen::Vector3f drawingColor;
        float pointSize;
        float lineWidth;
        float opacity;
        float normalLength;
        float correspondencesFraction;
        bool drawWireframe;
        bool useFaceNormals;
        bool useFaceColors;
        float minScalarValue;
        float maxScalarValue;
        ColormapType colormapType;

        inline RenderingProperties& setDrawingColor(const Eigen::Vector3f &color) { drawingColor = color; return *this; }
        inline RenderingProperties& setDrawingColor(float r, float g, float b) { drawingColor = Eigen::Vector3f(r,g,b); return *this; }
        inline RenderingProperties& setPointSize(float sz) { pointSize = sz; return *this; }
        inline RenderingProperties& setLineWidth(float lw) { lineWidth = lw; return *this; }
        inline RenderingProperties& setOpacity(float op) { opacity = op; return *this; }
        inline RenderingProperties& setNormalLength(float nl) { normalLength = nl; return *this; }
        inline RenderingProperties& setCorrespondencesFraction(float cf) { correspondencesFraction = cf; return *this; }
        inline RenderingProperties& setDrawWireframe(bool dw) { drawWireframe = dw; return *this; }
        inline RenderingProperties& setUseFaceNormals(bool fn) { useFaceNormals = fn; return *this; }
        inline RenderingProperties& setUseFaceColors(bool fc) { useFaceColors = fc; return *this; }
        inline RenderingProperties& setScalarValuesRange(float min, float max) { minScalarValue = min; maxScalarValue = max; return *this; }
        inline RenderingProperties& setColormapType(const ColormapType ct) { colormapType = ct; return *this; }
    };

    Visualizer(const std::string &window_name, const std::string &display_name);
    ~Visualizer();

    Visualizer& addPointCloud(const std::string &name, const PointCloud &cloud, const RenderingProperties &rp = RenderingProperties());
    Visualizer& addPointCloud(const std::string &name, const std::vector<Eigen::Vector3f> &points, const RenderingProperties &rp = RenderingProperties());
    Visualizer& addPointCloudColors(const std::string &name, const std::vector<Eigen::Vector3f> &colors);
    Visualizer& addPointCloudValues(const std::string &name, const std::vector<float> &point_values);

    Visualizer& addPointCloudNormals(const std::string &name, const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &normals, const RenderingProperties &rp = RenderingProperties());
    Visualizer& addPointCloudNormals(const std::string &name, const PointCloud &cloud, const RenderingProperties &rp = RenderingProperties());

    Visualizer& addPointCorrespondences(const std::string &name, const std::vector<Eigen::Vector3f> &points_src, const std::vector<Eigen::Vector3f> &points_dst, const RenderingProperties &rp = RenderingProperties());
    Visualizer& addPointCorrespondences(const std::string &name, const PointCloud &cloud_src, const PointCloud &cloud_dst, const RenderingProperties &rp = RenderingProperties());

    Visualizer& addCoordinateSystem(const std::string &name, float scale = 1.0f, const Eigen::Matrix4f &tf = Eigen::Matrix4f::Identity(), const RenderingProperties &rp = RenderingProperties());

    Visualizer& addTriangleMesh(const std::string &name, const PointCloud &cloud, const std::vector<std::vector<size_t> > &faces, const RenderingProperties &rp = RenderingProperties());
    Visualizer& addTriangleMesh(const std::string &name, const std::vector<Eigen::Vector3f> &vertices, const std::vector<std::vector<size_t> > &faces, const RenderingProperties &rp = RenderingProperties());
    Visualizer& addTriangleMeshVertexNormals(const std::string &name, const std::vector<Eigen::Vector3f> &vertex_normals);
    Visualizer& addTriangleMeshFaceNormals(const std::string &name, const std::vector<Eigen::Vector3f> &face_normals);
    Visualizer& addTriangleMeshVertexColors(const std::string &name, const std::vector<Eigen::Vector3f> &vertex_colors);
    Visualizer& addTriangleMeshFaceColors(const std::string &name, const std::vector<Eigen::Vector3f> &face_colors);
    Visualizer& addTriangleMeshVertexValues(const std::string &name, const std::vector<float> &vertex_values);
    Visualizer& addTriangleMeshFaceValues(const std::string &name, const std::vector<float> &face_values);

    inline Visualizer& clear() { renderables_.clear(); return *this; }
    inline Visualizer& remove(const std::string &name) { renderables_.erase(name); return *this; }

    inline void clearRenderArea() const {
        gl_context_->MakeCurrent();
        display_->Activate(*gl_render_state_);
        glClearColor(clear_color_(0), clear_color_(1), clear_color_(2), 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    void render() const;
    inline void finishFrame() const { gl_context_->MakeCurrent(); pangolin::FinishFrame(); }
    inline void spinOnce() const { clearRenderArea(); render(); finishFrame(); }

    inline bool wasStopped() const { return gl_context_->quit; }

    bool getVisibility(const std::string &name) const;
    void setVisibility(const std::string &name, bool visible);
    inline void toggleVisibility(const std::string &name) { setVisibility(name, !getVisibility(name)); }

    RenderingProperties getRenderingProperties(const std::string &name) const;
    void setRenderingProperties(const std::string &name, const RenderingProperties &rp);

    std::vector<std::string> getObjectNames() const;

    inline Eigen::Vector3f getClearColor() const { return clear_color_; }
    inline Visualizer& setClearColor(const Eigen::Vector3f &color) { clear_color_ = color; return *this; }
    inline Visualizer& setClearColor(float r, float g, float b) { clear_color_ = Eigen::Vector3f(r,g,b); return *this; }

    Visualizer& setProjectionMatrix(int w, int h, pangolin::GLprecision fu, pangolin::GLprecision fv, pangolin::GLprecision u0, pangolin::GLprecision v0, pangolin::GLprecision zNear, pangolin::GLprecision zFar);

    Visualizer& registerKeyboardCallback(const std::vector<int> &keys, std::function<void(Visualizer&,int,void*)> func, void *cookie);

private:
    struct Renderable_ {
        Renderable_() : visible(true), centroid(Eigen::Vector3f::Zero()) {}
        bool visible;
        Eigen::Vector3f centroid;                       // For render priority...
        RenderingProperties renderingProperties;
        virtual void applyRenderingProperties() = 0;    // Updates GPU buffers
        inline void applyRenderingProperties(const RenderingProperties &rp) { renderingProperties = rp; applyRenderingProperties(); }
        virtual void render() = 0;
    };

    struct PointsRenderable_ : public Renderable_ {
        std::vector<Eigen::Vector3f> points;
        std::vector<Eigen::Vector3f> colors;
        std::vector<float> pointValues;
        pangolin::GlBuffer pointBuffer;
        pangolin::GlBuffer colorBuffer;
        void applyRenderingProperties();
        void render();
    };

    struct NormalsRenderable_ : public Renderable_ {
        std::vector<Eigen::Vector3f> points;
        std::vector<Eigen::Vector3f> normals;
        pangolin::GlBuffer lineEndPointBuffer;
        void applyRenderingProperties();
        void render();
    };

    struct CorrespondencesRenderable_ : public Renderable_ {
        std::vector<Eigen::Vector3f> srcPoints;
        std::vector<Eigen::Vector3f> dstPoints;
        pangolin::GlBuffer lineEndPointBuffer;
        void applyRenderingProperties();
        void render();
    };

    struct AxisRenderable_ : public Renderable_ {
        float scale;
        Eigen::Matrix4f transform;
        void applyRenderingProperties();
        void render();
    };

    struct TriangleMeshRenderable_ : public Renderable_ {
        std::vector<Eigen::Vector3f> vertices;
        std::vector<std::vector<size_t> > faces;
        std::vector<Eigen::Vector3f> vertexNormals;
        std::vector<Eigen::Vector3f> faceNormals;
        std::vector<Eigen::Vector3f> vertexColors;
        std::vector<Eigen::Vector3f> faceColors;
        std::vector<float> vertexValues;
        std::vector<float> faceValues;
        pangolin::GlBuffer vertexBuffer;
        pangolin::GlBuffer colorBuffer;
        pangolin::GlBuffer normalBuffer;
        void applyRenderingProperties();
        void render();
    };

    pangolin::PangolinGl *gl_context_;
    pangolin::View *display_;

    std::shared_ptr<pangolin::OpenGlRenderState> gl_render_state_;
    std::shared_ptr<pangolin::Handler3D> input_handler_;
    pangolin::OpenGlMatrix initial_model_view_;

    static Eigen::Vector3f no_color_;
    static Eigen::Vector3f default_color_;
    Eigen::Vector3f clear_color_;

    std::map<std::string, std::shared_ptr<Renderable_> > renderables_;

    static void point_size_callback_(Visualizer &viz, int key);
    static void reset_view_callback_(Visualizer &viz);
    static void wireframe_toggle_callback_(Visualizer &viz);

    struct {
        inline bool operator()(const std::pair<Visualizer::Renderable_*, float> &p1, const std::pair<Visualizer::Renderable_*, float> &p2) const {
            if (p1.first->renderingProperties.opacity == 1.0f) {
                return true;
            } else if (p2.first->renderingProperties.opacity == 1.0f) {
                return false;
            } else {
                return p1.second > p2.second;
            }
        }
    } render_priority_comparator_;
};
