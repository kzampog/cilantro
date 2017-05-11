#pragma once

#include <sisyphus/point_cloud.hpp>
#include <pangolin/pangolin.h>
#include <pangolin/display/display_internal.h>

class CloudVisualizer {
public:
    CloudVisualizer(const std::string & window_name, const std::string &display_name);
    ~CloudVisualizer();

    void addPointCloud(const std::string &name, const PointCloud &cloud);
    void addPointCloudNormals(const std::string &name, const PointCloud &cloud);

    void clear();
    void remove(const std::string &name);

    void render();
    void render(const std::string &obj_name);

    void setProjectionMatrix(int w, int h, pangolin::GLprecision fu, pangolin::GLprecision fv, pangolin::GLprecision u0, pangolin::GLprecision v0, pangolin::GLprecision zNear, pangolin::GLprecision zFar);

private:
    struct Renderable_ {
        virtual void render() = 0;
    };

    struct PointsRenderable_ : public Renderable_
    {
        pangolin::GlBuffer pointsBuffer;
        pangolin::GlBuffer colorsBuffer;
        void render();
    };

    struct NormalsRenderable_ : public Renderable_
    {
        pangolin::GlBuffer lineEndPointsBuffer;
        void render();
    };

    pangolin::PangolinGl *gl_context_;
    pangolin::View *display_;

    std::unique_ptr<pangolin::OpenGlRenderState> gl_render_state_;
    std::unique_ptr<pangolin::Handler3D> input_handler_;

    Eigen::Vector3f clear_color_;

    std::map<std::string, std::unique_ptr<Renderable_> > renderables_;

};
