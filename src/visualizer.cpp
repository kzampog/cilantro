#include <sisyphus/visualizer.hpp>

void Visualizer::PointsRenderable_::applyRenderingProperties() {
    pointsBuffer.Reinitialise(pangolin::GlArrayBuffer, points.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
    pointsBuffer.Upload(points.data(), sizeof(float)*points.size()*3);

    if (renderingProperties.overrideColors || colors.size() != points.size()) {
        std::vector<Eigen::Vector3f> render_colors(points.size(), renderingProperties.drawingColor);
        colorsBuffer.Reinitialise(pangolin::GlArrayBuffer, render_colors.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        colorsBuffer.Upload(render_colors.data(), sizeof(float)*render_colors.size()*3);
    } else {
        colorsBuffer.Reinitialise(pangolin::GlArrayBuffer, colors.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        colorsBuffer.Upload(colors.data(), sizeof(float)*colors.size()*3);
    }
}

void Visualizer::PointsRenderable_::render() {
    glPointSize(renderingProperties.pointSize);
    pangolin::RenderVboCbo(pointsBuffer, colorsBuffer);
}

void Visualizer::NormalsRenderable_::applyRenderingProperties() {
    if (renderingProperties.normalsPercentage <= 0.0f) {
        renderingProperties.normalsPercentage = 0.0;
        lineEndPointsBuffer.Resize(0);
        return;
    }
    if (renderingProperties.normalsPercentage > 1.0f) renderingProperties.normalsPercentage = 1.0;

    size_t step = std::floor(1.0/renderingProperties.normalsPercentage);
    std::vector<Eigen::Vector3f> tmp(2*((points.size() - 1)/step + 1));

    for (size_t i = 0; i < points.size(); i += step) {
        tmp[2*i/step + 0] = points[i];
        tmp[2*i/step + 1] = points[i] + renderingProperties.normalLength * normals[i];
    }

    lineEndPointsBuffer.Reinitialise(pangolin::GlArrayBuffer, tmp.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
    lineEndPointsBuffer.Upload(tmp.data(), sizeof(float)*tmp.size()*3);
}

void Visualizer::NormalsRenderable_::render() {
    glPointSize(renderingProperties.pointSize);
    glColor3f(renderingProperties.drawingColor(0), renderingProperties.drawingColor(1), renderingProperties.drawingColor(2));
    glLineWidth(renderingProperties.lineWidth);
    pangolin::RenderVbo(lineEndPointsBuffer, GL_LINES);
}

Visualizer::Visualizer(const std::string &window_name, const std::string &display_name)
        : clear_color_(Eigen::Vector3f(0.7f, 0.7f, 1.0f))
{
    gl_context_ = pangolin::FindContext(window_name);
    if (!gl_context_) {
        pangolin::CreateWindowAndBind(window_name);
        gl_context_ = pangolin::FindContext(window_name);
    }

    // Pangolin searches internally for existing named displays
    gl_context_->MakeCurrent();
    display_ = &(pangolin::Display(display_name));

    gl_render_state_.reset(new pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(640, 480, 528, 528, 320, 240, 0.2, 100), pangolin::ModelViewLookAt(0, 0, 0, 0, 0, 1, pangolin::AxisNegY)));
    input_handler_.reset(new pangolin::Handler3D(*gl_render_state_));
    display_->SetHandler(input_handler_.get());
    display_->SetAspect(-4.0f/3.0f);
}

void Visualizer::addPointCloud(const std::string &name, const PointCloud &cloud, const RenderingProperties &rp) {
    renderables_[name] = std::unique_ptr<PointsRenderable_>(new PointsRenderable_);
    PointsRenderable_ *obj_ptr = (PointsRenderable_ *)renderables_[name].get();
    // Copy fields
    obj_ptr->points = cloud.points;
    obj_ptr->colors = cloud.colors;
    // Update buffers
    ((Renderable_ *)obj_ptr)->applyRenderingProperties(rp);
}

void Visualizer::addPointCloudNormals(const std::string &name, const PointCloud &cloud, const RenderingProperties &rp) {
    if (!cloud.hasNormals()) return;

    renderables_[name] = std::unique_ptr<NormalsRenderable_>(new NormalsRenderable_);
    NormalsRenderable_ *obj_ptr = (NormalsRenderable_ *)renderables_[name].get();
    // Copy fields
    obj_ptr->points = cloud.points;
    obj_ptr->normals = cloud.normals;
    // Update buffers
    ((Renderable_ *)obj_ptr)->applyRenderingProperties(rp);
}

void Visualizer::render() {
    gl_context_->MakeCurrent();
    display_->Activate(*gl_render_state_);
    glEnable(GL_DEPTH_TEST);
    glClearColor(clear_color_(0), clear_color_(1), clear_color_(2), 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    for (auto it = renderables_.begin(); it != renderables_.end(); ++it) {
        it->second->render();
    }
}

void Visualizer::render(const std::string &obj_name) {
    gl_context_->MakeCurrent();
    display_->Activate(*gl_render_state_);
    glEnable(GL_DEPTH_TEST);
    glClearColor(clear_color_(0), clear_color_(1), clear_color_(2), 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    auto it = renderables_.find(obj_name);
    if (it != renderables_.end()) it->second->render();
}

void Visualizer::setProjectionMatrix(int w, int h, pangolin::GLprecision fu, pangolin::GLprecision fv, pangolin::GLprecision u0, pangolin::GLprecision v0, pangolin::GLprecision zNear, pangolin::GLprecision zFar) {
    gl_render_state_->SetProjectionMatrix(pangolin::ProjectionMatrix(w, h, fu, fv, u0, v0, zNear, zFar));
    display_->SetAspect(-(double)w/((double)h));
}
