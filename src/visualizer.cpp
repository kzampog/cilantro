#include <cilantro/visualizer.hpp>

void Visualizer::PointsRenderable_::applyRenderingProperties() {
    pointsBuffer.Reinitialise(pangolin::GlArrayBuffer, points.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
    pointsBuffer.Upload(points.data(), sizeof(float)*points.size()*3);

    std::vector<Eigen::Vector4f> color_alpha;
    if (renderingProperties.overrideColors || colors.size() != points.size()) {
        Eigen::Vector4f tmp;
        tmp.head(3) = renderingProperties.drawingColor;
        tmp(3) = renderingProperties.opacity;
        color_alpha = std::vector<Eigen::Vector4f>(points.size(), tmp);
    } else {
        color_alpha.resize(colors.size());
        for(size_t i = 0; i < colors.size(); i++) {
            color_alpha[i].head(3) = colors[i];
            color_alpha[i](3) = renderingProperties.opacity;
        }
    }
    colorsBuffer.Reinitialise(pangolin::GlArrayBuffer, color_alpha.size(), GL_FLOAT, 4, GL_DYNAMIC_DRAW);
    colorsBuffer.Upload(color_alpha.data(), sizeof(float)*color_alpha.size()*4);
}

void Visualizer::PointsRenderable_::render() {
    glPointSize(renderingProperties.pointSize);
    pangolin::RenderVboCbo(pointsBuffer, colorsBuffer);
}

void Visualizer::NormalsRenderable_::applyRenderingProperties() {
    if (renderingProperties.correspondencesFraction <= 0.0f) {
        renderingProperties.correspondencesFraction = 0.0;
        lineEndPointsBuffer.Resize(0);
        return;
    }
    if (renderingProperties.correspondencesFraction > 1.0f) renderingProperties.correspondencesFraction = 1.0;

    size_t step = std::floor(1.0/renderingProperties.correspondencesFraction);
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
    glColor4f(renderingProperties.drawingColor(0), renderingProperties.drawingColor(1), renderingProperties.drawingColor(2), renderingProperties.opacity);
    glLineWidth(renderingProperties.lineWidth);
    pangolin::RenderVbo(lineEndPointsBuffer, GL_LINES);
}

void Visualizer::CorrespondencesRenderable_::applyRenderingProperties() {
    if (renderingProperties.correspondencesFraction <= 0.0f) {
        renderingProperties.correspondencesFraction = 0.0;
        lineEndPointsBuffer.Resize(0);
        return;
    }
    if (renderingProperties.correspondencesFraction > 1.0f) renderingProperties.correspondencesFraction = 1.0;

    size_t step = std::floor(1.0/renderingProperties.correspondencesFraction);
    std::vector<Eigen::Vector3f> tmp(2*((points_src.size() - 1)/step + 1));

    for (size_t i = 0; i < points_src.size(); i += step) {
        tmp[2*i/step + 0] = points_src[i];
        tmp[2*i/step + 1] = points_dst[i];
    }

    lineEndPointsBuffer.Reinitialise(pangolin::GlArrayBuffer, tmp.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
    lineEndPointsBuffer.Upload(tmp.data(), sizeof(float)*tmp.size()*3);
}

void Visualizer::CorrespondencesRenderable_::render() {
    glPointSize(renderingProperties.pointSize);
    glColor4f(renderingProperties.drawingColor(0), renderingProperties.drawingColor(1), renderingProperties.drawingColor(2), renderingProperties.opacity);
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

    gl_context_->MakeCurrent();

    // Default callbacks
    pangolin::RegisterKeyPressCallback('q', pangolin::Quit);
    pangolin::RegisterKeyPressCallback('Q', pangolin::Quit);
    auto inc_fun = std::bind(point_size_callback_, std::ref(*this), '+');
    pangolin::RegisterKeyPressCallback('+', inc_fun);
    auto dec_fun = std::bind(point_size_callback_, std::ref(*this), '-');
    pangolin::RegisterKeyPressCallback('-', dec_fun);
    auto reset_fun = std::bind(reset_view_callback_, std::ref(*this));
    pangolin::RegisterKeyPressCallback('r', reset_fun);
    pangolin::RegisterKeyPressCallback('R', reset_fun);

    // Pangolin searches internally for existing named displays
    display_ = &(pangolin::Display(display_name));

    initial_model_view_ = pangolin::ModelViewLookAt(0, 0, 0, 0, 0, 1, pangolin::AxisNegY);
    gl_render_state_.reset(new pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(640, 480, 528, 528, 320, 240, 0.2, 100), initial_model_view_));
    input_handler_.reset(new pangolin::Handler3D(*gl_render_state_));
    display_->SetHandler(input_handler_.get());
    display_->SetAspect(-4.0f/3.0f);
}

void Visualizer::addPointCloud(const std::string &name, const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &colors, const RenderingProperties &rp) {
    renderables_[name] = std::unique_ptr<PointsRenderable_>(new PointsRenderable_);
    PointsRenderable_ *obj_ptr = (PointsRenderable_ *)renderables_[name].get();
    // Copy fields
    obj_ptr->points = points;
    obj_ptr->colors = colors;
    // Update buffers
    ((Renderable_ *)obj_ptr)->applyRenderingProperties(rp);
}

void Visualizer::addPointCloud(const std::string &name, const std::vector<Eigen::Vector3f> &points, const RenderingProperties &rp) {
    addPointCloud(name, points, std::vector<Eigen::Vector3f>(), rp);
}

void Visualizer::addPointCloud(const std::string &name, const PointCloud &cloud, const RenderingProperties &rp) {
    addPointCloud(name, cloud.points, cloud.colors, rp);
}

void Visualizer::addPointCloudNormals(const std::string &name, const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &normals, const RenderingProperties &rp) {
    if (points.size() != normals.size()) return;
    renderables_[name] = std::unique_ptr<NormalsRenderable_>(new NormalsRenderable_);
    NormalsRenderable_ *obj_ptr = (NormalsRenderable_ *)renderables_[name].get();
    // Copy fields
    obj_ptr->points = points;
    obj_ptr->normals = normals;
    // Update buffers
    ((Renderable_ *)obj_ptr)->applyRenderingProperties(rp);
}

void Visualizer::addPointCloudNormals(const std::string &name, const PointCloud &cloud, const RenderingProperties &rp) {
    addPointCloudNormals(name, cloud.points, cloud.normals, rp);
}

void Visualizer::addPointCorrespondences(const std::string &name, const std::vector<Eigen::Vector3f> &points_src, const std::vector<Eigen::Vector3f> &points_dst, const RenderingProperties &rp) {
    if (points_src.size() != points_dst.size()) return;
    renderables_[name] = std::unique_ptr<CorrespondencesRenderable_>(new CorrespondencesRenderable_);
    CorrespondencesRenderable_ *obj_ptr = (CorrespondencesRenderable_ *)renderables_[name].get();
    // Copy fields
    obj_ptr->points_src = points_src;
    obj_ptr->points_dst = points_dst;
    // Update buffers
    ((Renderable_ *)obj_ptr)->applyRenderingProperties(rp);
}

void Visualizer::addPointCorrespondences(const std::string &name, const PointCloud &cloud_src, const PointCloud &cloud_dst, const RenderingProperties &rp) {
    addPointCorrespondences(name, cloud_src.points, cloud_dst.points, rp);
}

void Visualizer::render(const std::string &obj_name) {
    gl_context_->MakeCurrent();
    display_->Activate(*gl_render_state_);
    glEnable(GL_DEPTH_TEST);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(clear_color_(0), clear_color_(1), clear_color_(2), 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    auto it = renderables_.find(obj_name);
    if (it != renderables_.end()) it->second->render();
}

void Visualizer::render() {
    gl_context_->MakeCurrent();
    display_->Activate(*gl_render_state_);
    glEnable(GL_DEPTH_TEST);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(clear_color_(0), clear_color_(1), clear_color_(2), 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    for (auto it = renderables_.begin(); it != renderables_.end(); ++it) {
        it->second->render();
    }
}

Visualizer::RenderingProperties Visualizer::getRenderingProperties(const std::string &name) {
    auto it = renderables_.find(name);
    if (it == renderables_.end()) return RenderingProperties();
    return it->second->renderingProperties;
}

void Visualizer::setRenderingProperties(const std::string &name, const RenderingProperties &rp) {
    auto it = renderables_.find(name);
    if (it == renderables_.end()) return;
    it->second->applyRenderingProperties(rp);
}

std::vector<std::string> Visualizer::getObjectNames() {
    std::vector<std::string> res;
    for (auto it = renderables_.begin(); it != renderables_.end(); ++it) {
        res.push_back(it->first);
    }
    return res;
}

void Visualizer::setProjectionMatrix(int w, int h, pangolin::GLprecision fu, pangolin::GLprecision fv, pangolin::GLprecision u0, pangolin::GLprecision v0, pangolin::GLprecision zNear, pangolin::GLprecision zFar) {
    gl_render_state_->SetProjectionMatrix(pangolin::ProjectionMatrix(w, h, fu, fv, u0, v0, zNear, zFar));
    display_->SetAspect(-(double)w/((double)h));
}

void Visualizer::registerKeyboardCallback(const std::vector<int> &keys, std::function<void(Visualizer&,int,void*)> func, void *cookie) {
    gl_context_->MakeCurrent();
    for (size_t i = 0; i < keys.size(); i++) {
        auto fun = std::bind(func, std::ref(*this), keys[i], cookie);
        pangolin::RegisterKeyPressCallback(keys[i], fun);
    }
}

void Visualizer::point_size_callback_(Visualizer &viz, int key) {
    if (key == '+') {
        for (auto it = viz.renderables_.begin(); it != viz.renderables_.end(); ++it) {
            RenderingProperties rp = viz.getRenderingProperties(it->first);
            rp.pointSize += 1.0;
            rp.lineWidth = rp.pointSize/5.0f;
            viz.setRenderingProperties(it->first, rp);
        }
    } else if (key == '-') {
        for (auto it = viz.renderables_.begin(); it != viz.renderables_.end(); ++it) {
            RenderingProperties rp = viz.getRenderingProperties(it->first);
            rp.pointSize = std::max(rp.pointSize - 1.0f, 1.0f);
            rp.lineWidth = rp.pointSize/5.0f;
            viz.setRenderingProperties(it->first, rp);
        }
    }
}

void Visualizer::reset_view_callback_(Visualizer &viz) {
    viz.gl_render_state_->SetModelViewMatrix(viz.initial_model_view_);
}
