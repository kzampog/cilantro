#include <sisyphus/cloud_visualizer.hpp>

void CloudVisualizer::PointsRenderable_::render() {
    pangolin::RenderVboCbo(pointsBuffer, colorsBuffer);
}

void CloudVisualizer::NormalsRenderable_::render() {
    glColor3f(1.0f, 1.0f, 1.0f);
    glLineWidth(0.1f);
    pangolin::RenderVbo(lineEndPointsBuffer, GL_LINES);
}

CloudVisualizer::CloudVisualizer(const std::string &window_name, const std::string &display_name)
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

CloudVisualizer::~CloudVisualizer() {}

void CloudVisualizer::addPointCloud(const std::string &name, const PointCloud &cloud) {
    renderables_[name] = std::unique_ptr<PointsRenderable_>(new PointsRenderable_);

    // Populate buffer objects
    ((PointsRenderable_ *)renderables_[name].get())->pointsBuffer.Reinitialise(pangolin::GlArrayBuffer, cloud.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
    ((PointsRenderable_ *)renderables_[name].get())->pointsBuffer.Upload(cloud.points.data(), sizeof(float)*cloud.size()*3);

    ((PointsRenderable_ *)renderables_[name].get())->colorsBuffer.Reinitialise(pangolin::GlArrayBuffer, cloud.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
    ((PointsRenderable_ *)renderables_[name].get())->colorsBuffer.Upload(cloud.colors.data(), sizeof(float)*cloud.size()*3);
}

void CloudVisualizer::addPointCloudNormals(const std::string &name, const PointCloud &cloud) {
    renderables_[name] = std::unique_ptr<NormalsRenderable_>(new NormalsRenderable_);

    std::vector<Eigen::Vector3f> tmp(cloud.size()*2);
    for (size_t i = 0; i < cloud.size(); i++) {
        tmp[2*i + 0] = cloud.points[i];
        tmp[2*i + 1] = cloud.points[i] + 0.05f*cloud.normals[i];
    }

    ((NormalsRenderable_ *)renderables_[name].get())->lineEndPointsBuffer.Reinitialise(pangolin::GlArrayBuffer, tmp.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
    ((NormalsRenderable_ *)renderables_[name].get())->lineEndPointsBuffer.Upload(tmp.data(), sizeof(float)*tmp.size()*3);
}

void CloudVisualizer::clear() {
    renderables_.clear();
}

void CloudVisualizer::remove(const std::string &name) {
    renderables_.erase(name);
}

void CloudVisualizer::render() {
    gl_context_->MakeCurrent();
    display_->Activate(*gl_render_state_);
    glEnable(GL_DEPTH_TEST);
    glPointSize(5.0f);
    glClearColor(clear_color_(0), clear_color_(1), clear_color_(2), 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    for (auto it = renderables_.begin(); it != renderables_.end(); ++it) {
        it->second->render();
    }
}

void CloudVisualizer::render(const std::string &obj_name) {
    gl_context_->MakeCurrent();
    display_->Activate(*gl_render_state_);
    glEnable(GL_DEPTH_TEST);
    glPointSize(5.0f);
    glClearColor(clear_color_(0), clear_color_(1), clear_color_(2), 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    auto it = renderables_.find(obj_name);
    if (it != renderables_.end()) it->second->render();
}

void CloudVisualizer::setProjectionMatrix(int w, int h, pangolin::GLprecision fu, pangolin::GLprecision fv, pangolin::GLprecision u0, pangolin::GLprecision v0, pangolin::GLprecision zNear, pangolin::GLprecision zFar) {
    gl_render_state_->SetProjectionMatrix(pangolin::ProjectionMatrix(w, h, fu, fv, u0, v0, zNear, zFar));
    display_->SetAspect(-(double)w/((double)h));
}
