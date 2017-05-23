#include <cilantro/visualizer.hpp>

Eigen::Vector3f Visualizer::no_color_ = Eigen::Vector3f(-1,-1,-1);
Eigen::Vector3f Visualizer::default_color_ = Eigen::Vector3f(1,1,1);

void Visualizer::PointsRenderable_::applyRenderingProperties() {
    pointBuffer.Reinitialise(pangolin::GlArrayBuffer, points.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
    pointBuffer.Upload(points.data(), sizeof(float)*points.size()*3);

    std::vector<Eigen::Vector4f> color_alpha;
    if (renderingProperties.drawingColor != no_color_) {
        Eigen::Vector4f tmp;
        tmp.head(3) = renderingProperties.drawingColor;
        tmp(3) = renderingProperties.opacity;
        color_alpha = std::vector<Eigen::Vector4f>(points.size(), tmp);
    } else if (colors.size() != points.size()) {
        Eigen::Vector4f tmp;
        tmp.head(3) = default_color_;
        tmp(3) = renderingProperties.opacity;
        color_alpha = std::vector<Eigen::Vector4f>(points.size(), tmp);
    } else {
        color_alpha.resize(colors.size());
        for(size_t i = 0; i < colors.size(); i++) {
            color_alpha[i].head(3) = colors[i];
            color_alpha[i](3) = renderingProperties.opacity;
        }
    }
    colorBuffer.Reinitialise(pangolin::GlArrayBuffer, color_alpha.size(), GL_FLOAT, 4, GL_DYNAMIC_DRAW);
    colorBuffer.Upload(color_alpha.data(), sizeof(float)*color_alpha.size()*4);
}

void Visualizer::PointsRenderable_::render() {
    glPointSize(renderingProperties.pointSize);
    pangolin::RenderVboCbo(pointBuffer, colorBuffer);
}

void Visualizer::NormalsRenderable_::applyRenderingProperties() {
    if (renderingProperties.correspondencesFraction <= 0.0f) {
        renderingProperties.correspondencesFraction = 0.0;
        lineEndPointBuffer.Resize(0);
        return;
    }
    if (renderingProperties.correspondencesFraction > 1.0f) renderingProperties.correspondencesFraction = 1.0;

    size_t step = std::floor(1.0/renderingProperties.correspondencesFraction);
    std::vector<Eigen::Vector3f> tmp(2*((points.size() - 1)/step + 1));

    for (size_t i = 0; i < points.size(); i += step) {
        tmp[2*i/step + 0] = points[i];
        tmp[2*i/step + 1] = points[i] + renderingProperties.normalLength * normals[i];
    }

    lineEndPointBuffer.Reinitialise(pangolin::GlArrayBuffer, tmp.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
    lineEndPointBuffer.Upload(tmp.data(), sizeof(float)*tmp.size()*3);
}

void Visualizer::NormalsRenderable_::render() {
    glPointSize(renderingProperties.pointSize);
    if (renderingProperties.drawingColor == no_color_)
        glColor4f(default_color_(0), default_color_(1), default_color_(2), renderingProperties.opacity);
    else
        glColor4f(renderingProperties.drawingColor(0), renderingProperties.drawingColor(1), renderingProperties.drawingColor(2), renderingProperties.opacity);
    glLineWidth(renderingProperties.lineWidth);
    pangolin::RenderVbo(lineEndPointBuffer, GL_LINES);
}

void Visualizer::CorrespondencesRenderable_::applyRenderingProperties() {
    if (renderingProperties.correspondencesFraction <= 0.0f) {
        renderingProperties.correspondencesFraction = 0.0;
        lineEndPointBuffer.Resize(0);
        return;
    }
    if (renderingProperties.correspondencesFraction > 1.0f) renderingProperties.correspondencesFraction = 1.0;

    size_t step = std::floor(1.0/renderingProperties.correspondencesFraction);
    std::vector<Eigen::Vector3f> tmp(2*((srcPoints.size() - 1)/step + 1));

    for (size_t i = 0; i < srcPoints.size(); i += step) {
        tmp[2*i/step + 0] = srcPoints[i];
        tmp[2*i/step + 1] = dstPoints[i];
    }

    lineEndPointBuffer.Reinitialise(pangolin::GlArrayBuffer, tmp.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
    lineEndPointBuffer.Upload(tmp.data(), sizeof(float)*tmp.size()*3);
}

void Visualizer::CorrespondencesRenderable_::render() {
    glPointSize(renderingProperties.pointSize);
    if (renderingProperties.drawingColor == no_color_)
        glColor4f(default_color_(0), default_color_(1), default_color_(2), renderingProperties.opacity);
    else
        glColor4f(renderingProperties.drawingColor(0), renderingProperties.drawingColor(1), renderingProperties.drawingColor(2), renderingProperties.opacity);
    glLineWidth(renderingProperties.lineWidth);
    pangolin::RenderVbo(lineEndPointBuffer, GL_LINES);
}

void Visualizer::AxisRenderable_::applyRenderingProperties() {}

void Visualizer::AxisRenderable_::render() {
    glLineWidth(renderingProperties.lineWidth);
    pangolin::glDrawAxis<float>(transform, scale);
}

void Visualizer::TriangleMeshRenderable_::applyRenderingProperties() {
    vertexBuffer.Reinitialise(pangolin::GlArrayBuffer, vertices.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
    vertexBuffer.Upload(vertices.data(), sizeof(float)*vertices.size()*3);
    if (renderingProperties.useFaceNormals) {
        normalBuffer.Reinitialise(pangolin::GlArrayBuffer, faceNormals.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        normalBuffer.Upload(faceNormals.data(), sizeof(float)*faceNormals.size()*3);
    } else {
        normalBuffer.Reinitialise(pangolin::GlArrayBuffer, vertexNormals.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        normalBuffer.Upload(vertexNormals.data(), sizeof(float)*vertexNormals.size()*3);
    }
}

void Visualizer::TriangleMeshRenderable_::render() {
    glPointSize(renderingProperties.pointSize);
    if (renderingProperties.drawingColor == no_color_)
        glColor4f(default_color_(0), default_color_(1), default_color_(2), renderingProperties.opacity);
    else
        glColor4f(renderingProperties.drawingColor(0), renderingProperties.drawingColor(1), renderingProperties.drawingColor(2), renderingProperties.opacity);
    glLineWidth(renderingProperties.lineWidth);

    bool use_normals = (renderingProperties.useFaceNormals && faceNormals.size() == vertices.size()) ||
            (!renderingProperties.useFaceNormals && vertexNormals.size() == vertices.size());

    if (use_normals) {
        normalBuffer.Bind();
        glNormalPointer(normalBuffer.datatype, (GLsizei)(normalBuffer.count_per_element * pangolin::GlDataTypeBytes(normalBuffer.datatype)), 0);
        glEnableClientState(GL_NORMAL_ARRAY);
    }
    if (renderingProperties.drawWireframe) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        pangolin::RenderVbo(vertexBuffer, GL_TRIANGLES);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    } else {
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        glEnable(GL_COLOR_MATERIAL);
        pangolin::RenderVbo(vertexBuffer, GL_TRIANGLES);
        glDisable(GL_COLOR_MATERIAL);
        glDisable(GL_LIGHT0);
        glDisable(GL_LIGHTING);
    }
    if (use_normals) {
        glDisableClientState(GL_NORMAL_ARRAY);
        normalBuffer.Unbind();
    }
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
    auto wireframe_fun = std::bind(wireframe_toggle_callback_, std::ref(*this));
    pangolin::RegisterKeyPressCallback('w', wireframe_fun);
    pangolin::RegisterKeyPressCallback('W', wireframe_fun);

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
    obj_ptr->position = Eigen::Map<Eigen::MatrixXf>((float *)points.data(), 3, points.size()).rowwise().mean();
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
    obj_ptr->position = Eigen::Map<Eigen::MatrixXf>((float *)points.data(), 3, points.size()).rowwise().mean();
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
    obj_ptr->srcPoints = points_src;
    obj_ptr->dstPoints = points_dst;
    obj_ptr->position = (Eigen::Map<Eigen::MatrixXf>((float *)points_src.data(), 3, points_src.size()).rowwise().mean() +
                         Eigen::Map<Eigen::MatrixXf>((float *)points_dst.data(), 3, points_dst.size()).rowwise().mean()) / 2.0f;
    // Update buffers
    ((Renderable_ *)obj_ptr)->applyRenderingProperties(rp);
}

void Visualizer::addPointCorrespondences(const std::string &name, const PointCloud &cloud_src, const PointCloud &cloud_dst, const RenderingProperties &rp) {
    addPointCorrespondences(name, cloud_src.points, cloud_dst.points, rp);
}

void Visualizer::addCoordinateSystem(const std::string &name, float scale, const Eigen::Matrix4f &tf, const RenderingProperties &rp) {
    renderables_[name] = std::unique_ptr<AxisRenderable_>(new AxisRenderable_);
    AxisRenderable_ *obj_ptr = (AxisRenderable_ *)renderables_[name].get();
    // Copy fields
    obj_ptr->scale = scale;
    obj_ptr->transform = tf;
    obj_ptr->position = tf.topRightCorner(3,1);
    // Update buffers
    ((Renderable_ *)obj_ptr)->applyRenderingProperties(rp);
}

void Visualizer::addTriangleMesh(const std::string &name, const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &point_normals, const std::vector<std::vector<size_t> > &faces, const std::vector<Eigen::Vector3f> &face_normals, const RenderingProperties &rp) {
    renderables_[name] = std::unique_ptr<TriangleMeshRenderable_>(new TriangleMeshRenderable_);
    TriangleMeshRenderable_ *obj_ptr = (TriangleMeshRenderable_ *)renderables_[name].get();

    // Populate triangle vertices
    size_t k = 0;
    std::vector<Eigen::Vector3f> vertices(faces.size()*3);
    for (size_t i = 0; i < faces.size(); i++) {
        for (size_t j = 0; j < faces[i].size(); j++) {
            vertices[k++] = points[faces[i][j]];
        }
    }
    obj_ptr->vertices = vertices;
    obj_ptr->position = Eigen::Map<Eigen::MatrixXf>((float *)vertices.data(), 3, vertices.size()).rowwise().mean();

    // Populate vertex normals
    if (points.size() == point_normals.size()) {
        k = 0;
        std::vector<Eigen::Vector3f> vertex_normals(faces.size()*3);
        for (size_t i = 0; i < faces.size(); i++) {
            for (size_t j = 0; j < faces[i].size(); j++) {
                vertex_normals[k++] = point_normals[faces[i][j]];
            }
        }
        obj_ptr->vertexNormals = vertex_normals;
    }

    // Populate face normals
    if (faces.size() == face_normals.size()) {
        k = 0;
        std::vector<Eigen::Vector3f> face_normals_flat(faces.size()*3);
        for (size_t i = 0; i < faces.size(); i++) {
            face_normals_flat[k++] = face_normals[i];
            face_normals_flat[k++] = face_normals[i];
            face_normals_flat[k++] = face_normals[i];
        }
        obj_ptr->faceNormals = face_normals_flat;
    } else {
        k = 0;
        std::vector<Eigen::Vector3f> face_normals_flat(faces.size()*3);
        for (size_t i = 0; i < faces.size(); i++) {
            Eigen::Vector3f pt0(points[faces[i][0]]), pt1(points[faces[i][1]]), pt2(points[faces[i][2]]);
            Eigen::Vector3f normal = ((pt1-pt0).cross(pt2-pt0)).normalized();
            face_normals_flat[k++] = normal;
            face_normals_flat[k++] = normal;
            face_normals_flat[k++] = normal;
        }
        obj_ptr->faceNormals = face_normals_flat;
    }

    ((Renderable_ *)obj_ptr)->applyRenderingProperties(rp);
}

void Visualizer::addTriangleMesh(const std::string &name, const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &point_normals, const std::vector<std::vector<size_t> > &faces, const RenderingProperties &rp) {
    addTriangleMesh(name, points, point_normals, faces, std::vector<Eigen::Vector3f>(0), rp);
}

void Visualizer::addTriangleMesh(const std::string &name, const std::vector<Eigen::Vector3f> &points, const std::vector<std::vector<size_t> > &faces, const std::vector<Eigen::Vector3f> &face_normals, const RenderingProperties &rp) {
    addTriangleMesh(name, points, std::vector<Eigen::Vector3f>(0), faces, face_normals, rp);
}

void Visualizer::addTriangleMesh(const std::string &name, const std::vector<Eigen::Vector3f> &points, const std::vector<std::vector<size_t> > &faces, const RenderingProperties &rp) {
    addTriangleMesh(name, points, std::vector<Eigen::Vector3f>(0), faces, std::vector<Eigen::Vector3f>(0), rp);
}

void Visualizer::addTriangleMesh(const std::string &name, const PointCloud &cloud, const std::vector<std::vector<size_t> > &faces, const std::vector<Eigen::Vector3f> &face_normals, const RenderingProperties &rp) {
    addTriangleMesh(name, cloud.points, cloud.normals, faces, face_normals, rp);
}

void Visualizer::addTriangleMesh(const std::string &name, const PointCloud &cloud, const std::vector<std::vector<size_t> > &faces, const RenderingProperties &rp) {
    addTriangleMesh(name, cloud.points, cloud.normals, faces, std::vector<Eigen::Vector3f>(0), rp);
}

void Visualizer::render() const {
    // Set render sequence
    pangolin::OpenGlMatrix mv = gl_render_state_->GetModelViewMatrix();
    Eigen::Matrix3f R;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R(i,j) = mv(i,j);
        }
    }
    Eigen::Vector3f t(mv(0,3), mv(1,3), mv(2,3));

    size_t k = 0;
    std::vector<std::pair<Visualizer::Renderable_*, float> > objects(renderables_.size());
    for (auto it = renderables_.begin(); it != renderables_.end(); ++it) {
        if (it->second->visible) {
            objects[k].first = it->second.get();
            objects[k].second = (R*(it->second->position) + t).norm();
            k++;
        }
    }
    objects.resize(k);

    std::sort(objects.begin(), objects.end(), render_priority_comparator_);

    gl_context_->MakeCurrent();
    display_->Activate(*gl_render_state_);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(clear_color_(0), clear_color_(1), clear_color_(2), 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    for (size_t i = 0; i < objects.size(); i++) {
        objects[i].first->render();
    }
}

bool Visualizer::getVisibility(const std::string &name) const {
    auto it = renderables_.find(name);
    if (it == renderables_.end()) return false;
    return it->second->visible;
}

void Visualizer::setVisibility(const std::string &name, bool visible) {
    auto it = renderables_.find(name);
    if (it == renderables_.end()) return;
    it->second->visible = visible;
}

Visualizer::RenderingProperties Visualizer::getRenderingProperties(const std::string &name) const {
    auto it = renderables_.find(name);
    if (it == renderables_.end()) return RenderingProperties();
    return it->second->renderingProperties;
}

void Visualizer::setRenderingProperties(const std::string &name, const RenderingProperties &rp) {
    auto it = renderables_.find(name);
    if (it == renderables_.end()) return;
    it->second->applyRenderingProperties(rp);
}

std::vector<std::string> Visualizer::getObjectNames() const {
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

void Visualizer::wireframe_toggle_callback_(Visualizer &viz) {
    for (auto it = viz.renderables_.begin(); it != viz.renderables_.end(); ++it) {
        RenderingProperties rp = viz.getRenderingProperties(it->first);
        bool wireframe = rp.drawWireframe;
        rp.setDrawWireframe(!wireframe);
        viz.setRenderingProperties(it->first, rp);
    }
}
