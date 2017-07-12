#include <cilantro/visualizer.hpp>

Eigen::Vector3f Visualizer::no_color_ = Eigen::Vector3f(-1.0f,-1.0f,-1.0f);
Eigen::Vector3f Visualizer::default_color_ = Eigen::Vector3f(1.0f,1.0f,1.0f);

void Visualizer::PointsRenderable_::applyRenderingProperties() {
    pointBuffer.Reinitialise(pangolin::GlArrayBuffer, points.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
    pointBuffer.Upload(points.data(), sizeof(float)*points.size()*3);

    std::vector<Eigen::Vector4f> color_alpha;
    if (renderingProperties.drawingColor != no_color_) {
        Eigen::Vector4f tmp;
        tmp.head(3) = renderingProperties.drawingColor;
        tmp(3) = renderingProperties.opacity;
        color_alpha = std::vector<Eigen::Vector4f>(points.size(), tmp);
    } else if (pointValues.size() == points.size() && renderingProperties.colormapType != ColormapType::NONE) {
        std::vector<Eigen::Vector3f> color_tmp = colormap(pointValues, renderingProperties.minScalarValue, renderingProperties.maxScalarValue, renderingProperties.colormapType);
        color_alpha.resize(pointValues.size());
        for(size_t i = 0; i < pointValues.size(); i++) {
            color_alpha[i].head(3) = color_tmp[i];
            color_alpha[i](3) = renderingProperties.opacity;
        }
    } else if (colors.size() == points.size()) {
        color_alpha.resize(colors.size());
        for(size_t i = 0; i < colors.size(); i++) {
            color_alpha[i].head(3) = colors[i];
            color_alpha[i](3) = renderingProperties.opacity;
        }
    } else {
        // Fallback to default color
        Eigen::Vector4f tmp;
        tmp.head(3) = default_color_;
        tmp(3) = renderingProperties.opacity;
        color_alpha = std::vector<Eigen::Vector4f>(points.size(), tmp);
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
    pangolin::glDrawAxis<Eigen::Matrix4f,float>(transform, scale);
}

void Visualizer::TriangleMeshRenderable_::applyRenderingProperties() {
    // Populate flattened vertices and update centroid
    size_t k = 0;
    std::vector<Eigen::Vector3f> vertices_flat(faces.size()*3);
    for (size_t i = 0; i < faces.size(); i++) {
        for (size_t j = 0; j < faces[i].size(); j++) {
            vertices_flat[k++] = vertices[faces[i][j]];
        }
    }
    centroid = Eigen::Map<Eigen::MatrixXf>((float *)vertices_flat.data(), 3, vertices_flat.size()).rowwise().mean();

    // Populate flattened normals
    std::vector<Eigen::Vector3f> normals_flat;
    if (renderingProperties.useFaceNormals && faceNormals.size() == faces.size()) {
        normals_flat.resize(faces.size()*3);
        k = 0;
        for (size_t i = 0; i < faces.size(); i++) {
            for (size_t j = 0; j < faces[i].size(); j++) {
                normals_flat[k++] = faceNormals[i];
            }
        }
    }
    if (!renderingProperties.useFaceNormals && vertexNormals.size() == vertices.size()) {
        normals_flat.resize(faces.size()*3);
        k = 0;
        for (size_t i = 0; i < faces.size(); i++) {
            for (size_t j = 0; j < faces[i].size(); j++) {
                normals_flat[k++] = vertexNormals[faces[i][j]];
            }
        }
    }

    // Populate flattened colors
    std::vector<Eigen::Vector4f> colors_flat;
    if (renderingProperties.drawingColor != no_color_) {
        Eigen::Vector4f tmp;
        tmp.head(3) = renderingProperties.drawingColor;
        tmp(3) = renderingProperties.opacity;
        colors_flat = std::vector<Eigen::Vector4f>(faces.size()*3, tmp);
    } else if (renderingProperties.useFaceColors && (faceValues.size() == faces.size() || faceColors.size() == faces.size())) {
        if (faceValues.size() == faces.size() && renderingProperties.colormapType != ColormapType::NONE) {
            colors_flat.resize(faces.size()*3);
            std::vector<Eigen::Vector3f> colors_tmp = colormap(faceValues, renderingProperties.minScalarValue, renderingProperties.maxScalarValue, renderingProperties.colormapType);
            k = 0;
            for (size_t i = 0; i < faces.size(); i++) {
                for (size_t j = 0; j < faces[i].size(); j++) {
                    colors_flat[k].head(3) = colors_tmp[i];
                    colors_flat[k++](3) = renderingProperties.opacity;
                }
            }
        } else if (faceColors.size() == faces.size()) {
            colors_flat.resize(faces.size()*3);
            k = 0;
            for (size_t i = 0; i < faces.size(); i++) {
                for (size_t j = 0; j < faces[i].size(); j++) {
                    colors_flat[k].head(3) = faceColors[i];
                    colors_flat[k++](3) = renderingProperties.opacity;
                }
            }
        }
    } else if (!renderingProperties.useFaceColors && (vertexValues.size() == vertices.size() || vertexColors.size() == vertices.size())) {
        if (vertexValues.size() == vertices.size() && renderingProperties.colormapType != ColormapType::NONE) {
            colors_flat.resize(faces.size()*3);
            std::vector<Eigen::Vector3f> colors_tmp = colormap(vertexValues, renderingProperties.minScalarValue, renderingProperties.maxScalarValue, renderingProperties.colormapType);
            k = 0;
            for (size_t i = 0; i < faces.size(); i++) {
                for (size_t j = 0; j < faces[i].size(); j++) {
                    colors_flat[k].head(3) = colors_tmp[faces[i][j]];
                    colors_flat[k++](3) = renderingProperties.opacity;
                }
            }
        } else if (vertexColors.size() == vertices.size()) {
            colors_flat.resize(faces.size()*3);
            k = 0;
            for (size_t i = 0; i < faces.size(); i++) {
                for (size_t j = 0; j < faces[i].size(); j++) {
                    colors_flat[k].head(3) = vertexColors[faces[i][j]];
                    colors_flat[k++](3) = renderingProperties.opacity;
                }
            }
        }
    } else {
        // Fallback to default color
        Eigen::Vector4f tmp;
        tmp.head(3) = default_color_;
        tmp(3) = renderingProperties.opacity;
        colors_flat = std::vector<Eigen::Vector4f>(faces.size()*3, tmp);
    }

    vertexBuffer.Reinitialise(pangolin::GlArrayBuffer, vertices_flat.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
    vertexBuffer.Upload(vertices_flat.data(), sizeof(float)*vertices_flat.size()*3);
    normalBuffer.Reinitialise(pangolin::GlArrayBuffer, normals_flat.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
    normalBuffer.Upload(normals_flat.data(), sizeof(float)*normals_flat.size()*3);
    colorBuffer.Reinitialise(pangolin::GlArrayBuffer, colors_flat.size(), GL_FLOAT, 4, GL_DYNAMIC_DRAW);
    colorBuffer.Upload(colors_flat.data(), sizeof(float)*colors_flat.size()*4);
}

void Visualizer::TriangleMeshRenderable_::render() {
    glPointSize(renderingProperties.pointSize);
    glLineWidth(renderingProperties.lineWidth);

    bool use_normals = (renderingProperties.useFaceNormals && faceNormals.size() == faces.size()) ||
            (!renderingProperties.useFaceNormals && vertexNormals.size() == vertices.size());

    if (use_normals) {
        normalBuffer.Bind();
        glNormalPointer(normalBuffer.datatype, (GLsizei)(normalBuffer.count_per_element * pangolin::GlDataTypeBytes(normalBuffer.datatype)), 0);
        glEnableClientState(GL_NORMAL_ARRAY);
    }
    if (renderingProperties.drawWireframe) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        pangolin::RenderVboCbo(vertexBuffer, colorBuffer, true, GL_TRIANGLES);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    } else {
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        glEnable(GL_COLOR_MATERIAL);
        pangolin::RenderVboCbo(vertexBuffer, colorBuffer, true, GL_TRIANGLES);
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

Visualizer::~Visualizer() {}

Visualizer& Visualizer::addPointCloud(const std::string &name, const PointCloud &cloud, const RenderingProperties &rp) {
    gl_context_->MakeCurrent();
    if (cloud.empty()) {
        renderables_.erase(name);
        return *this;
    }
    renderables_[name] = std::shared_ptr<PointsRenderable_>(new PointsRenderable_);
    PointsRenderable_ *obj_ptr = (PointsRenderable_ *)renderables_[name].get();
    // Copy fields
    obj_ptr->points = cloud.points;
    if (cloud.hasColors()) obj_ptr->colors = cloud.colors;
    obj_ptr->centroid = Eigen::Map<Eigen::MatrixXf>((float *)cloud.points.data(), 3, cloud.points.size()).rowwise().mean();
    // Update buffers
    ((Renderable_ *)obj_ptr)->applyRenderingProperties(rp);

    return *this;
}

Visualizer& Visualizer::addPointCloud(const std::string &name, const std::vector<Eigen::Vector3f> &points, const RenderingProperties &rp) {
    gl_context_->MakeCurrent();
    if (points.empty()) {
        renderables_.erase(name);
        return *this;
    }
    renderables_[name] = std::shared_ptr<PointsRenderable_>(new PointsRenderable_);
    PointsRenderable_ *obj_ptr = (PointsRenderable_ *)renderables_[name].get();
    // Copy fields
    obj_ptr->points = points;
    obj_ptr->centroid = Eigen::Map<Eigen::MatrixXf>((float *)points.data(), 3, points.size()).rowwise().mean();
    // Update buffers
    ((Renderable_ *)obj_ptr)->applyRenderingProperties(rp);

    return *this;
}

Visualizer& Visualizer::addPointCloudColors(const std::string &name, const std::vector<Eigen::Vector3f> &colors) {
    gl_context_->MakeCurrent();
    auto it = renderables_.find(name);
    if (it == renderables_.end()) return *this;
    PointsRenderable_ *obj_ptr = (PointsRenderable_ *)it->second.get();
    if (colors.size() != obj_ptr->points.size()) return *this;
    obj_ptr->colors = colors;
    obj_ptr->applyRenderingProperties();

    return *this;
}

Visualizer& Visualizer::addPointCloudValues(const std::string &name, const std::vector<float> &point_values) {
    gl_context_->MakeCurrent();
    auto it = renderables_.find(name);
    if (it == renderables_.end()) return *this;
    PointsRenderable_ *obj_ptr = (PointsRenderable_ *)it->second.get();
    if (point_values.size() != obj_ptr->points.size()) return *this;
    obj_ptr->pointValues = point_values;
    obj_ptr->applyRenderingProperties();

    return *this;
}

Visualizer& Visualizer::addPointCloudNormals(const std::string &name, const std::vector<Eigen::Vector3f> &points, const std::vector<Eigen::Vector3f> &normals, const RenderingProperties &rp) {
    gl_context_->MakeCurrent();
    if (points.empty() || points.size() != normals.size()) {
        renderables_.erase(name);
        return *this;
    }
    renderables_[name] = std::shared_ptr<NormalsRenderable_>(new NormalsRenderable_);
    NormalsRenderable_ *obj_ptr = (NormalsRenderable_ *)renderables_[name].get();
    // Copy fields
    obj_ptr->points = points;
    obj_ptr->normals = normals;
    obj_ptr->centroid = Eigen::Map<Eigen::MatrixXf>((float *)points.data(), 3, points.size()).rowwise().mean();
    // Update buffers
    ((Renderable_ *)obj_ptr)->applyRenderingProperties(rp);

    return *this;
}

Visualizer& Visualizer::addPointCloudNormals(const std::string &name, const PointCloud &cloud, const RenderingProperties &rp) {
    return addPointCloudNormals(name, cloud.points, cloud.normals, rp);
}

Visualizer& Visualizer::addPointCorrespondences(const std::string &name, const std::vector<Eigen::Vector3f> &points_src, const std::vector<Eigen::Vector3f> &points_dst, const RenderingProperties &rp) {
    gl_context_->MakeCurrent();
    if (points_src.empty() || points_src.size() != points_dst.size()) {
        renderables_.erase(name);
        return *this;
    }
    renderables_[name] = std::shared_ptr<CorrespondencesRenderable_>(new CorrespondencesRenderable_);
    CorrespondencesRenderable_ *obj_ptr = (CorrespondencesRenderable_ *)renderables_[name].get();
    // Copy fields
    obj_ptr->srcPoints = points_src;
    obj_ptr->dstPoints = points_dst;
    obj_ptr->centroid = (Eigen::Map<Eigen::MatrixXf>((float *)points_src.data(), 3, points_src.size()).rowwise().mean() +
                         Eigen::Map<Eigen::MatrixXf>((float *)points_dst.data(), 3, points_dst.size()).rowwise().mean()) / 2.0f;
    // Update buffers
    ((Renderable_ *)obj_ptr)->applyRenderingProperties(rp);

    return *this;
}

Visualizer& Visualizer::addPointCorrespondences(const std::string &name, const PointCloud &cloud_src, const PointCloud &cloud_dst, const RenderingProperties &rp) {
    return addPointCorrespondences(name, cloud_src.points, cloud_dst.points, rp);
}

Visualizer& Visualizer::addCoordinateSystem(const std::string &name, float scale, const Eigen::Matrix4f &tf, const RenderingProperties &rp) {
    gl_context_->MakeCurrent();
    renderables_[name] = std::shared_ptr<AxisRenderable_>(new AxisRenderable_);
    AxisRenderable_ *obj_ptr = (AxisRenderable_ *)renderables_[name].get();
    // Copy fields
    obj_ptr->scale = scale;
    obj_ptr->transform = tf;
    obj_ptr->centroid = tf.topRightCorner(3,1);
    // Update buffers
    ((Renderable_ *)obj_ptr)->applyRenderingProperties(rp);

    return *this;
}

Visualizer& Visualizer::addTriangleMesh(const std::string &name, const PointCloud &cloud, const std::vector<std::vector<size_t> > &faces, const RenderingProperties &rp) {
    gl_context_->MakeCurrent();
    if (cloud.empty() || faces.empty()) {
        renderables_.erase(name);
        return *this;
    }
    renderables_[name] = std::shared_ptr<TriangleMeshRenderable_>(new TriangleMeshRenderable_);
    TriangleMeshRenderable_ *obj_ptr = (TriangleMeshRenderable_ *)renderables_[name].get();
    obj_ptr->vertices = cloud.points;
    obj_ptr->faces = faces;
    if (cloud.hasNormals()) obj_ptr->vertexNormals = cloud.normals;
    if (cloud.hasColors()) obj_ptr->vertexColors = cloud.colors;
    obj_ptr->faceNormals.resize(faces.size());
    for (size_t i = 0; i < faces.size(); i++) {
        Eigen::Vector3f pt0(obj_ptr->vertices[faces[i][0]]), pt1(obj_ptr->vertices[faces[i][1]]), pt2(obj_ptr->vertices[faces[i][2]]);
        Eigen::Vector3f normal = ((pt1-pt0).cross(pt2-pt0)).normalized();
        obj_ptr->faceNormals[i] = normal;
    }
    ((Renderable_ *)obj_ptr)->applyRenderingProperties(rp);

    return *this;
}

Visualizer& Visualizer::addTriangleMesh(const std::string &name, const std::vector<Eigen::Vector3f> &vertices, const std::vector<std::vector<size_t> > &faces, const RenderingProperties &rp) {
    gl_context_->MakeCurrent();
    if (vertices.empty() || faces.empty()) {
        renderables_.erase(name);
        return *this;
    }
    renderables_[name] = std::shared_ptr<TriangleMeshRenderable_>(new TriangleMeshRenderable_);
    TriangleMeshRenderable_ *obj_ptr = (TriangleMeshRenderable_ *)renderables_[name].get();
    obj_ptr->vertices = vertices;
    obj_ptr->faces = faces;
    obj_ptr->faceNormals.resize(faces.size());
    for (size_t i = 0; i < faces.size(); i++) {
        Eigen::Vector3f pt0(obj_ptr->vertices[faces[i][0]]), pt1(obj_ptr->vertices[faces[i][1]]), pt2(obj_ptr->vertices[faces[i][2]]);
        Eigen::Vector3f normal = ((pt1-pt0).cross(pt2-pt0)).normalized();
        obj_ptr->faceNormals[i] = normal;
    }
    ((Renderable_ *)obj_ptr)->applyRenderingProperties(rp);

    return *this;
}

Visualizer& Visualizer::addTriangleMeshVertexNormals(const std::string &name, const std::vector<Eigen::Vector3f> &vertex_normals) {
    gl_context_->MakeCurrent();
    auto it = renderables_.find(name);
    if (it == renderables_.end()) return *this;
    TriangleMeshRenderable_ *obj_ptr = (TriangleMeshRenderable_ *)it->second.get();
    if (vertex_normals.size() != obj_ptr->vertices.size()) return *this;
    obj_ptr->vertexNormals = vertex_normals;
    obj_ptr->applyRenderingProperties();

    return *this;
}

Visualizer& Visualizer::addTriangleMeshFaceNormals(const std::string &name, const std::vector<Eigen::Vector3f> &face_normals) {
    gl_context_->MakeCurrent();
    auto it = renderables_.find(name);
    if (it == renderables_.end()) return *this;
    TriangleMeshRenderable_ *obj_ptr = (TriangleMeshRenderable_ *)it->second.get();
    if (face_normals.size() != obj_ptr->faces.size()) return *this;
    obj_ptr->faceNormals = face_normals;
    obj_ptr->applyRenderingProperties();

    return *this;
}

Visualizer& Visualizer::addTriangleMeshVertexColors(const std::string &name, const std::vector<Eigen::Vector3f> &vertex_colors) {
    gl_context_->MakeCurrent();
    auto it = renderables_.find(name);
    if (it == renderables_.end()) return *this;
    TriangleMeshRenderable_ *obj_ptr = (TriangleMeshRenderable_ *)it->second.get();
    if (vertex_colors.size() != obj_ptr->vertices.size()) return *this;
    obj_ptr->vertexColors = vertex_colors;
    obj_ptr->applyRenderingProperties();

    return *this;
}

Visualizer& Visualizer::addTriangleMeshFaceColors(const std::string &name, const std::vector<Eigen::Vector3f> &face_colors) {
    gl_context_->MakeCurrent();
    auto it = renderables_.find(name);
    if (it == renderables_.end()) return *this;
    TriangleMeshRenderable_ *obj_ptr = (TriangleMeshRenderable_ *)it->second.get();
    if (face_colors.size() != obj_ptr->faces.size()) return *this;
    obj_ptr->faceColors = face_colors;
    obj_ptr->applyRenderingProperties();

    return *this;
}

Visualizer& Visualizer::addTriangleMeshVertexValues(const std::string &name, const std::vector<float> &vertex_values) {
    gl_context_->MakeCurrent();
    auto it = renderables_.find(name);
    if (it == renderables_.end()) return *this;
    TriangleMeshRenderable_ *obj_ptr = (TriangleMeshRenderable_ *)it->second.get();
    if (vertex_values.size() != obj_ptr->vertices.size()) return *this;
    obj_ptr->vertexValues = vertex_values;
    obj_ptr->applyRenderingProperties();

    return *this;
}

Visualizer& Visualizer::addTriangleMeshFaceValues(const std::string &name, const std::vector<float> &face_values) {
    gl_context_->MakeCurrent();
    auto it = renderables_.find(name);
    if (it == renderables_.end()) return *this;
    TriangleMeshRenderable_ *obj_ptr = (TriangleMeshRenderable_ *)it->second.get();
    if (face_values.size() != obj_ptr->faces.size()) return *this;
    obj_ptr->faceValues = face_values;
    obj_ptr->applyRenderingProperties();

    return *this;
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
            objects[k].second = (R*(it->second->centroid) + t).norm();
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
//    glClearColor(clear_color_(0), clear_color_(1), clear_color_(2), 1.0f);
//    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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
    gl_context_->MakeCurrent();
    it->second->applyRenderingProperties(rp);
}

std::vector<std::string> Visualizer::getObjectNames() const {
    std::vector<std::string> res;
    for (auto it = renderables_.begin(); it != renderables_.end(); ++it) {
        res.push_back(it->first);
    }
    return res;
}

Visualizer& Visualizer::setProjectionMatrix(int w, int h, pangolin::GLprecision fu, pangolin::GLprecision fv, pangolin::GLprecision u0, pangolin::GLprecision v0, pangolin::GLprecision zNear, pangolin::GLprecision zFar) {
    gl_context_->MakeCurrent();
    gl_render_state_->SetProjectionMatrix(pangolin::ProjectionMatrix(w, h, fu, fv, u0, v0, zNear, zFar));
    display_->SetAspect(-(double)w/((double)h));

    return *this;
}

Visualizer& Visualizer::registerKeyboardCallback(const std::vector<int> &keys, std::function<void(Visualizer&,int,void*)> func, void *cookie) {
    gl_context_->MakeCurrent();
    for (size_t i = 0; i < keys.size(); i++) {
        auto fun = std::bind(func, std::ref(*this), keys[i], cookie);
        pangolin::RegisterKeyPressCallback(keys[i], fun);
    }

    return *this;
}

void Visualizer::point_size_callback_(Visualizer &viz, int key) {
    if (key == '+') {
        for (auto it = viz.renderables_.begin(); it != viz.renderables_.end(); ++it) {
            RenderingProperties rp = viz.getRenderingProperties(it->first);
            rp.pointSize += 1.0f;
            rp.lineWidth += 1.0f;
            viz.setRenderingProperties(it->first, rp);
        }
    } else if (key == '-') {
        for (auto it = viz.renderables_.begin(); it != viz.renderables_.end(); ++it) {
            RenderingProperties rp = viz.getRenderingProperties(it->first);
            rp.pointSize = std::max(rp.pointSize - 1.0f, 1.0f);
            rp.lineWidth = std::max(rp.lineWidth - 1.0f, 1.0f);
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
