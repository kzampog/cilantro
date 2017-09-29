#include <cilantro/renderables.hpp>

Eigen::Vector3f RenderingProperties::defaultColor = Eigen::Vector3f(1.0f,1.0f,1.0f);
Eigen::Vector3f RenderingProperties::noColor = Eigen::Vector3f(-1.0f,-1.0f,-1.0f);

void PointsRenderable::applyRenderingProperties() {
    pointBuffer.Reinitialise(pangolin::GlArrayBuffer, points.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
    pointBuffer.Upload(points.data(), sizeof(float)*points.size()*3);

    std::vector<Eigen::Vector4f> color_alpha;
    if (renderingProperties.drawingColor != RenderingProperties::noColor) {
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
        tmp.head(3) = RenderingProperties::defaultColor;
        tmp(3) = renderingProperties.opacity;
        color_alpha = std::vector<Eigen::Vector4f>(points.size(), tmp);
    }
    colorBuffer.Reinitialise(pangolin::GlArrayBuffer, color_alpha.size(), GL_FLOAT, 4, GL_DYNAMIC_DRAW);
    colorBuffer.Upload(color_alpha.data(), sizeof(float)*color_alpha.size()*4);
}

void PointsRenderable::render() {
    glPointSize(renderingProperties.pointSize);
    pangolin::RenderVboCbo(pointBuffer, colorBuffer);
}

void NormalsRenderable::applyRenderingProperties() {
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

void NormalsRenderable::render() {
    glPointSize(renderingProperties.pointSize);
    if (renderingProperties.drawingColor == RenderingProperties::noColor)
        glColor4f(RenderingProperties::defaultColor(0), RenderingProperties::defaultColor(1), RenderingProperties::defaultColor(2), renderingProperties.opacity);
    else
        glColor4f(renderingProperties.drawingColor(0), renderingProperties.drawingColor(1), renderingProperties.drawingColor(2), renderingProperties.opacity);
    glLineWidth(renderingProperties.lineWidth);
    pangolin::RenderVbo(lineEndPointBuffer, GL_LINES);
}

void CorrespondencesRenderable::applyRenderingProperties() {
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

void CorrespondencesRenderable::render() {
    glPointSize(renderingProperties.pointSize);
    if (renderingProperties.drawingColor == RenderingProperties::noColor)
        glColor4f(RenderingProperties::defaultColor(0), RenderingProperties::defaultColor(1), RenderingProperties::defaultColor(2), renderingProperties.opacity);
    else
        glColor4f(renderingProperties.drawingColor(0), renderingProperties.drawingColor(1), renderingProperties.drawingColor(2), renderingProperties.opacity);
    glLineWidth(renderingProperties.lineWidth);
    pangolin::RenderVbo(lineEndPointBuffer, GL_LINES);
}

void AxisRenderable::applyRenderingProperties() {}

void AxisRenderable::render() {
    glLineWidth(renderingProperties.lineWidth);
    pangolin::glDrawAxis<Eigen::Matrix4f,float>(transform, scale);
}

void TriangleMeshRenderable::applyRenderingProperties() {
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
    if (renderingProperties.drawingColor != RenderingProperties::noColor) {
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
        tmp.head(3) = RenderingProperties::defaultColor;
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

void TriangleMeshRenderable::render() {
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
