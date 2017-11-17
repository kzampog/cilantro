#include <cilantro/renderables.hpp>

namespace cilantro {
    extern "C" const unsigned char AnonymousPro_ttf[];

    Eigen::Vector3f RenderingProperties::defaultColor = Eigen::Vector3f(1.0f,1.0f,1.0f);
    Eigen::Vector3f RenderingProperties::noColor = Eigen::Vector3f(-1.0f,-1.0f,-1.0f);

    void PointCloudRenderable::applyRenderingProperties() {
        pointBuffer.Reinitialise(pangolin::GlArrayBuffer, points.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        pointBuffer.Upload(points.data(), sizeof(float)*points.size()*3);

        normalBuffer.Reinitialise(pangolin::GlArrayBuffer, normals.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        normalBuffer.Upload(normals.data(), sizeof(float)*normals.size()*3);

        std::vector<Eigen::Vector3f> line_end_points;
        if (renderingProperties.drawNormals && renderingProperties.lineDensityFraction > 0.0f) {
            if (renderingProperties.lineDensityFraction > 1.0f) renderingProperties.lineDensityFraction = 1.0;
            size_t step = (size_t)std::llround(1.0/renderingProperties.lineDensityFraction);
            step = std::max(step,(size_t)1);

            line_end_points.reserve((size_t)(std::ceil(renderingProperties.lineDensityFraction*normals.size())) + 1);
            for (size_t i = 0; i < normals.size(); i += step) {
                line_end_points.emplace_back(points[i]);
                line_end_points.emplace_back(points[i] + renderingProperties.normalLength*normals[i]);
            }
        }
        normalEndPointBuffer.Reinitialise(pangolin::GlArrayBuffer, line_end_points.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        normalEndPointBuffer.Upload(line_end_points.data(), sizeof(float)*line_end_points.size()*3);

        std::vector<Eigen::Vector4f> color_alpha;
        if (renderingProperties.pointColor != RenderingProperties::noColor) {
            Eigen::Vector4f tmp;
            tmp.head(3) = renderingProperties.pointColor;
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

    void PointCloudRenderable::render() {
        glPointSize(renderingProperties.pointSize);

        bool use_normals = normals.size() == points.size();
        bool use_lighting = use_normals && renderingProperties.useLighting;

        if (use_normals) {
            normalBuffer.Bind();
            glNormalPointer(normalBuffer.datatype, (GLsizei)(normalBuffer.count_per_element * pangolin::GlDataTypeBytes(normalBuffer.datatype)), NULL);
            glEnableClientState(GL_NORMAL_ARRAY);
        }
        if (use_lighting) {
            glEnable(GL_LIGHTING);
            glEnable(GL_LIGHT0);
            glEnable(GL_COLOR_MATERIAL);
            pangolin::RenderVboCbo(pointBuffer, colorBuffer, true, GL_POINTS);
            glDisable(GL_COLOR_MATERIAL);
            glDisable(GL_LIGHT0);
            glDisable(GL_LIGHTING);
        } else {
            pangolin::RenderVboCbo(pointBuffer, colorBuffer, true, GL_POINTS);
        }
        if (use_normals) {
            glDisableClientState(GL_NORMAL_ARRAY);
            normalBuffer.Unbind();
        }

        if (renderingProperties.drawNormals) {
            if (renderingProperties.lineColor == RenderingProperties::noColor)
                glColor4f(RenderingProperties::defaultColor(0), RenderingProperties::defaultColor(1), RenderingProperties::defaultColor(2), renderingProperties.opacity);
            else
                glColor4f(renderingProperties.lineColor(0), renderingProperties.lineColor(1), renderingProperties.lineColor(2), renderingProperties.opacity);
            glLineWidth(renderingProperties.lineWidth);
            pangolin::RenderVbo(normalEndPointBuffer, GL_LINES);
        }
    }

    void CorrespondencesRenderable::applyRenderingProperties() {
        std::vector<Eigen::Vector3f> line_end_points;
        if (renderingProperties.lineDensityFraction > 0.0f) {
            if (renderingProperties.lineDensityFraction > 1.0f) renderingProperties.lineDensityFraction = 1.0;
            size_t step = (size_t)std::llround(1.0/renderingProperties.lineDensityFraction);
            step = std::max(step,(size_t)1);

            line_end_points.reserve((size_t)(std::ceil(renderingProperties.lineDensityFraction*srcPoints.size())) + 1);
            for (size_t i = 0; i < srcPoints.size(); i += step) {
                line_end_points.emplace_back(srcPoints[i]);
                line_end_points.emplace_back(dstPoints[i]);
            }
        }
        lineEndPointBuffer.Reinitialise(pangolin::GlArrayBuffer, line_end_points.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        lineEndPointBuffer.Upload(line_end_points.data(), sizeof(float)*line_end_points.size()*3);
    }

    void CorrespondencesRenderable::render() {
        glPointSize(renderingProperties.pointSize);
        if (renderingProperties.lineColor == RenderingProperties::noColor)
            glColor4f(RenderingProperties::defaultColor(0), RenderingProperties::defaultColor(1), RenderingProperties::defaultColor(2), renderingProperties.opacity);
        else
            glColor4f(renderingProperties.lineColor(0), renderingProperties.lineColor(1), renderingProperties.lineColor(2), renderingProperties.opacity);
        glLineWidth(renderingProperties.lineWidth);
        pangolin::RenderVbo(lineEndPointBuffer, GL_LINES);
    }

    void CoordinateFrameRenderable::applyRenderingProperties() {}

    void CoordinateFrameRenderable::render() {
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
        vertexBuffer.Reinitialise(pangolin::GlArrayBuffer, vertices_flat.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        vertexBuffer.Upload(vertices_flat.data(), sizeof(float)*vertices_flat.size()*3);

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
        normalBuffer.Reinitialise(pangolin::GlArrayBuffer, normals_flat.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        normalBuffer.Upload(normals_flat.data(), sizeof(float)*normals_flat.size()*3);

        std::vector<Eigen::Vector3f> line_end_points;
        if (renderingProperties.drawNormals && renderingProperties.lineDensityFraction > 0.0f) {
            if (renderingProperties.lineDensityFraction > 1.0f) renderingProperties.lineDensityFraction = 1.0;
            size_t step = (size_t)std::llround(1.0/renderingProperties.lineDensityFraction);
            step = std::max(step,(size_t)1);

            if (renderingProperties.useFaceNormals && faceNormals.size() == faces.size()) {
                line_end_points.reserve((size_t)(std::ceil(renderingProperties.lineDensityFraction*faces.size())) + 1);
                for (size_t i = 0; i < faces.size(); i += step) {
                    Eigen::Vector3f face_center(Eigen::Vector3f::Zero());
                    for (size_t j = 0; j < faces[i].size(); j++) {
                        face_center += vertices[faces[i][j]];
                    }
                    float scale = 1.0f/faces[i].size();
                    face_center *= scale;

                    line_end_points.emplace_back(face_center);
                    line_end_points.emplace_back(face_center + renderingProperties.normalLength*faceNormals[i]);
                }
            }
            if (!renderingProperties.useFaceNormals && vertexNormals.size() == vertices.size()) {
                line_end_points.reserve((size_t)(std::ceil(renderingProperties.lineDensityFraction*vertices.size())) + 1);
                for (size_t i = 0; i < vertices.size(); i += step) {
                    line_end_points.emplace_back(vertices[i]);
                    line_end_points.emplace_back(vertices[i] + renderingProperties.normalLength*vertexNormals[i]);
                }
            }
        }
        normalEndPointBuffer.Reinitialise(pangolin::GlArrayBuffer, line_end_points.size(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        normalEndPointBuffer.Upload(line_end_points.data(), sizeof(float)*line_end_points.size()*3);

        // Populate flattened colors
        std::vector<Eigen::Vector4f> colors_flat;
        if (renderingProperties.pointColor != RenderingProperties::noColor) {
            Eigen::Vector4f tmp;
            tmp.head(3) = renderingProperties.pointColor;
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
        colorBuffer.Reinitialise(pangolin::GlArrayBuffer, colors_flat.size(), GL_FLOAT, 4, GL_DYNAMIC_DRAW);
        colorBuffer.Upload(colors_flat.data(), sizeof(float)*colors_flat.size()*4);
    }

    void TriangleMeshRenderable::render() {
        glPointSize(renderingProperties.pointSize);
        glLineWidth(renderingProperties.lineWidth);

        bool use_normals = (renderingProperties.useFaceNormals && faceNormals.size() == faces.size()) ||
                           (!renderingProperties.useFaceNormals && vertexNormals.size() == vertices.size());
        bool use_lighting = use_normals && renderingProperties.useLighting;

        if (use_normals) {
            normalBuffer.Bind();
            glNormalPointer(normalBuffer.datatype, (GLsizei)(normalBuffer.count_per_element * pangolin::GlDataTypeBytes(normalBuffer.datatype)), NULL);
            glEnableClientState(GL_NORMAL_ARRAY);
        }
        if (renderingProperties.drawWireframe) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            pangolin::RenderVboCbo(vertexBuffer, colorBuffer, true, GL_TRIANGLES);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        } else {
            if (use_lighting) {
                glEnable(GL_LIGHTING);
                glEnable(GL_LIGHT0);
                glEnable(GL_COLOR_MATERIAL);
                pangolin::RenderVboCbo(vertexBuffer, colorBuffer, true, GL_TRIANGLES);
                glDisable(GL_COLOR_MATERIAL);
                glDisable(GL_LIGHT0);
                glDisable(GL_LIGHTING);
            } else {
                pangolin::RenderVboCbo(vertexBuffer, colorBuffer, true, GL_TRIANGLES);
            }
        }
        if (use_normals) {
            glDisableClientState(GL_NORMAL_ARRAY);
            normalBuffer.Unbind();
        }

        if (renderingProperties.drawNormals) {
            if (renderingProperties.lineColor == RenderingProperties::noColor)
                glColor4f(RenderingProperties::defaultColor(0), RenderingProperties::defaultColor(1), RenderingProperties::defaultColor(2), renderingProperties.opacity);
            else
                glColor4f(renderingProperties.lineColor(0), renderingProperties.lineColor(1), renderingProperties.lineColor(2), renderingProperties.opacity);
            glLineWidth(renderingProperties.lineWidth);
            pangolin::RenderVbo(normalEndPointBuffer, GL_LINES);
        }
    }

    void TextRenderable::applyRenderingProperties() {
        glFont.reset(new pangolin::GlFont(AnonymousPro_ttf, renderingProperties.fontSize));
        glText = glFont->Text(text);
    }

    void TextRenderable::render() {
        if (renderingProperties.pointColor == RenderingProperties::noColor)
            glColor4f(RenderingProperties::defaultColor(0), RenderingProperties::defaultColor(1), RenderingProperties::defaultColor(2), renderingProperties.opacity);
        else
            glColor4f(renderingProperties.pointColor(0), renderingProperties.pointColor(1), renderingProperties.pointColor(2), renderingProperties.opacity);

        // find object point (x,y,z)' in pixel coords
        GLdouble projection[16];
        GLdouble modelview[16];
        GLint    view[4];
        GLdouble scrn[3];

#ifdef HAVE_GLES_2
        std::copy(glEngine().projection.top().m, glEngine().projection.top().m+16, projection);
    std::copy(glEngine().modelview.top().m, glEngine().modelview.top().m+16, modelview);
#else
        glGetDoublev(GL_PROJECTION_MATRIX, projection );
        glGetDoublev(GL_MODELVIEW_MATRIX, modelview );
#endif
        glGetIntegerv(GL_VIEWPORT, view );

        pangolin::glProject(centroid[0], centroid[1], centroid[2], modelview, projection, view, scrn, scrn + 1, scrn + 2);

        pangolin::DisplayBase().Activate();
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(-0.5, pangolin::DisplayBase().v.w-0.5, -0.5, pangolin::DisplayBase().v.h-0.5, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        const Eigen::Vector2f& anchor = renderingProperties.textAnchorPoint;
        glTranslatef((GLfloat)scrn[0] - anchor[0]*glText.Width(), (GLfloat)scrn[1] - anchor[1]*glText.Height(), (GLfloat)scrn[2]);
        glText.Draw();

        // Restore viewport
        glViewport(view[0],view[1],view[2],view[3]);

        // Restore modelview / project matrices
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
    }
}
