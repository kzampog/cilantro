#include <cilantro/renderables.hpp>

namespace cilantro {
    extern "C" const unsigned char AnonymousPro_ttf[];

    Eigen::Vector3f RenderingProperties::defaultColor = Eigen::Vector3f(1.0f,1.0f,1.0f);
    Eigen::Vector3f RenderingProperties::noColor = Eigen::Vector3f(-1.0f,-1.0f,-1.0f);

    void PointCloudRenderable::applyRenderingProperties() {
        pointBuffer.Reinitialise(pangolin::GlArrayBuffer, points.cols(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        pointBuffer.Upload(points.data(), sizeof(float)*points.cols()*3);

        normalBuffer.Reinitialise(pangolin::GlArrayBuffer, normals.cols(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        normalBuffer.Upload(normals.data(), sizeof(float)*normals.cols()*3);

        PointSet<float,3> line_end_points;
        if (renderingProperties.drawNormals && normals.cols() > 0 && renderingProperties.lineDensityFraction > 0.0f) {
            if (renderingProperties.lineDensityFraction > 1.0f) renderingProperties.lineDensityFraction = 1.0;
            size_t step = (size_t)std::llround(1.0/renderingProperties.lineDensityFraction);
            step = std::max(step,(size_t)1);
            line_end_points.resize(Eigen::NoChange, 2*((normals.cols()-1)/step + 1));
            size_t k = 0;
            for (size_t i = 0; i < normals.cols(); i += step) {
                line_end_points.col(k++) = points.col(i);
                line_end_points.col(k++) = points.col(i) + renderingProperties.normalLength*normals.col(i);
            }
        }
        normalEndPointBuffer.Reinitialise(pangolin::GlArrayBuffer, line_end_points.cols(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        normalEndPointBuffer.Upload(line_end_points.data(), sizeof(float)*line_end_points.cols()*3);

        PointSet<float,4> color_alpha;
        if (renderingProperties.pointColor != RenderingProperties::noColor) {
            Eigen::Vector4f tmp;
            tmp.head(3) = renderingProperties.pointColor;
            tmp(3) = renderingProperties.opacity;
            color_alpha = tmp.replicate(1,points.cols());
        } else if (pointValues.cols() == points.cols() && renderingProperties.useScalarValueMappedColors) {
            PointSet<float,3> color_tmp = colormap(pointValues, renderingProperties.minScalarValue, renderingProperties.maxScalarValue, renderingProperties.colormapType);
            color_alpha.resize(Eigen::NoChange, pointValues.cols());
            color_alpha.topRows(3) = color_tmp;
            color_alpha.row(3).setConstant(renderingProperties.opacity);
        } else if (colors.cols() == points.cols()) {
            color_alpha.resize(Eigen::NoChange, colors.cols());
            color_alpha.topRows(3) = colors;
            color_alpha.row(3).setConstant(renderingProperties.opacity);
        } else {
            // Fallback to default color
            Eigen::Vector4f tmp;
            tmp.head(3) = RenderingProperties::defaultColor;
            tmp(3) = renderingProperties.opacity;
            color_alpha = tmp.replicate(1,points.cols());
        }
        colorBuffer.Reinitialise(pangolin::GlArrayBuffer, color_alpha.cols(), GL_FLOAT, 4, GL_DYNAMIC_DRAW);
        colorBuffer.Upload(color_alpha.data(), sizeof(float)*color_alpha.cols()*4);
    }

    void PointCloudRenderable::render() {
        glPointSize(renderingProperties.pointSize);

        bool use_normals = normals.cols() == points.cols();
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
        PointSet<float,3> line_end_points;
        if (srcPoints.cols() > 0 && renderingProperties.lineDensityFraction > 0.0f) {
            if (renderingProperties.lineDensityFraction > 1.0f) renderingProperties.lineDensityFraction = 1.0;
            size_t step = (size_t)std::llround(1.0/renderingProperties.lineDensityFraction);
            step = std::max(step,(size_t)1);
            line_end_points.resize(Eigen::NoChange, 2*((srcPoints.cols()-1)/step + 1));
            size_t k = 0;
            for (size_t i = 0; i < srcPoints.cols(); i += step) {
                line_end_points.col(k++) = srcPoints.col(i);
                line_end_points.col(k++) = dstPoints.col(i);
            }
        }
        lineEndPointBuffer.Reinitialise(pangolin::GlArrayBuffer, line_end_points.cols(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        lineEndPointBuffer.Upload(line_end_points.data(), sizeof(float)*line_end_points.cols()*3);
    }

    void CorrespondencesRenderable::render() {
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

    void CameraFrustumRenderable::applyRenderingProperties() {}

    void CameraFrustumRenderable::render() {
        if (renderingProperties.lineColor == RenderingProperties::noColor)
            glColor4f(RenderingProperties::defaultColor(0), RenderingProperties::defaultColor(1), RenderingProperties::defaultColor(2), renderingProperties.opacity);
        else
            glColor4f(renderingProperties.lineColor(0), renderingProperties.lineColor(1), renderingProperties.lineColor(2), renderingProperties.opacity);
        glLineWidth(renderingProperties.lineWidth);
        pangolin::glDrawFrustrum(inverseIntrinsics, (int)width, (int)height, pose, scale);
    }

    void TriangleMeshRenderable::applyRenderingProperties() {
        // Populate flattened vertices and update centroid
        size_t k = 0;
        PointSet<float,3> vertices_flat(3, faces.size()*3);
        for (size_t i = 0; i < faces.size(); i++) {
            for (size_t j = 0; j < faces[i].size(); j++) {
                vertices_flat.col(k++) = vertices.col(faces[i][j]);
            }
        }
        vertexBuffer.Reinitialise(pangolin::GlArrayBuffer, vertices_flat.cols(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        vertexBuffer.Upload(vertices_flat.data(), sizeof(float)*vertices_flat.cols()*3);

        centroid = vertices_flat.rowwise().mean();

        // Populate flattened normals
        PointSet<float,3> normals_flat;
        if (renderingProperties.useFaceNormals && faceNormals.cols() == faces.size()) {
            normals_flat.resize(Eigen::NoChange, faces.size()*3);
            k = 0;
            for (size_t i = 0; i < faces.size(); i++) {
                for (size_t j = 0; j < faces[i].size(); j++) {
                    normals_flat.col(k++) = faceNormals.col(i);
                }
            }
        }
        if (!renderingProperties.useFaceNormals && vertexNormals.cols() == vertices.cols()) {
            normals_flat.resize(Eigen::NoChange, faces.size()*3);
            k = 0;
            for (size_t i = 0; i < faces.size(); i++) {
                for (size_t j = 0; j < faces[i].size(); j++) {
                    normals_flat.col(k++) = vertexNormals.col(faces[i][j]);
                }
            }
        }
        normalBuffer.Reinitialise(pangolin::GlArrayBuffer, normals_flat.cols(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        normalBuffer.Upload(normals_flat.data(), sizeof(float)*normals_flat.cols()*3);

        PointSet<float,3> line_end_points;
        if (renderingProperties.drawNormals && renderingProperties.lineDensityFraction > 0.0f) {
            if (renderingProperties.lineDensityFraction > 1.0f) renderingProperties.lineDensityFraction = 1.0;
            size_t step = (size_t)std::llround(1.0/renderingProperties.lineDensityFraction);
            step = std::max(step,(size_t)1);
            if (renderingProperties.useFaceNormals && !faces.empty() && faceNormals.cols() == faces.size()) {
                line_end_points.resize(Eigen::NoChange, 2*((faces.size()-1)/step + 1));
                size_t k = 0;
                for (size_t i = 0; i < faces.size(); i += step) {
                    Eigen::Vector3f face_center(Eigen::Vector3f::Zero());
                    for (size_t j = 0; j < faces[i].size(); j++) {
                        face_center += vertices.col(faces[i][j]);
                    }
                    float scale = 1.0f/faces[i].size();
                    face_center *= scale;

                    line_end_points.col(k++) = face_center;
                    line_end_points.col(k++) = face_center + renderingProperties.normalLength*faceNormals.col(i);
                }
            }
            if (!renderingProperties.useFaceNormals && vertices.cols() > 0 && vertexNormals.cols() == vertices.cols()) {
                line_end_points.resize(Eigen::NoChange, 2*((vertices.cols()-1)/step + 1));
                size_t k = 0;
                for (size_t i = 0; i < vertices.cols(); i += step) {
                    line_end_points.col(k++) = vertices.col(i);
                    line_end_points.col(k++) = vertices.col(i) + renderingProperties.normalLength*vertexNormals.col(i);
                }
            }
        }
        normalEndPointBuffer.Reinitialise(pangolin::GlArrayBuffer, line_end_points.cols(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        normalEndPointBuffer.Upload(line_end_points.data(), sizeof(float)*line_end_points.cols()*3);

        // Populate flattened colors
        PointSet<float,4> colors_flat;
        if (renderingProperties.pointColor != RenderingProperties::noColor) {
            Eigen::Vector4f tmp;
            tmp.head(3) = renderingProperties.pointColor;
            tmp(3) = renderingProperties.opacity;
            colors_flat = tmp.replicate(1,faces.size()*3);
        } else if (renderingProperties.useFaceColors && (faceValues.cols() == faces.size() || faceColors.cols() == faces.size())) {
            if (faceValues.cols() == faces.size() && renderingProperties.useScalarValueMappedColors) {
                colors_flat.resize(Eigen::NoChange, faces.size()*3);
                PointSet<float,3> colors_tmp = colormap(faceValues, renderingProperties.minScalarValue, renderingProperties.maxScalarValue, renderingProperties.colormapType);
                k = 0;
                for (size_t i = 0; i < faces.size(); i++) {
                    for (size_t j = 0; j < faces[i].size(); j++) {
                        colors_flat.col(k).head(3) = colors_tmp.col(i);
                        colors_flat(3,k++) = renderingProperties.opacity;
                    }
                }
            } else if (faceColors.cols() == faces.size()) {
                colors_flat.resize(Eigen::NoChange, faces.size()*3);
                k = 0;
                for (size_t i = 0; i < faces.size(); i++) {
                    for (size_t j = 0; j < faces[i].size(); j++) {
                        colors_flat.col(k).head(3) = faceColors.col(i);
                        colors_flat(3,k++) = renderingProperties.opacity;
                    }
                }
            }
        } else if (!renderingProperties.useFaceColors && (vertexValues.cols() == vertices.cols() || vertexColors.cols() == vertices.cols())) {
            if (vertexValues.cols() == vertices.cols() && renderingProperties.useScalarValueMappedColors) {
                colors_flat.resize(Eigen::NoChange, faces.size()*3);
                PointSet<float,3> colors_tmp = colormap(vertexValues, renderingProperties.minScalarValue, renderingProperties.maxScalarValue, renderingProperties.colormapType);
                k = 0;
                for (size_t i = 0; i < faces.size(); i++) {
                    for (size_t j = 0; j < faces[i].size(); j++) {
                        colors_flat.col(k).head(3) = colors_tmp.col(faces[i][j]);
                        colors_flat(3,k++) = renderingProperties.opacity;
                    }
                }
            } else if (vertexColors.cols() == vertices.cols()) {
                colors_flat.resize(Eigen::NoChange, faces.size()*3);
                k = 0;
                for (size_t i = 0; i < faces.size(); i++) {
                    for (size_t j = 0; j < faces[i].size(); j++) {
                        colors_flat.col(k).head(3) = vertexColors.col(faces[i][j]);
                        colors_flat(3,k++) = renderingProperties.opacity;
                    }
                }
            }
        } else {
            // Fallback to default color
            Eigen::Vector4f tmp;
            tmp.head(3) = RenderingProperties::defaultColor;
            tmp(3) = renderingProperties.opacity;
            colors_flat = tmp.replicate(1,faces.size()*3);
        }
        colorBuffer.Reinitialise(pangolin::GlArrayBuffer, colors_flat.cols(), GL_FLOAT, 4, GL_DYNAMIC_DRAW);
        colorBuffer.Upload(colors_flat.data(), sizeof(float)*colors_flat.cols()*4);
    }

    void TriangleMeshRenderable::render() {
        glPointSize(renderingProperties.pointSize);
        glLineWidth(renderingProperties.lineWidth);

        bool use_normals = (renderingProperties.useFaceNormals && faceNormals.cols() == faces.size()) ||
                           (!renderingProperties.useFaceNormals && vertexNormals.cols() == vertices.cols());
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
        GLint view[4];
        GLdouble scrn[3];

#ifdef HAVE_GLES_2
        std::copy(glEngine().projection.top().m, glEngine().projection.top().m+16, projection);
        std::copy(glEngine().modelview.top().m, glEngine().modelview.top().m+16, modelview);
#else
        glGetDoublev(GL_PROJECTION_MATRIX, projection);
        glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
#endif
        glGetIntegerv(GL_VIEWPORT, view);

        pangolin::glProject(centroid[0], centroid[1], centroid[2], modelview, projection, view, scrn, scrn+1, scrn+2);

        pangolin::DisplayBase().Activate();
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(-0.5, pangolin::DisplayBase().v.w-0.5, -0.5, pangolin::DisplayBase().v.h-0.5, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        const Eigen::Vector2f& anchor = renderingProperties.textAnchorPoint;
        glTranslatef((GLfloat)scrn[0]-anchor[0]*glText.Width(), (GLfloat)scrn[1]-anchor[1]*glText.Height(), (GLfloat)scrn[2]);
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
