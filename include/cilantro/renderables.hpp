#pragma once

#include <cilantro/colormap.hpp>
#include <pangolin/pangolin.h>

namespace cilantro {
    struct RenderingProperties {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        inline RenderingProperties() : pointColor(noColor),
                                       lineColor(noColor),
                                       pointSize(2.0f),
                                       lineWidth(1.0f),
                                       opacity(1.0f),
                                       useLighting(true),
                                       drawNormals(false),
                                       normalLength(0.05f),
                                       lineDensityFraction(1.0),
                                       drawWireframe(false),
                                       useFaceNormals(true),
                                       useFaceColors(false),
                                       minScalarValue(std::numeric_limits<float>::quiet_NaN()),
                                       maxScalarValue(std::numeric_limits<float>::quiet_NaN()),
                                       colormapType(ColormapType::JET)
        {}
        inline ~RenderingProperties() {}

        static Eigen::Vector3f defaultColor;
        static Eigen::Vector3f noColor;

        Eigen::Vector3f pointColor;
        Eigen::Vector3f lineColor;
        float pointSize;
        float lineWidth;
        float opacity;
        bool useLighting;
        bool drawNormals;
        float normalLength;
        float lineDensityFraction;
        bool drawWireframe;
        bool useFaceNormals;
        bool useFaceColors;
        float minScalarValue;
        float maxScalarValue;
        ColormapType colormapType;

        inline RenderingProperties& setPointColor(const Eigen::Vector3f &color) { pointColor = color; return *this; }
        inline RenderingProperties& setPointColor(float r, float g, float b) { pointColor = Eigen::Vector3f(r,g,b); return *this; }
        inline RenderingProperties& setLineColor(const Eigen::Vector3f &color) { lineColor = color; return *this; }
        inline RenderingProperties& setLineColor(float r, float g, float b) { lineColor = Eigen::Vector3f(r,g,b); return *this; }
        inline RenderingProperties& setPointSize(float sz) { pointSize = sz; return *this; }
        inline RenderingProperties& setLineWidth(float lw) { lineWidth = lw; return *this; }
        inline RenderingProperties& setOpacity(float op) { opacity = op; return *this; }
        inline RenderingProperties& setUseLighting(bool ul) { useLighting = ul; return *this; }
        inline RenderingProperties& setDrawNormals(bool dn) { drawNormals = dn; return *this; }
        inline RenderingProperties& setNormalLength(float nl) { normalLength = nl; return *this; }
        inline RenderingProperties& setLineDensityFraction(float ldf) { lineDensityFraction = ldf; return *this; }
        inline RenderingProperties& setDrawWireframe(bool dw) { drawWireframe = dw; return *this; }
        inline RenderingProperties& setUseFaceNormals(bool fn) { useFaceNormals = fn; return *this; }
        inline RenderingProperties& setUseFaceColors(bool fc) { useFaceColors = fc; return *this; }
        inline RenderingProperties& setScalarValuesRange(float min, float max) { minScalarValue = min; maxScalarValue = max; return *this; }
        inline RenderingProperties& setColormapType(const ColormapType ct) { colormapType = ct; return *this; }
    };

    struct Renderable {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Renderable() : visible(true), centroid(Eigen::Vector3f::Zero()) {}

        virtual void applyRenderingProperties() = 0;    // Updates GPU buffers
        inline void applyRenderingProperties(const RenderingProperties &rp) { renderingProperties = rp; applyRenderingProperties(); }
        virtual void render() = 0;

        bool visible;
        Eigen::Vector3f centroid;                       // For render priority...
        RenderingProperties renderingProperties;
    };

    struct PointCloudRenderable : public Renderable {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        void applyRenderingProperties();
        void render();

        std::vector<Eigen::Vector3f> points;
        std::vector<Eigen::Vector3f> normals;
        std::vector<Eigen::Vector3f> colors;
        std::vector<float> pointValues;
        pangolin::GlBuffer pointBuffer;
        pangolin::GlBuffer normalBuffer;
        pangolin::GlBuffer colorBuffer;
        pangolin::GlBuffer normalEndPointBuffer;
    };

    struct CorrespondencesRenderable : public Renderable {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        void applyRenderingProperties();
        void render();

        std::vector<Eigen::Vector3f> srcPoints;
        std::vector<Eigen::Vector3f> dstPoints;
        pangolin::GlBuffer lineEndPointBuffer;
    };

    struct CoordinateFrameRenderable : public Renderable {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        void applyRenderingProperties();
        void render();

        float scale;
        Eigen::Matrix4f transform;
    };

    struct TriangleMeshRenderable : public Renderable {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        void applyRenderingProperties();
        void render();

        std::vector<Eigen::Vector3f> vertices;
        std::vector<std::vector<size_t> > faces;
        std::vector<Eigen::Vector3f> vertexNormals;
        std::vector<Eigen::Vector3f> faceNormals;
        std::vector<Eigen::Vector3f> vertexColors;
        std::vector<Eigen::Vector3f> faceColors;
        std::vector<float> vertexValues;
        std::vector<float> faceValues;
        pangolin::GlBuffer vertexBuffer;
        pangolin::GlBuffer normalBuffer;
        pangolin::GlBuffer colorBuffer;
        pangolin::GlBuffer normalEndPointBuffer;
    };
}
