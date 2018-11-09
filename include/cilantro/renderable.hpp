#pragma once

#include <cilantro/colormap.hpp>

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
                                       lineDensityFraction(1.0f),
                                       drawWireframe(false),
                                       useFaceNormals(true),
                                       useFaceColors(false),
                                       useScalarValueMappedColors(true),
                                       minScalarValue(std::numeric_limits<float>::quiet_NaN()),
                                       maxScalarValue(std::numeric_limits<float>::quiet_NaN()),
                                       colormapType(ColormapType::JET),
                                       fontSize(15.0f),
                                       textAnchorPoint(0.5f,0.5f)
        {}

        virtual inline ~RenderingProperties() {}

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
        bool useScalarValueMappedColors;
        float minScalarValue;
        float maxScalarValue;
        ColormapType colormapType;
        float fontSize;
        Eigen::Vector2f textAnchorPoint;

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
        inline RenderingProperties& setUseScalarValueMappedColors(bool cm) { useScalarValueMappedColors = cm; return *this; }
        inline RenderingProperties& setScalarValuesRange(float min, float max) { minScalarValue = min; maxScalarValue = max; return *this; }
        inline RenderingProperties& setColormapType(const ColormapType ct) { colormapType = ct; return *this; }
        inline RenderingProperties& setFontSize(float fs) { fontSize = fs; return *this; }
        inline RenderingProperties& setTextAnchorPoint(const Eigen::Vector2f &ap) { textAnchorPoint = ap; return *this; }
        inline RenderingProperties& setTextAnchorPoint(float x_fraction, float y_fraction) { textAnchorPoint = Eigen::Vector2f(x_fraction,y_fraction); return *this; }

        static inline void setDefaultColor(const Eigen::Vector3f &color) { defaultColor = color; }
        static inline void setDefaultColor(float r, float g, float b) { defaultColor = Eigen::Vector3f(r,g,b); }
    };

    struct GPUBufferObjects {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        inline GPUBufferObjects() {}
        virtual inline ~GPUBufferObjects() {}
    };

    class Renderable {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        inline Renderable()
                : centroid(Eigen::Vector3f::Zero()), visible(true), drawLast(false), buffersUpToDate(false)
        {}

        inline Renderable(const RenderingProperties &rp,
                          const Eigen::Ref<const Eigen::Vector3f> &centroid = Eigen::Vector3f::Zero())
                : centroid(centroid), renderingProperties(rp), visible(true), drawLast(false), buffersUpToDate(false)
        {}

        virtual inline ~Renderable() {}

        virtual void updateGPUBuffers(GPUBufferObjects &gl_objects) = 0;

        virtual void render(GPUBufferObjects &gl_objects) = 0;

        inline void updateGPUBuffersAndRender(GPUBufferObjects &gl_objects) {
            if (!buffersUpToDate) {
                updateGPUBuffers(gl_objects);
                buffersUpToDate = true;
            }
            render(gl_objects);
        }

        inline const Eigen::Vector3f& getCentroid() const { return centroid; }

        inline const RenderingProperties& getRenderingProperties() const { return renderingProperties; }

        inline void setRenderingProperties(const RenderingProperties &rp) {
            renderingProperties = rp;
            buffersUpToDate = false;
        }

        inline bool getVisibility() const { return visible; }

        inline bool& getVisibility() { return visible; }

        inline void setVisibility(bool v) { visible = v; }

        inline void toggleVisibility() { visible = !visible; }

        inline bool getDrawLast() const { return drawLast; }

        inline bool& getDrawLast() { return drawLast; }

        inline void setDrawLast(bool dl) { drawLast = dl; }

    protected:
        Eigen::Vector3f centroid;   // For render priority
        RenderingProperties renderingProperties;
        bool visible;
        bool drawLast;
        bool buffersUpToDate;
    };
}
