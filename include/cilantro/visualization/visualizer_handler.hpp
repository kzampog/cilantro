#pragma once

#include <cilantro/config.hpp>

#ifdef HAVE_PANGOLIN
#include <pangolin/pangolin.h>

namespace cilantro {
    class Visualizer;

    struct VisualizerHandler : pangolin::Handler
    {
        friend class Visualizer;

        VisualizerHandler(Visualizer *visualizer);

        virtual bool ValidWinDepth(pangolin::GLprecision depth);

        virtual void PixelUnproject(pangolin::View& view,
                                    pangolin::GLprecision winx, pangolin::GLprecision winy, pangolin::GLprecision winz,
                                    pangolin::GLprecision Pc[3]);

        virtual void GetPosNormal(pangolin::View& view,
                                  int x, int y,
                                  pangolin::GLprecision p[3],
                                  pangolin::GLprecision Pw[3],
                                  pangolin::GLprecision Pc[3],
                                  pangolin::GLprecision nw[3],
                                  pangolin::GLprecision default_z = 1.0);

        void Keyboard(pangolin::View&, unsigned char key, int x, int y, bool pressed);

        void Mouse(pangolin::View&, pangolin::MouseButton button, int x, int y, bool pressed, int button_state);

        void MouseMotion(pangolin::View&, int x, int y, int button_state);

        void Special(pangolin::View&, pangolin::InputSpecial inType,
                     float x, float y,
                     float p1, float p2, float p3, float p4,
                     int button_state);


        void SetPerspectiveProjectionMatrix(const pangolin::OpenGlMatrix &mat);

        void SetOrthographicProjectionMatrix(pangolin::GLprecision left, pangolin::GLprecision right,
                                             pangolin::GLprecision bottom, pangolin::GLprecision top,
                                             pangolin::GLprecision near, pangolin::GLprecision far);

        void EnablePerspectiveProjection();

        void EnableOrthographicProjection();

        void ToggleProjectionMode();

#ifdef USE_EIGEN
        // Return selected point in world coordinates
        inline Eigen::Vector3d Selected_P_w() const {
            return Eigen::Map<const Eigen::Matrix<pangolin::GLprecision,3,1>>(Pw).cast<double>();
        }
#endif

        float translationFactor;    // translation factor
        float zoomFraction;         // zoom fraction
        float pointSizeStep;
        float minPointSize;
        float lineWidthStep;
        float minLineWidth;

    protected:
        Visualizer * visualizer;
        bool ortho;
        float ortho_left;
        float ortho_right;
        float ortho_bottom;
        float ortho_top;
        float ortho_near;
        float ortho_far;
        pangolin::OpenGlMatrix perspective_projection;
        pangolin::OpenGlMatrix default_model_view;
        std::map<unsigned char, std::function<void(void)> > key_callback_map;

        // pangolin::OpenGlRenderState* cam_state;
        const static int hwin = 8;
        pangolin::AxisDirection enforce_up;
        pangolin::CameraSpec cameraspec;
        pangolin::GLprecision last_z;
        float last_pos[2];
        pangolin::GLprecision rot_center[3];

        pangolin::GLprecision p[3];
        pangolin::GLprecision Pw[3];
        pangolin::GLprecision Pc[3];
        pangolin::GLprecision n[3];
    };
}
#endif
