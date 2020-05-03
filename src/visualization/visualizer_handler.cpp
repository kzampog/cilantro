#include <cilantro/visualization/visualizer_handler.hpp>
#include <cilantro/visualization/visualizer.hpp>

#ifdef HAVE_PANGOLIN
namespace cilantro {
    VisualizerHandler::VisualizerHandler(Visualizer *visualizer)
            : translationFactor(0.01f),
              zoomFraction(PANGO_DFLT_HANDLER3D_ZF),
              pointSizeStep(1.0f),
              minPointSize(1.0f),
              lineWidthStep(1.0f),
              minLineWidth(1.0f),
              visualizer(visualizer),
              ortho(false),
              ortho_left(-(float)std::abs(visualizer->display_->aspect)),
              ortho_right((float)std::abs(visualizer->display_->aspect)),
              ortho_bottom(-1.0f),
              ortho_top(1.0f),
              ortho_near(-100.0f),
              ortho_far(100.0f),
              perspective_projection(visualizer->gl_render_state_->GetProjectionMatrix()),
              default_model_view(visualizer->gl_render_state_->GetModelViewMatrix()),
              // cam_state(visualizer->gl_render_state_.get()),
              enforce_up(pangolin::AxisNone),
              cameraspec(pangolin::CameraSpecOpenGl),
              last_z(0.8)
    {
        pangolin::SetZero<3,1>(rot_center);
    }

    void VisualizerHandler::Keyboard(pangolin::View&, unsigned char key, int x, int y, bool pressed)
    {
        if (!pressed) return;

        switch (key) {
            case 'r':
            case 'R':
                visualizer->gl_render_state_->SetModelViewMatrix(default_model_view);
                break;
            case '+':
                visualizer->gl_context_->MakeCurrent();
                for (auto it = visualizer->renderables_.begin(); it != visualizer->renderables_.end(); ++it) {
                    RenderingProperties rp = it->second.first->getRenderingProperties();
                    rp.pointSize += pointSizeStep;
                    rp.lineWidth += lineWidthStep;
                    it->second.first->setRenderingProperties(rp);
                }
                break;
            case '-':
                visualizer->gl_context_->MakeCurrent();
                for (auto it = visualizer->renderables_.begin(); it != visualizer->renderables_.end(); ++it) {
                    RenderingProperties rp = it->second.first->getRenderingProperties();
                    rp.pointSize = std::max(rp.pointSize - pointSizeStep, minPointSize);
                    rp.lineWidth = std::max(rp.lineWidth - lineWidthStep, minLineWidth);
                    it->second.first->setRenderingProperties(rp);
                }
                break;
            case 'n':
            case 'N':
                visualizer->gl_context_->MakeCurrent();
                for (auto it = visualizer->renderables_.begin(); it != visualizer->renderables_.end(); ++it) {
                    RenderingProperties rp = it->second.first->getRenderingProperties();
                    rp.drawNormals = !rp.drawNormals;
                    it->second.first->setRenderingProperties(rp);
                }
                break;
            case 'w':
            case 'W':
                visualizer->gl_context_->MakeCurrent();
                for (auto it = visualizer->renderables_.begin(); it != visualizer->renderables_.end(); ++it) {
                    RenderingProperties rp = it->second.first->getRenderingProperties();
                    rp.drawWireframe = !rp.drawWireframe;
                    it->second.first->setRenderingProperties(rp);
                }
                break;
            case 'p':
            case 'P':
                if (ortho) {
                    ortho = false;
                    visualizer->gl_render_state_->SetProjectionMatrix(perspective_projection);
                } else {
                    ortho = true;
                    visualizer->gl_render_state_->SetProjectionMatrix(pangolin::ProjectionMatrixOrthographic(ortho_left, ortho_right, ortho_bottom, ortho_top, ortho_near, ortho_far));
                }
                break;
            case 'l':
            case 'L':
                visualizer->gl_context_->MakeCurrent();
                for (auto it = visualizer->renderables_.begin(); it != visualizer->renderables_.end(); ++it) {
                    RenderingProperties rp = it->second.first->getRenderingProperties();
                    rp.useLighting = !rp.useLighting;
                    it->second.first->setRenderingProperties(rp);
                }
                break;
            case 'q':
            case 'Q':
                pangolin::Quit();
                break;
            default:
                break;
        }

        auto it = key_callback_map.find(key);
        if (it != key_callback_map.end()) {
            it->second();
        }
    }

    bool VisualizerHandler::ValidWinDepth(pangolin::GLprecision depth)
    {
        return depth != 1;
    }

    void VisualizerHandler::PixelUnproject(pangolin::View& view, pangolin::GLprecision winx, pangolin::GLprecision winy, pangolin::GLprecision winz, pangolin::GLprecision Pc[3])
    {
        const GLint viewport[4] = {view.v.l,view.v.b,view.v.w,view.v.h};
        const pangolin::OpenGlMatrix proj = visualizer->gl_render_state_->GetProjectionMatrix();
        pangolin::glUnProject(winx, winy, winz, pangolin::Identity4d, proj.m, viewport, &Pc[0], &Pc[1], &Pc[2]);
    }

    void VisualizerHandler::GetPosNormal(pangolin::View& view, int winx, int winy, pangolin::GLprecision p[3], pangolin::GLprecision Pw[3], pangolin::GLprecision Pc[3], pangolin::GLprecision nw[3], pangolin::GLprecision default_z)
    {
        // TODO: Get to work on android

        const int zl = (hwin*2+1);
        const int zsize = zl*zl;
        GLfloat zs[zsize];

#ifndef HAVE_GLES
        glReadBuffer(GL_FRONT);
        glReadPixels(winx-hwin,winy-hwin,zl,zl,GL_DEPTH_COMPONENT,GL_FLOAT,zs);
#else
        std::fill(zs,zs+zsize, 1);
#endif
        GLfloat mindepth = *(std::min_element(zs,zs+zsize));

        if(mindepth == 1) mindepth = (GLfloat)default_z;

        p[0] = winx; p[1] = winy; p[2] = mindepth;
        PixelUnproject(view, winx, winy, mindepth, Pc);

        const pangolin::OpenGlMatrix mv = visualizer->gl_render_state_->GetModelViewMatrix();

        pangolin::GLprecision T_wc[3*4];
        pangolin::LieSE3from4x4(T_wc, mv.Inverse().m );
        pangolin::LieApplySE3vec(Pw, T_wc, Pc);

        // Neighboring points in camera coordinates
        pangolin::GLprecision Pl[3]; pangolin::GLprecision Pr[3]; pangolin::GLprecision Pb[3]; pangolin::GLprecision Pt[3];
        PixelUnproject(view, winx-hwin, winy, zs[hwin*zl + 0],    Pl );
        PixelUnproject(view, winx+hwin, winy, zs[hwin*zl + zl-1], Pr );
        PixelUnproject(view, winx, winy-hwin, zs[hwin+1],         Pb );
        PixelUnproject(view, winx, winy+hwin, zs[zsize-(hwin+1)], Pt );

        // n = ((Pr-Pl).cross(Pt-Pb)).normalized();
        pangolin::GLprecision PrmPl[3]; pangolin::GLprecision PtmPb[3];
        pangolin::MatSub<3,1>(PrmPl,Pr,Pl);
        pangolin::MatSub<3,1>(PtmPb,Pt,Pb);

        pangolin::GLprecision nc[3];
        pangolin::CrossProduct(nc, PrmPl, PtmPb);
        pangolin::Normalise<3>(nc);

        // T_wc is col major, so the rotation component is first.
        pangolin::LieApplySO3(nw,T_wc,nc);
    }

    void VisualizerHandler::Mouse(pangolin::View& display, pangolin::MouseButton button, int x, int y, bool pressed, int button_state)
    {
        // mouse down
        last_pos[0] = (float)x;
        last_pos[1] = (float)y;

        if (!pressed) return;

        pangolin::GLprecision T_nc[3*4];
        pangolin::LieSetIdentity(T_nc);

        GetPosNormal(display,x,y,p,Pw,Pc,n,last_z);
        if( ValidWinDepth(p[2]) )
        {
            last_z = p[2];
            std::copy(Pc,Pc+3,rot_center);
        }

        if(button == pangolin::MouseWheelUp || button == pangolin::MouseWheelDown) {
            if (ortho) {
                ortho_left *= 1.0f + ((button == pangolin::MouseWheelUp) ? -1.0f : 1.0f) * zoomFraction;
                ortho_right *= 1.0f + ((button == pangolin::MouseWheelUp) ? -1.0f : 1.0f) * zoomFraction;
                ortho_bottom *= 1.0f + ((button == pangolin::MouseWheelUp) ? -1.0f : 1.0f) * zoomFraction;
                ortho_top *= 1.0f + ((button == pangolin::MouseWheelUp) ? -1.0f : 1.0f) * zoomFraction;
                visualizer->gl_render_state_->SetProjectionMatrix(pangolin::ProjectionMatrixOrthographic(ortho_left,ortho_right,ortho_bottom,ortho_top,ortho_near,ortho_far));
            } else {
                pangolin::LieSetIdentity(T_nc);
                const pangolin::GLprecision t[3] = { 0,0,(button==pangolin::MouseWheelUp?1:-1)*100*translationFactor};
                pangolin::LieSetTranslation<>(T_nc,t);
                if( !(button_state & pangolin::MouseButtonRight) && !(rot_center[0]==0 && rot_center[1]==0 && rot_center[2]==0) )
                {
                    pangolin::LieSetTranslation<>(T_nc,rot_center);
                    const pangolin::GLprecision s = (button==pangolin::MouseWheelUp?-1.0:1.0) * zoomFraction;
                    pangolin::MatMul<3,1>(T_nc+(3*3), s);
                }
                pangolin::OpenGlMatrix& spec = visualizer->gl_render_state_->GetModelViewMatrix();
                pangolin::LieMul4x4bySE3<>(spec.m,T_nc,spec.m);
            }
        }

    }

    void VisualizerHandler::MouseMotion(pangolin::View& display, int x, int y, int button_state)
    {
        const pangolin::GLprecision rf = 0.01;
        const float delta[2] = { (float)x - last_pos[0], (float)y - last_pos[1] };
        const float mag = delta[0]*delta[0] + delta[1]*delta[1];

        // TODO: convert delta to degrees based of fov
        // TODO: make transformation with respect to cam spec

        if( mag < 50.0f*50.0f )
        {
            pangolin::OpenGlMatrix& mv = visualizer->gl_render_state_->GetModelViewMatrix();
            const pangolin::GLprecision* up = pangolin::AxisDirectionVector[enforce_up];
            pangolin::GLprecision T_nc[3*4];
            pangolin::LieSetIdentity(T_nc);
            bool rotation_changed = false;

            if( button_state == pangolin::MouseButtonMiddle )
            {
                // Middle Drag: Rotate around view

                // Try to correct for different coordinate conventions.
                pangolin::GLprecision aboutx = -rf * delta[1];
                pangolin::GLprecision abouty = rf * delta[0];
                pangolin::OpenGlMatrix& pm = visualizer->gl_render_state_->GetProjectionMatrix();
                abouty *= -pm.m[2 * 4 + 3];

                pangolin::Rotation<>(T_nc, aboutx, abouty, (pangolin::GLprecision)0.0);
            }else if( button_state == pangolin::MouseButtonLeft )
            {
                // Left Drag: in plane translate
                if( ValidWinDepth(last_z) )
                {
                    pangolin::GLprecision np[3];
                    PixelUnproject(display, x, y, last_z, np);
                    const pangolin::GLprecision t[] = { np[0] - rot_center[0], np[1] - rot_center[1], 0};
                    pangolin::LieSetTranslation<>(T_nc,t);
                    std::copy(np,np+3,rot_center);
                }else{
                    const pangolin::GLprecision t[] = { -10*delta[0]*translationFactor, 10*delta[1]*translationFactor, 0};
                    pangolin::LieSetTranslation<>(T_nc,t);
                }
            }else if( button_state == (pangolin::MouseButtonLeft | pangolin::MouseButtonRight) )
            {
                // Left and Right Drag: in plane rotate about object
                //        Rotation<>(T_nc,0.0,0.0, delta[0]*0.01);

                pangolin::GLprecision T_2c[3*4];
                pangolin::Rotation<>(T_2c, (pangolin::GLprecision)0.0, (pangolin::GLprecision)0.0, delta[0]*rf);
                pangolin::GLprecision mrotc[3];
                pangolin::MatMul<3,1>(mrotc, rot_center, (pangolin::GLprecision)-1.0);
                pangolin::LieApplySO3<>(T_2c+(3*3),T_2c,mrotc);
                pangolin::GLprecision T_n2[3*4];
                pangolin::LieSetIdentity<>(T_n2);
                pangolin::LieSetTranslation<>(T_n2,rot_center);
                pangolin::LieMulSE3(T_nc, T_n2, T_2c );
                rotation_changed = true;
            }else if( button_state == pangolin::MouseButtonRight)
            {
                pangolin::GLprecision aboutx = -rf * delta[1];
                pangolin::GLprecision abouty = -rf * delta[0];

                // Try to correct for different coordinate conventions.
                if(visualizer->gl_render_state_->GetProjectionMatrix().m[2*4+3] <= 0) {
                    abouty *= -1;
                }

                if(enforce_up) {
                    // Special case if view direction is parallel to up vector
                    const pangolin::GLprecision updotz = mv.m[2]*up[0] + mv.m[6]*up[1] + mv.m[10]*up[2];
                    if(updotz > 0.98) aboutx = std::min(aboutx, (pangolin::GLprecision)0.0);
                    if(updotz <-0.98) aboutx = std::max(aboutx, (pangolin::GLprecision)0.0);
                    // Module rotation around y so we don't spin too fast!
                    abouty *= (1-0.6*fabs(updotz));
                }

                // Right Drag: object centric rotation
                pangolin::GLprecision T_2c[3*4];
                pangolin::Rotation<>(T_2c, aboutx, abouty, (pangolin::GLprecision)0.0);
                pangolin::GLprecision mrotc[3];
                pangolin::MatMul<3,1>(mrotc, rot_center, (pangolin::GLprecision)-1.0);
                pangolin::LieApplySO3<>(T_2c+(3*3),T_2c,mrotc);
                pangolin::GLprecision T_n2[3*4];
                pangolin::LieSetIdentity<>(T_n2);
                pangolin::LieSetTranslation<>(T_n2,rot_center);
                pangolin::LieMulSE3(T_nc, T_n2, T_2c );
                rotation_changed = true;
            }

            pangolin::LieMul4x4bySE3<>(mv.m,T_nc,mv.m);

            if(enforce_up != pangolin::AxisNone && rotation_changed) {
                pangolin::EnforceUpT_cw(mv.m, up);
            }
        }

        last_pos[0] = (float)x;
        last_pos[1] = (float)y;
    }

    void VisualizerHandler::Special(pangolin::View& display, pangolin::InputSpecial inType, float x, float y, float p1, float p2, float /*p3*/, float /*p4*/, int button_state)
    {
        if( !(inType == pangolin::InputSpecialScroll || inType == pangolin::InputSpecialRotate) )
            return;

        // mouse down
        last_pos[0] = x;
        last_pos[1] = y;

        pangolin::GLprecision T_nc[3*4];
        pangolin::LieSetIdentity(T_nc);

        GetPosNormal(display, (int)x, (int)y, p, Pw, Pc, n, last_z);
        if(p[2] < 1.0) {
            last_z = p[2];
            std::copy(Pc,Pc+3,rot_center);
        }

        if( inType == pangolin::InputSpecialScroll ) {
            if(button_state & pangolin::KeyModifierCmd) {
                const pangolin::GLprecision rx = -p2 / 1000;
                const pangolin::GLprecision ry = -p1 / 1000;

                pangolin::Rotation<>(T_nc,rx, ry, (pangolin::GLprecision)0.0);
                pangolin::OpenGlMatrix& spec = visualizer->gl_render_state_->GetModelViewMatrix();
                pangolin::LieMul4x4bySE3<>(spec.m,T_nc,spec.m);
            }else{
                const pangolin::GLprecision scrolly = p2/10;

                pangolin::LieSetIdentity(T_nc);
                const pangolin::GLprecision t[] = { 0,0, -scrolly*100*translationFactor};
                pangolin::LieSetTranslation<>(T_nc,t);
                if( !(button_state & pangolin::MouseButtonRight) && !(rot_center[0]==0 && rot_center[1]==0 && rot_center[2]==0) )
                {
                    pangolin::LieSetTranslation<>(T_nc,rot_center);
                    pangolin::MatMul<3,1>(T_nc+(3*3), -scrolly * zoomFraction);
                }
                pangolin::OpenGlMatrix& spec = visualizer->gl_render_state_->GetModelViewMatrix();
                pangolin::LieMul4x4bySE3<>(spec.m,T_nc,spec.m);
            }
        }else if(inType == pangolin::InputSpecialRotate) {
            const pangolin::GLprecision r = p1 / 20;

            pangolin::GLprecision T_2c[3*4];
            pangolin::Rotation<>(T_2c, (pangolin::GLprecision)0.0, (pangolin::GLprecision)0.0, r);
            pangolin::GLprecision mrotc[3];
            pangolin::MatMul<3,1>(mrotc, rot_center, (pangolin::GLprecision)-1.0);
            pangolin::LieApplySO3<>(T_2c+(3*3),T_2c,mrotc);
            pangolin::GLprecision T_n2[3*4];
            pangolin::LieSetIdentity<>(T_n2);
            pangolin::LieSetTranslation<>(T_n2,rot_center);
            pangolin::LieMulSE3(T_nc, T_n2, T_2c );
            pangolin::OpenGlMatrix& spec = visualizer->gl_render_state_->GetModelViewMatrix();
            pangolin::LieMul4x4bySE3<>(spec.m,T_nc,spec.m);
        }
    }

    void VisualizerHandler::SetPerspectiveProjectionMatrix(const pangolin::OpenGlMatrix &mat) {
        perspective_projection = mat;
        if (!ortho) {
            visualizer->gl_render_state_->SetProjectionMatrix(perspective_projection);
        }
    }

    void VisualizerHandler::SetOrthographicProjectionMatrix(pangolin::GLprecision left, pangolin::GLprecision right, pangolin::GLprecision bottom, pangolin::GLprecision top, pangolin::GLprecision near, pangolin::GLprecision far) {
        ortho_left = (float)left;
        ortho_right = (float)right;
        ortho_bottom = (float)bottom;
        ortho_top = (float)top;
        ortho_near = (float)near;
        ortho_far = (float)far;
        if (ortho) {
            visualizer->gl_render_state_->SetProjectionMatrix(pangolin::ProjectionMatrixOrthographic(left, right, bottom, top, near, far));
        }
    }

    void VisualizerHandler::EnablePerspectiveProjection() {
        if (ortho) {
            ortho = false;
            visualizer->gl_render_state_->SetProjectionMatrix(perspective_projection);
        }
    }

    void VisualizerHandler::EnableOrthographicProjection() {
        if (!ortho) {
            ortho = true;
            visualizer->gl_render_state_->SetProjectionMatrix(pangolin::ProjectionMatrixOrthographic(ortho_left,ortho_right,ortho_bottom,ortho_top,ortho_near,ortho_far));
        }
    }

    void VisualizerHandler::ToggleProjectionMode() {
        if (ortho) {
            ortho = false;
            visualizer->gl_render_state_->SetProjectionMatrix(perspective_projection);
        } else {
            ortho = true;
            visualizer->gl_render_state_->SetProjectionMatrix(pangolin::ProjectionMatrixOrthographic(ortho_left,ortho_right,ortho_bottom,ortho_top,ortho_near,ortho_far));
        }
    }
}
#endif
