#pragma once

#include <pangolin/pangolin.h>
#include <pangolin/display/display_internal.h>

class ImageViewer {
public:
    ImageViewer(const std::string &window_name, const std::string &display_name);
    ~ImageViewer();

    ImageViewer& setImage(void * data, size_t w, size_t h, const std::string &fmt);
    ImageViewer& setImage(const pangolin::Image<unsigned char>& img, const std::string &fmt);
    ImageViewer& setImage(const pangolin::TypedImage& img);
    ImageViewer& setImage(const pangolin::GlTexture& texture);

    ImageViewer& clear();

    inline ImageViewer& clearRenderArea() {
        gl_context_->MakeCurrent();
        display_->Activate();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        return *this;
    }
    ImageViewer& render();
    inline ImageViewer& finishFrame() { gl_context_->MakeCurrent(); pangolin::FinishFrame(); return *this; }
    inline ImageViewer& spinOnce() { clearRenderArea(); render(); finishFrame(); return *this; }

    inline bool wasStopped() const { return gl_context_->quit; }

private:
    pangolin::PangolinGl *gl_context_;
    pangolin::View *display_;
    pangolin::GlTexture gl_texture_;
    pangolin::GlPixFormat gl_pix_format_;
};
