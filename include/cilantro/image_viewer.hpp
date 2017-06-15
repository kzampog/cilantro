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

    inline void clearRenderArea() const {
        gl_context_->MakeCurrent();
        display_->Activate();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    void render() const;
    inline void finishFrame() const { gl_context_->MakeCurrent(); pangolin::FinishFrame(); }
    inline void spinOnce() const { clearRenderArea(); render(); finishFrame(); }

    inline bool wasStopped() const { return gl_context_->quit; }

private:
    pangolin::PangolinGl *gl_context_;
    pangolin::View *display_;
    pangolin::GlTexture gl_texture_;
    pangolin::GlPixFormat gl_pix_format_;
};
