#pragma once

#include <cilantro/config.hpp>

#ifdef HAVE_PANGOLIN
#include <pangolin/pangolin.h>

namespace cilantro {

class ImageViewer {
public:
  inline ImageViewer(const std::string& window_name, const std::string& display_name) {
    init_(window_name, display_name);
  }
  inline ~ImageViewer() { pangolin::BindToContext(window_name_); }

  ImageViewer& setImage(void* data, size_t w, size_t h, const std::string& fmt);
  ImageViewer& setImage(const pangolin::Image<unsigned char>& img, const std::string& fmt);
  ImageViewer& setImage(const pangolin::TypedImage& img);
  ImageViewer& setImage(const pangolin::GlTexture& texture);

  ImageViewer& clear();

  ImageViewer& clearRenderArea();
  ImageViewer& render();
  inline ImageViewer& finishFrame() {
    pangolin::BindToContext(window_name_);
    pangolin::FinishFrame();
    return *this;
  }
  inline ImageViewer& spinOnce() {
    clearRenderArea();
    render();
    finishFrame();
    return *this;
  }

  inline bool wasStopped() {
    pangolin::BindToContext(window_name_);
    return pangolin::ShouldQuit();
  }

  inline const std::string& getWindowName() const { return window_name_; }
  inline pangolin::View* getDisplay() const { return display_; }

private:
  std::string window_name_;
  pangolin::View* display_;
  pangolin::GlTexture gl_texture_;
  pangolin::GlPixFormat gl_pix_format_;

  void init_(const std::string& window_name, const std::string& display_name);
};

}  // namespace cilantro
#endif
