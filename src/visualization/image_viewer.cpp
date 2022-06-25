#include <cilantro/visualization/image_viewer.hpp>

#ifdef HAVE_PANGOLIN
namespace cilantro {

ImageViewer& ImageViewer::setImage(void* data, size_t w, size_t h, const std::string& fmt) {
  pangolin::BindToContext(window_name_);
  gl_pix_format_ = pangolin::GlPixFormat(pangolin::PixelFormatFromString(fmt));

  // Initialise if it didn't already exist or the size was too small
  if (!gl_texture_.tid || gl_texture_.width != (int)w || gl_texture_.height != (int)h ||
      gl_texture_.internal_format != gl_pix_format_.scalable_internal_format) {
    display_->SetAspect((float)w / (float)h);
    gl_texture_.Reinitialise(w, h, gl_pix_format_.scalable_internal_format, true, 0,
                             gl_pix_format_.glformat, gl_pix_format_.gltype, data);
  } else {
    gl_texture_.Upload(data, gl_pix_format_.glformat, gl_pix_format_.gltype);
  }

  return *this;
}

ImageViewer& ImageViewer::setImage(const pangolin::Image<unsigned char>& img,
                                   const std::string& fmt) {
  return setImage(img.ptr, img.w, img.h, fmt);
}

ImageViewer& ImageViewer::setImage(const pangolin::TypedImage& img) {
  return setImage(img.ptr, img.w, img.h, img.fmt.format);
}

ImageViewer& ImageViewer::setImage(const pangolin::GlTexture& texture) {
  pangolin::BindToContext(window_name_);
  // Initialise if it didn't already exist or the size was too small
  if (!gl_texture_.tid || gl_texture_.width != texture.width ||
      gl_texture_.height != texture.height ||
      gl_texture_.internal_format != texture.internal_format) {
    display_->SetAspect((float)texture.width / (float)texture.height);
    gl_texture_.Reinitialise(texture.width, texture.height, texture.internal_format, true);
  }
  glCopyImageSubData(texture.tid, GL_TEXTURE_2D, 0, 0, 0, 0, gl_texture_.tid, GL_TEXTURE_2D, 0, 0,
                     0, 0, gl_texture_.width, gl_texture_.height, 1);

  return *this;
}

ImageViewer& ImageViewer::clear() {
  pangolin::BindToContext(window_name_);
  gl_texture_.Delete();
  return *this;
}

ImageViewer& ImageViewer::clearRenderArea() {
  pangolin::BindToContext(window_name_);
  display_->Activate();
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  return *this;
}

ImageViewer& ImageViewer::render() {
  pangolin::BindToContext(window_name_);
  display_->Activate();
  glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
  gl_texture_.RenderToViewportFlipY();
  return *this;
}

void ImageViewer::init_(const std::string& window_name, const std::string& display_name) {
  window_name_ = window_name;
  pangolin::BindToContext(window_name_);
  pangolin::RegisterKeyPressCallback('q', pangolin::Quit);
  pangolin::RegisterKeyPressCallback('Q', pangolin::Quit);
  display_ = &(pangolin::Display(display_name));
}

}  // namespace cilantro
#endif
