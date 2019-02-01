#include <cilantro/image_viewer.hpp>

#ifdef HAVE_PANGOLIN
namespace cilantro {
    ImageViewer::ImageViewer() {
        init_("Window", "Display");
    }

    ImageViewer::ImageViewer(const std::string &window_name, const std::string &display_name) {
        init_(window_name, display_name);
    }

    ImageViewer::~ImageViewer() {}

    ImageViewer& ImageViewer::setImage(void * data, size_t w, size_t h, const std::string &fmt) {
        gl_context_->MakeCurrent();
        gl_pix_format_ = pangolin::GlPixFormat(pangolin::PixelFormatFromString(fmt));

        // Initialise if it didn't already exist or the size was too small
        if(!gl_texture_.tid || gl_texture_.width != (int)w || gl_texture_.height != (int)h ||
           gl_texture_.internal_format != gl_pix_format_.scalable_internal_format)
        {
            display_->SetAspect((float)w / (float)h);
            gl_texture_.Reinitialise(w, h, gl_pix_format_.scalable_internal_format, true, 0, gl_pix_format_.glformat, gl_pix_format_.gltype, data);
        } else {
            gl_texture_.Upload(data, gl_pix_format_.glformat, gl_pix_format_.gltype);
        }

        return *this;
    }

    ImageViewer& ImageViewer::setImage(const pangolin::Image<unsigned char>& img, const std::string &fmt) {
        return setImage(img.ptr, img.w, img.h, fmt);
    }

    ImageViewer& ImageViewer::setImage(const pangolin::TypedImage& img) {
        return setImage(img.ptr, img.w, img.h, img.fmt.format);
    }

    ImageViewer& ImageViewer::setImage(const pangolin::GlTexture& texture) {
        gl_context_->MakeCurrent();
        // Initialise if it didn't already exist or the size was too small
        if(!gl_texture_.tid || gl_texture_.width != texture.width || gl_texture_.height != texture.height ||
           gl_texture_.internal_format != texture.internal_format)
        {
            display_->SetAspect((float)texture.width / (float)texture.height);
            gl_texture_.Reinitialise(texture.width, texture.height, texture.internal_format, true);
        }
        glCopyImageSubData(texture.tid, GL_TEXTURE_2D, 0, 0, 0, 0, gl_texture_.tid, GL_TEXTURE_2D, 0, 0, 0, 0, gl_texture_.width, gl_texture_.height, 1);

        return *this;
    }

    ImageViewer& ImageViewer::clear() {
        gl_context_->MakeCurrent();
        gl_texture_.Delete();
        return *this;
    }

    ImageViewer& ImageViewer::clearRenderArea() {
        gl_context_->MakeCurrent();
        display_->Activate();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        return *this;
    }

    ImageViewer& ImageViewer::render() {
        gl_context_->MakeCurrent();
        display_->Activate();
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        gl_texture_.RenderToViewportFlipY();
        return *this;
    }

    void ImageViewer::init_(const std::string &window_name, const std::string &display_name) {
        gl_context_ = pangolin::FindContext(window_name);
        if (!gl_context_) {
            pangolin::CreateWindowAndBind(window_name);
            gl_context_ = pangolin::FindContext(window_name);
        }
        gl_context_->MakeCurrent();
        pangolin::RegisterKeyPressCallback('q', pangolin::Quit);
        pangolin::RegisterKeyPressCallback('Q', pangolin::Quit);
        display_ = &(pangolin::Display(display_name));
    }
}
#endif
