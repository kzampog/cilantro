#include <cilantro/image_point_cloud_conversions.hpp>

void depthImageToPoints(const pangolin::Image<unsigned short> &depth_img,
                        const Eigen::Matrix3f &intr,
                        std::vector<Eigen::Vector3f> &points,
                        bool keep_invalid)
{
    if (!depth_img.ptr) return;

    points.resize(depth_img.w*depth_img.h);
    size_t k = 0;
    for (size_t y = 0; y < depth_img.h; y++) {
        for (size_t x = 0; x < depth_img.w; x++) {
            float d = depth_img(x,y)/1000.0f;
            if (keep_invalid || d > 0.0f) {
                points[k] = Eigen::Vector3f((x - intr(0,2)) * d / intr(0,0), (y - intr(1,2)) * d / intr(1,1), d);
                k++;
            }
        }
    }
    points.resize(k);
}

void RGBDImagesToPointsColors(const pangolin::Image<Eigen::Matrix<unsigned char,3,1> > &rgb_img,
                              const pangolin::Image<unsigned short> &depth_img,
                              const Eigen::Matrix3f &intr,
                              std::vector<Eigen::Vector3f> &points,
                              std::vector<Eigen::Vector3f> &colors,
                              bool keep_invalid)
{
    if (!depth_img.ptr || !rgb_img.ptr || depth_img.w != rgb_img.w || depth_img.h != rgb_img.h) return;

    points.resize(depth_img.w*depth_img.h);
    colors.resize(depth_img.w*depth_img.w);
    size_t k = 0;
    for (size_t y = 0; y < depth_img.h; y++) {
        for (size_t x = 0; x < depth_img.w; x++) {
            float d = depth_img(x,y)/1000.0f;
            if (keep_invalid || d > 0.0f) {
                points[k] = Eigen::Vector3f((x - intr(0,2)) * d / intr(0,0), (y - intr(1,2)) * d / intr(1,1), d);
                colors[k] = rgb_img(x,y).cast<float>()/255.0f;
                k++;
            }
        }
    }
    points.resize(k);
    colors.resize(k);
}

void pointsToDepthImage(const std::vector<Eigen::Vector3f> &points,
                        const Eigen::Matrix3f &intr,
                        pangolin::Image<unsigned short> &depth_img)
{
    if (points.empty() || !depth_img.ptr) return;

    depth_img.Memset(0);
    for (size_t i = 0; i < points.size(); i++) {
        const Eigen::Vector3f& pt = points[i];
        size_t x = (size_t)std::llround(pt[0]*intr(0,0)/pt[2] + intr(0,2));
        size_t y = (size_t)std::llround(pt[1]*intr(1,1)/pt[2] + intr(1,2));
        if (x < depth_img.w && y < depth_img.h && pt[2] >= 0.0f && (depth_img(x,y) == 0 || 1000.0f*pt[2] < depth_img(x,y))) {
            depth_img(x,y) = (unsigned short)(pt[2]*1000.0f);
        }
    }
}

void pointsToDepthImage(const std::vector<Eigen::Vector3f> &points,
                        const Eigen::Matrix3f &intr,
                        const Eigen::Matrix3f &rot_mat,
                        const Eigen::Vector3f &t_vec,
                        pangolin::Image<unsigned short> &depth_img)
{
    if (points.empty() || !depth_img.ptr) return;
    std::vector<Eigen::Vector3f> points_t(points.size());
    Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)points_t.data(), 3, points_t.size()) = (rot_mat*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)points.data(), 3, points.size())).colwise() + t_vec;
    pointsToDepthImage(points_t, intr, depth_img);
}

void pointsColorsToRGBDImages(const std::vector<Eigen::Vector3f> &points,
                              const std::vector<Eigen::Vector3f> &colors,
                              const Eigen::Matrix3f &intr,
                              pangolin::Image<Eigen::Matrix<unsigned char,3,1> > &rgb_img,
                              pangolin::Image<unsigned short> &depth_img)
{
    if (!rgb_img.ptr || !depth_img.ptr || points.empty() || points.size() != colors.size() || rgb_img.w != depth_img.w || rgb_img.h != depth_img.h) return;

    rgb_img.Memset(0);
    depth_img.Memset(0);
    for (size_t i = 0; i < points.size(); i++) {
        const Eigen::Vector3f& pt = points[i];
        size_t x = (size_t)std::llround(pt[0]*intr(0,0)/pt[2] + intr(0,2));
        size_t y = (size_t)std::llround(pt[1]*intr(1,1)/pt[2] + intr(1,2));
        if (x < depth_img.w && y < depth_img.h && pt[2] >= 0.0f && (depth_img(x,y) == 0 || 1000.0f*pt[2] < depth_img(x,y))) {
            depth_img(x,y) = (unsigned short)(pt[2]*1000.0f);
            rgb_img(x,y) = (255.0f*colors[i]).cast<unsigned char>();
        }
    }
}

void pointsColorsToRGBDImages(const std::vector<Eigen::Vector3f> &points,
                              const std::vector<Eigen::Vector3f> &colors,
                              const Eigen::Matrix3f &intr,
                              const Eigen::Matrix3f &rot_mat,
                              const Eigen::Vector3f &t_vec,
                              pangolin::Image<Eigen::Matrix<unsigned char,3,1> > &rgb_img,
                              pangolin::Image<unsigned short> &depth_img)
{
    if (!rgb_img.ptr || !depth_img.ptr || points.empty() || points.size() != colors.size() || rgb_img.w != depth_img.w || rgb_img.h != depth_img.h) return;
    std::vector<Eigen::Vector3f> points_t(points.size());
    Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)points_t.data(), 3, points_t.size()) = (rot_mat*Eigen::Map<Eigen::Matrix<float,3,Eigen::Dynamic> >((float *)points.data(), 3, points.size())).colwise() + t_vec;
    pointsColorsToRGBDImages(points_t, colors, intr, rgb_img, depth_img);
}
