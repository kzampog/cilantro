#include <cilantro/image_point_cloud_conversions.hpp>

namespace cilantro {
    void depthImageToPoints(const pangolin::Image<unsigned short> &depth_img,
                            const Eigen::Matrix3f &intr,
                            VectorSet<float,3> &points,
                            bool keep_invalid)
    {
        if (!depth_img.ptr) return;

        points.resize(Eigen::NoChange, depth_img.w*depth_img.h);
        size_t k = 0;
        for (size_t y = 0; y < depth_img.h; y++) {
            for (size_t x = 0; x < depth_img.w; x++) {
                float d = depth_img(x,y)/1000.0f;
                if (keep_invalid || d > 0.0f) {
                    points.col(k) = Eigen::Vector3f((x - intr(0,2)) * d / intr(0,0), (y - intr(1,2)) * d / intr(1,1), d);
                    k++;
                }
            }
        }
        points.conservativeResize(Eigen::NoChange, k);
    }

    void RGBDImagesToPointsColors(const pangolin::Image<Eigen::Matrix<unsigned char,3,1>> &rgb_img,
                                  const pangolin::Image<unsigned short> &depth_img,
                                  const Eigen::Matrix3f &intr,
                                  VectorSet<float,3> &points,
                                  VectorSet<float,3> &colors,
                                  bool keep_invalid)
    {
        if (!depth_img.ptr || !rgb_img.ptr || depth_img.w != rgb_img.w || depth_img.h != rgb_img.h) return;

        points.resize(Eigen::NoChange, depth_img.w*depth_img.h);
        colors.resize(Eigen::NoChange, depth_img.w*depth_img.w);
        size_t k = 0;
        for (size_t y = 0; y < depth_img.h; y++) {
            for (size_t x = 0; x < depth_img.w; x++) {
                float d = depth_img(x,y)/1000.0f;
                if (keep_invalid || d > 0.0f) {
                    points.col(k) = Eigen::Vector3f((x - intr(0,2)) * d / intr(0,0), (y - intr(1,2)) * d / intr(1,1), d);
                    colors.col(k) = rgb_img(x,y).cast<float>()/255.0f;
                    k++;
                }
            }
        }
        points.conservativeResize(Eigen::NoChange, k);
        colors.conservativeResize(Eigen::NoChange, k);
    }

    void pointsToDepthImage(const VectorSet<float,3> &points,
                            const Eigen::Matrix3f &intr,
                            pangolin::Image<unsigned short> &depth_img)
    {
        if (!depth_img.ptr) return;

        depth_img.Memset(0);
        for (size_t i = 0; i < points.cols(); i++) {
            const Eigen::Vector3f& pt = points.col(i);
            size_t x = (size_t)std::llround(pt[0]*intr(0,0)/pt[2] + intr(0,2));
            size_t y = (size_t)std::llround(pt[1]*intr(1,1)/pt[2] + intr(1,2));
            if (x < depth_img.w && y < depth_img.h && pt[2] >= 0.0f && (depth_img(x,y) == 0 || 1000.0f*pt[2] < depth_img(x,y))) {
                depth_img(x,y) = (unsigned short)(pt[2]*1000.0f);
            }
        }
    }

    void pointsToDepthImage(const VectorSet<float,3> &points,
                            const Eigen::Matrix3f &intr,
                            const Eigen::Matrix3f &rot_mat,
                            const Eigen::Vector3f &t_vec,
                            pangolin::Image<unsigned short> &depth_img)
    {
        VectorSet<float,3> points_t = (rot_mat*points).colwise() + t_vec;
        pointsToDepthImage(points_t, intr, depth_img);
    }

    void pointsColorsToRGBDImages(const VectorSet<float,3> &points,
                                  const VectorSet<float,3> &colors,
                                  const Eigen::Matrix3f &intr,
                                  pangolin::Image<Eigen::Matrix<unsigned char,3,1>> &rgb_img,
                                  pangolin::Image<unsigned short> &depth_img)
    {
        if (!rgb_img.ptr || !depth_img.ptr || points.cols() != colors.cols() || rgb_img.w != depth_img.w || rgb_img.h != depth_img.h) return;

        rgb_img.Memset(0);
        depth_img.Memset(0);
        for (size_t i = 0; i < points.cols(); i++) {
            const Eigen::Vector3f& pt = points.col(i);
            size_t x = (size_t)std::llround(pt[0]*intr(0,0)/pt[2] + intr(0,2));
            size_t y = (size_t)std::llround(pt[1]*intr(1,1)/pt[2] + intr(1,2));
            if (x < depth_img.w && y < depth_img.h && pt[2] >= 0.0f && (depth_img(x,y) == 0 || 1000.0f*pt[2] < depth_img(x,y))) {
                depth_img(x,y) = (unsigned short)(pt[2]*1000.0f);
                rgb_img(x,y) = (255.0f*colors.col(i)).cast<unsigned char>();
            }
        }
    }

    void pointsColorsToRGBDImages(const VectorSet<float,3> &points,
                                  const VectorSet<float,3> &colors,
                                  const Eigen::Matrix3f &intr,
                                  const Eigen::Matrix3f &rot_mat,
                                  const Eigen::Vector3f &t_vec,
                                  pangolin::Image<Eigen::Matrix<unsigned char,3,1>> &rgb_img,
                                  pangolin::Image<unsigned short> &depth_img)
    {
        VectorSet<float,3> points_t = (rot_mat*points).colwise() + t_vec;
        pointsColorsToRGBDImages(points_t, colors, intr, rgb_img, depth_img);
    }

    void pointsToIndexMap(const VectorSet<float,3> &points,
                          const Eigen::Matrix3f &intr,
                          pangolin::Image<size_t> &index_map)
    {
        if (!index_map.ptr) return;

        size_t empty = std::numeric_limits<std::size_t>::max();
        index_map.Fill(empty);

        for (size_t i = 0; i < points.cols(); i++) {
            const Eigen::Vector3f& pt = points.col(i);
            size_t x = (size_t)std::llround(pt[0]*intr(0,0)/pt[2] + intr(0,2));
            size_t y = (size_t)std::llround(pt[1]*intr(1,1)/pt[2] + intr(1,2));
            if (x < index_map.w && y < index_map.h && pt[2] >= 0.0f && (index_map(x,y) == empty || pt[2] < points(2,index_map(x,y)))) {
                index_map(x,y) = i;
            }
        }
    }

    void pointsToIndexMap(const VectorSet<float,3> &points,
                          const Eigen::Matrix3f &intr,
                          const Eigen::Matrix3f &rot_mat,
                          const Eigen::Vector3f &t_vec,
                          pangolin::Image<size_t> &index_map)
    {
        VectorSet<float,3> points_t = (rot_mat*points).colwise() + t_vec;
        pointsToIndexMap(points_t, intr, index_map);
    }
}
