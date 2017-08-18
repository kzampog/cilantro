#include <cilantro/space_region.hpp>
#include <cilantro/image_viewer.hpp>

int main(int argc, char ** argv) {

    std::vector<Eigen::Vector2f> vertices;
    vertices.emplace_back(100, 100);
    vertices.emplace_back(100, 200);
    vertices.emplace_back(200, 100);
    vertices.emplace_back(200, 200);

    SpaceRegion2D sr1(vertices);

    for (size_t i = 0; i < vertices.size(); i++) {
        vertices[i] += Eigen::Vector2f(50,50);
    }

    SpaceRegion2D sr2(vertices);

//    SpaceRegion2D sr = sr1.unionWith(sr2).complement();
    SpaceRegion2D sr = sr1.intersectionWith(sr2.complement()).unionWith(sr2.intersectionWith(sr1.complement()));

    pangolin::ManagedImage<unsigned char> img(640,480);
    for (size_t x = 0; x < img.w; x++) {
        for (size_t y = 0; y < img.h; y++) {
            img(x,y) = (unsigned char)255*sr.containsPoint(Eigen::Vector2f(x,y));
        }
    }

    ImageViewer viz("win", "disp");
    viz.setImage(img, "GRAY8");
    while (!viz.wasStopped()) {
        viz.spinOnce();
    }

    return 0;
}

