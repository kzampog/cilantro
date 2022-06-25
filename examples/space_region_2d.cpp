#include <cilantro/visualization.hpp>
#include <cilantro/spatial/space_region.hpp>

int main(int argc, char** argv) {
  std::vector<Eigen::Vector2f> vertices;
  vertices.emplace_back(100, 100);
  vertices.emplace_back(100, 200);
  vertices.emplace_back(200, 100);
  vertices.emplace_back(200, 200);

  cilantro::SpaceRegion2f sr1(vertices);

  for (size_t i = 0; i < vertices.size(); i++) {
    vertices[i] += Eigen::Vector2f(50, 50);
  }

  cilantro::SpaceRegion2f sr2(vertices);

  cilantro::SpaceRegion2f sr3(std::vector<Eigen::Vector3f>(1, Eigen::Vector3f(-1, 1, -70)));

  // SpaceRegion2D sr = sr1.unionWith(sr2).complement();
  cilantro::SpaceRegion2f sr = sr1.intersectionWith(sr2.complement())
                                   .unionWith(sr2.intersectionWith(sr1.complement()))
                                   .intersectionWith(sr3);
  // SpaceRegion2D sr = sr2.relativeComplement(sr1);

  sr.transform(Eigen::Rotation2D<float>(-M_PI / 4.0).toRotationMatrix(), Eigen::Vector2f(80, 220));

  pangolin::ManagedImage<unsigned char> img(640, 480);
  for (size_t x = 0; x < img.w; x++) {
    for (size_t y = 0; y < img.h; y++) {
      img(x, y) = (unsigned char)255 * sr.containsPoint(Eigen::Vector2f(x, y));
    }
  }

  const std::string window_name = "SpaceRegion2D example";
  pangolin::CreateWindowAndBind(window_name, 640, 480);
  cilantro::ImageViewer viz(window_name, "disp");
  viz.setImage(img, "GRAY8");
  while (!viz.wasStopped()) {
    viz.spinOnce();
  }

  return 0;
}
