//
// Created by kganguly on 5/4/17.
//

#include "sisyphus/ply_reader.h"

void PlyReader::read_ply_file(const std::string &filename, PointCloud &poClo) {
    try {
        std::ifstream ss(filename, std::ios::binary);

        tinyply::PlyFile file(ss);

        // Data holders
        std::vector<float> verts;
        std::vector<float> norms;
        std::vector<uint8_t> colors;

        // Return values for each data type
        uint32_t vertexCount;
        vertexCount = 0;

        // Initialize PLY data holders
        vertexCount = file.request_properties_from_element("vertex", {"x", "y", "z"}, verts);
        file.request_properties_from_element("vertex", {"nx", "ny", "nz"}, norms);
        file.request_properties_from_element("vertex", {"red", "green", "blue"}, colors);

        // Read PLY data
        file.read(ss);

        // Save to PointCloud struct
        poClo.num_points = vertexCount;

        // Add to vertex list
        for (int i = 0; i < verts.size(); i += 3) {
            Eigen::Vector3f add_pt(verts.data()[i], verts.data()[i + 1], verts.data()[i + 2]);
            poClo.points.push_back(add_pt);
        }
        // Add to normals list
        for (int i = 0; i < norms.size(); i += 3) {
            Eigen::Vector3f add_norm(norms.data()[i], norms.data()[i + 1], norms.data()[i + 2]);
            poClo.normals.push_back(add_norm);
        }
        // Add to colors list
        if (colors.size() > 0) {
            for (int i = 0; i < colors.size(); i += 3) {
                Eigen::Vector3f add_col((float) colors.data()[i] / 255.0, (float) colors.data()[i + 1] / 255.0,
                                        (float) colors.data()[i + 2] / 255.0);
                poClo.colors.push_back(add_col);
            }
        } else {
            for (int i = 0; i < verts.size(); i += 3) {
                Eigen::Vector3f add_col(0.0f, 0.0f, 0.0f);
                poClo.colors.push_back(add_col);
            }
        }

    } catch (const std::exception &e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }
}
