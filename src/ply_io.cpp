//
// Created by kganguly on 5/4/17.
//

#include "sisyphus/ply_io.h"

void PlyIO::write_ply_file(const std::string &filename, PointCloud &poClo) {
    // Data holders
    std::vector<float> verts;
    std::vector<float> norms;
    std::vector<uint8_t> colors;

    // Add to vertex list
    for (int i = 0; i < poClo.points.size(); i++) {
        Eigen::Vector3f add_pt(poClo.points.data()[i]);
        verts.push_back(add_pt[0]);
        verts.push_back(add_pt[1]);
        verts.push_back(add_pt[2]);
    }
    // Add to normals list
    for (int i = 0; i < poClo.normals.size(); i++) {
        Eigen::Vector3f add_norm(poClo.normals.data()[i]);
        norms.push_back(add_norm[0]);
        norms.push_back(add_norm[1]);
        norms.push_back(add_norm[2]);
    }
    // Add to colors list
    for (int i = 0; i < poClo.colors.size(); i++) {
        Eigen::Vector3f add_col(poClo.colors.data()[i]);
        colors.push_back((uint8_t) add_col[0] * 255);
        colors.push_back((uint8_t) add_col[1] * 255);
        colors.push_back((uint8_t) add_col[2] * 255);
    }

    std::filebuf fb;
    fb.open(filename, std::ios::out | std::ios::binary);
    std::ostream outputStream(&fb);

    tinyply::PlyFile file_to_write;

    // Initialize PLY data holders
    file_to_write.add_properties_to_element("vertex", {"x", "y", "z"}, verts);
    file_to_write.add_properties_to_element("vertex", {"nx", "ny", "nz"}, norms);
    file_to_write.add_properties_to_element("vertex", {"red", "green", "blue"}, colors);

    // Write PLY data
    file_to_write.write(outputStream, true);

    fb.close();
}

void PlyIO::read_ply_file(const std::string &filename, PointCloud &poClo) {
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
