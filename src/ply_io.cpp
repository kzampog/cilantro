#include <fstream>
#include <tinyply/tinyply.h>
#include <sisyphus/ply_io.h>

void readPointCloudFromPLYFile(const std::string &filename, PointCloud &cloud) {
    try {
        std::ifstream ss(filename, std::ios::binary);
        tinyply::PlyFile file(ss);

        // Data holders
        std::vector<float> vertex_data;
        std::vector<float> normal_data;
        std::vector<uint8_t> color_data;

        // Initialize PLY data holders
        size_t vertex_count = file.request_properties_from_element("vertex", {"x", "y", "z"}, vertex_data);
        size_t normal_count = file.request_properties_from_element("vertex", {"nx", "ny", "nz"}, normal_data);
        size_t color_count = file.request_properties_from_element("vertex", {"red", "green", "blue"}, color_data);

        // Read PLY data
        file.read(ss);

        // Populate cloud
        cloud.points.resize(vertex_count);
        std::memcpy(cloud.points.data(), vertex_data.data(), 3*vertex_count*sizeof(float));

        cloud.normals.resize(normal_count);
        std::memcpy(cloud.normals.data(), normal_data.data(), 3*normal_count*sizeof(float));

        cloud.colors.resize(color_count);
        for (int i = 0; i < color_count; i++) {
            cloud.colors[i] = Eigen::Vector3f(color_data[3*i], color_data[3*i+1], color_data[3*i+2])/255.0f;
        }

    } catch (const std::exception &e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }
}

void writePointCloudToPLYFile(const std::string &filename, PointCloud &cloud) {
    // Populate data holders
    std::vector<float> vertex_data(3*cloud.points.size());
    std::memcpy(vertex_data.data(), cloud.points.data(), 3*cloud.points.size()*sizeof(float));

    std::vector<float> normal_data(3*cloud.normals.size());
    std::memcpy(normal_data.data(), cloud.normals.data(), 3*cloud.normals.size()*sizeof(float));

    std::vector<uint8_t> color_data(3*cloud.colors.size());
    for (int i = 0; i < cloud.colors.size(); i++) {
        color_data[3*i + 0] = (uint8_t)(cloud.colors[i](0)*255.0f);
        color_data[3*i + 1] = (uint8_t)(cloud.colors[i](1)*255.0f);
        color_data[3*i + 2] = (uint8_t)(cloud.colors[i](2)*255.0f);
    }

    // Write to file
    std::filebuf fb;
    fb.open(filename, std::ios::out | std::ios::binary);
    std::ostream outputStream(&fb);

    tinyply::PlyFile file_to_write;

    file_to_write.add_properties_to_element("vertex", {"x", "y", "z"}, vertex_data);
    file_to_write.add_properties_to_element("vertex", {"nx", "ny", "nz"}, normal_data);
    file_to_write.add_properties_to_element("vertex", {"red", "green", "blue"}, color_data);

    file_to_write.write(outputStream, true);

    fb.close();
}
