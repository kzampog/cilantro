#include <cilantro/io.hpp>
#include <cilantro/3rd_party/tinyply/tinyply.h>

namespace cilantro {
    void readPointCloudFromPLYFile(const std::string &filename, PointCloud &cloud) {
        // Data holders
        std::vector<float> vertex_data;
        std::vector<float> normal_data;
        std::vector<uint8_t> color_data;

        std::ifstream ss(filename, std::ios::binary);
        tinyply::PlyFile file(ss);

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
    }

    void writePointCloudToPLYFile(const std::string &filename, const PointCloud &cloud, bool binary) {
        tinyply::PlyFile file;

        std::vector<float> vertex_data(3*cloud.size());
        std::memcpy(vertex_data.data(), cloud.points.data(), 3*cloud.size()*sizeof(float));
        file.add_properties_to_element("vertex", {"x", "y", "z"}, vertex_data);

        std::vector<float> normal_data;
        if (cloud.hasNormals()) {
            normal_data.resize(3*cloud.normals.size());
            std::memcpy(normal_data.data(), cloud.normals.data(), 3*cloud.normals.size()*sizeof(float));
            file.add_properties_to_element("vertex", {"nx", "ny", "nz"}, normal_data);
        }

        std::vector<uint8_t> color_data;
        if (cloud.hasColors()) {
            color_data.resize(3*cloud.colors.size());
            for (int i = 0; i < cloud.colors.size(); i++) {
                color_data[3 * i + 0] = (uint8_t)(cloud.colors[i](0)*255.0f);
                color_data[3 * i + 1] = (uint8_t)(cloud.colors[i](1)*255.0f);
                color_data[3 * i + 2] = (uint8_t)(cloud.colors[i](2)*255.0f);
            }
            file.add_properties_to_element("vertex", {"red", "green", "blue"}, color_data);
        }

        // Write to file
        std::filebuf fb;
        fb.open(filename, std::ios::out | std::ios::binary);
        std::ostream output_stream(&fb);
        file.write(output_stream, binary);
        fb.close();
    }

    size_t getFileSizeInBytes(const std::string &file_name) {
        std::ifstream in(file_name, std::ifstream::ate | std::ifstream::binary);
        return in.tellg();
    }

    size_t readRawDataFromFile(const std::string &file_name, void * data_ptr, size_t num_bytes) {
        size_t num_bytes_to_read = (num_bytes == 0) ? getFileSizeInBytes(file_name) : num_bytes;
        std::ifstream in(file_name, std::ios::in | std::ios::binary);
        in.read((char*)data_ptr, num_bytes_to_read);
        in.close();
        return num_bytes_to_read;
    }

    void writeRawDataToFile(const std::string &file_name, void * data_ptr, size_t num_bytes) {
        std::ofstream out(file_name, std::ios::out | std::ios::binary);
        out.write((char*)data_ptr, num_bytes);
        out.close();
    }
}

//    void readPointCloudFromPLYFile(const std::string &filename, PointCloud &cloud) {
//        std::ifstream ss(filename, std::ios::binary);
//        tinyply::PlyFile file;
//        file.parse_header(ss);
//
//        // Initialize PLY data holders
//        std::shared_ptr<tinyply::PlyData> vertex_data = file.request_properties_from_element("vertex", {"x", "y", "z"});
//        std::shared_ptr<tinyply::PlyData> normal_data = file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
//        std::shared_ptr<tinyply::PlyData> color_data = file.request_properties_from_element("vertex", {"red", "green", "blue"});
//
//        // Read PLY data
//        file.read(ss);
//
//        // Populate cloud
//        cloud.points.resize(vertex_data->count);
//        std::memcpy(cloud.points.data(), vertex_data->buffer.get(), vertex_data->buffer.size_bytes());
//
//        cloud.normals.resize(normal_data->count);
//        std::memcpy(cloud.normals.data(), normal_data->buffer.get(), normal_data->buffer.size_bytes());
//
//        cloud.colors.resize(color_data->count);
//        uint8_t * cdata = (uint8_t *)color_data->buffer.get();
//        for (int i = 0; i < color_data->count; i++) {
//            cloud.colors[i] = Eigen::Vector3f(cdata[3*i]/255.0f, cdata[3*i+1]/255.0f, cdata[3*i+2]/255.0f);
//        }
//    }
//
//    void writePointCloudToPLYFile(const std::string &filename, const PointCloud &cloud, bool binary) {
//        tinyply::PlyFile file;
//
//        file.add_properties_to_element("vertex", {"x", "y", "z"}, tinyply::Type::FLOAT32, cloud.points.size(), (uint8_t *)(cloud.points.data()), tinyply::Type::INVALID, 0);
//
//        if (cloud.hasNormals()) {
//            file.add_properties_to_element("vertex", {"nx", "ny", "nz"}, tinyply::Type::FLOAT32, cloud.normals.size(), (uint8_t *)(cloud.normals.data()), tinyply::Type::INVALID, 0);
//        }
//
//        std::vector<uint8_t> color_data;
//        if (cloud.hasColors()) {
//            color_data.resize(3*cloud.colors.size());
//            for (int i = 0; i < cloud.colors.size(); i++) {
//                color_data[3*i+0] = (uint8_t)(cloud.colors[i](0)*255.0f);
//                color_data[3*i+1] = (uint8_t)(cloud.colors[i](1)*255.0f);
//                color_data[3*i+2] = (uint8_t)(cloud.colors[i](2)*255.0f);
//            }
//            file.add_properties_to_element("vertex", {"red", "green", "blue"}, tinyply::Type::UINT8, cloud.colors.size(), (uint8_t *)(color_data.data()), tinyply::Type::INVALID, 0);
//        }
//
//        // Write to file
//        std::filebuf fb;
//        fb.open(filename, std::ios::out | std::ios::binary);
//        std::ostream output_stream(&fb);
//        file.write(output_stream, binary);
//        fb.close();
//    }
