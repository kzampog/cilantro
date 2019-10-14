#pragma once

#include <fstream>
#include <cilantro/3rd_party/tinyply/tinyply.h>
#include <cilantro/data_containers.hpp>

namespace cilantro {
    template <typename ScalarT>
    void readPointCloudFromPLYFile(const std::string &file_name,
                                   VectorSet<ScalarT,3> &points,
                                   VectorSet<ScalarT,3> &normals,
                                   VectorSet<float,3> &colors)
    {
        std::ifstream ss(file_name, std::ios::binary);
        tinyply::PlyFile file;

        file.parse_header(ss);

        const auto& elements = file.get_elements();

        bool has_vertex_element = false;
        size_t vertex_element_ind;
        for (size_t i = 0; i < elements.size(); i++) {
            if (elements[i].name == "vertex") {
                has_vertex_element = true;
                vertex_element_ind = i;
                break;
            }
        }

        if (!has_vertex_element) return;

        const auto& vertex_props = elements[vertex_element_ind].properties;

        bool has_point = false, has_normal = false, has_color = false;
        tinyply::Type point_type, normal_type, color_type;
        for (size_t i = 0; i < vertex_props.size(); i++) {
            if (vertex_props[i].name == "x") {
                has_point = true;
                point_type = vertex_props[i].propertyType;
            } else if (vertex_props[i].name == "nx") {
                has_normal = true;
                normal_type = vertex_props[i].propertyType;
            } else if (vertex_props[i].name == "red") {
                has_color = true;
                color_type = vertex_props[i].propertyType;
            }
        }

        // Data holders
        std::shared_ptr<tinyply::PlyData> vertex_data, normal_data, color_data;

        // Initialize PLY data holders
        if (has_point) vertex_data = file.request_properties_from_element("vertex", {"x", "y", "z"});
        if (has_normal) normal_data = file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
        if (has_color) color_data = file.request_properties_from_element("vertex", {"red", "green", "blue"});

        // Read PLY data
        file.read(ss);

        // Populate cloud
        if (has_point) {
            if (point_type == tinyply::Type::FLOAT32) {
                points = ConstDataMatrixMap<float,3>((float *)vertex_data->buffer.get(), vertex_data->count).template cast<ScalarT>();
            } else if (point_type == tinyply::Type::FLOAT64) {
                points = ConstDataMatrixMap<double,3>((double *)vertex_data->buffer.get(), vertex_data->count).template cast<ScalarT>();
            }
        }

        if (has_normal) {
            if (normal_type == tinyply::Type::FLOAT32) {
                normals = ConstDataMatrixMap<float,3>((float *)normal_data->buffer.get(), normal_data->count).template cast<ScalarT>();
            } else if (normal_type == tinyply::Type::FLOAT64) {
                normals = ConstDataMatrixMap<double,3>((double *)normal_data->buffer.get(), normal_data->count).template cast<ScalarT>();
            }
        }

        if (has_color) {
            colors = ConstDataMatrixMap<uint8_t,3>((uint8_t *)color_data->buffer.get(), color_data->count).cast<float>()*(1.0f/255.0f);
        }
    }

    template <typename ScalarT, typename ScalarOutT = ScalarT>
    void writePointCloudToPLYFile(const std::string &file_name,
                                  const ConstVectorSetMatrixMap<ScalarT,3> &points,
                                  const ConstVectorSetMatrixMap<ScalarT,3> &normals,
                                  const ConstVectorSetMatrixMap<float,3> &colors,
                                  bool binary = true)
    {
        tinyply::PlyFile file;

        cilantro::VectorSet<ScalarOutT,3> points_tmp, normals_tmp;
        if (std::is_same<ScalarT,ScalarOutT>::value == false) {
            points_tmp = points.template cast<ScalarOutT>();
            if (points.cols() == normals.cols()) normals_tmp = normals.template cast<ScalarOutT>();
        }

        auto vertex_data = (std::is_same<ScalarT,ScalarOutT>::value) ? ConstVectorSetMatrixMap<ScalarOutT,3>((const ScalarOutT *)points.data(), points.cols()) :
                                                                       ConstVectorSetMatrixMap<ScalarOutT,3>((const ScalarOutT *)points_tmp.data(), points_tmp.cols());

        if (std::is_same<ScalarOutT,float>::value == true) {
            file.add_properties_to_element("vertex", {"x", "y", "z"}, tinyply::Type::FLOAT32, vertex_data.cols(), (uint8_t*)vertex_data.data(), tinyply::Type::INVALID, 0);
        } else if (std::is_same<ScalarOutT,double>::value == true) {
            file.add_properties_to_element("vertex", {"x", "y", "z"}, tinyply::Type::FLOAT64, vertex_data.cols(), (uint8_t*)vertex_data.data(), tinyply::Type::INVALID, 0);
        }

        auto normal_data = (std::is_same<ScalarT,ScalarOutT>::value) ? ConstVectorSetMatrixMap<ScalarOutT,3>((const ScalarOutT *)normals.data(), normals.cols()) :
                                                                       ConstVectorSetMatrixMap<ScalarOutT,3>((const ScalarOutT *)normals_tmp.data(), normals_tmp.cols());

        if (normals.cols() == points.cols()) {
            if (std::is_same<ScalarOutT,float>::value == true) {
                file.add_properties_to_element("vertex", {"nx", "ny", "nz"}, tinyply::Type::FLOAT32, normal_data.cols(), (uint8_t*)normal_data.data(), tinyply::Type::INVALID, 0);
            } else if (std::is_same<ScalarOutT,double>::value == true) {
                file.add_properties_to_element("vertex", {"nx", "ny", "nz"}, tinyply::Type::FLOAT64, normal_data.cols(), (uint8_t*)normal_data.data(), tinyply::Type::INVALID, 0);
            }
        }

        VectorSet<uint8_t,3> color_data;
        if (colors.cols() == points.cols()) {
            color_data = (255.0f*colors).cast<uint8_t>();
            file.add_properties_to_element("vertex", {"red", "green", "blue"}, tinyply::Type::UINT8, color_data.cols(), (uint8_t*)color_data.data(), tinyply::Type::INVALID, 0);
        }

        // Write to file
        std::filebuf fb;
        if (binary) {
            fb.open(file_name, std::ios::out | std::ios::binary);
        } else {
            fb.open(file_name, std::ios::out);
        }
        std::ostream output_stream(&fb);
        file.write(output_stream, binary);
    }

    template<class Matrix>
    void readEigenMatrixFromFile(const std::string &file_name, Matrix &matrix, bool binary = true) {
        if (binary) {
            std::ifstream in(file_name, std::ifstream::binary);
            typename Matrix::Index rows = 0, cols = 0;
            in.read((char*)(&rows), sizeof(typename Matrix::Index));
            in.read((char*)(&cols), sizeof(typename Matrix::Index));
            matrix.resize(rows, cols);
            in.read((char*)matrix.data(), rows*cols*sizeof(typename Matrix::Scalar));
            in.close();
        } else {
            std::ifstream in(file_name.c_str(), std::ios::in);

            // Read file contents into a vector
            std::string line;
            typename Matrix::Scalar d;
            std::vector<typename Matrix::Scalar> v;
            size_t n_rows = 0;
            while (getline(in, line)) {
                n_rows++;
                std::stringstream input_line(line);
                while (!input_line.eof()) {
                    input_line >> d;
                    v.emplace_back(d);
                }
            }
            in.close();

            // Construct matrix
            size_t n_cols = v.size()/n_rows;
            matrix.resize(n_rows, n_cols);
            for (size_t i = 0; i < n_rows; i++) {
                for (size_t j = 0; j < n_cols; j++) {
                    matrix(i,j) = v[i * n_cols + j];
                }
            }
        }
    }

    template<class Matrix>
    void writeEigenMatrixToFile(const std::string &file_name, const Matrix &matrix, bool binary = true) {
        if (binary) {
            std::ofstream out(file_name, std::ofstream::binary);
            typename Matrix::Index rows = matrix.rows(), cols = matrix.cols();
            out.write((char*)(&rows), sizeof(typename Matrix::Index));
            out.write((char*)(&cols), sizeof(typename Matrix::Index));
            out.write((char*)matrix.data(), rows*cols*sizeof(typename Matrix::Scalar));
            out.close();
        } else {
            std::ofstream out(file_name.c_str(), std::ios::out);
            out << matrix << "\n";
            out.close();
        }
    }

    template<typename ScalarT>
    void readVectorFromFile(const std::string &file_name, std::vector<ScalarT> &vec, bool binary = true) {
        Eigen::Matrix<ScalarT, Eigen::Dynamic, 1> vec_e;
        readEigenMatrixFromFile<Eigen::Matrix<ScalarT, Eigen::Dynamic, 1> >(file_name, vec_e, binary);
        vec.resize(vec_e.size());
        Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>::Map(&vec[0], vec_e.size()) = vec_e;
    }

    template<typename ScalarT>
    void writeVectorToFile(const std::string &file_name, const std::vector<ScalarT> &vec, bool binary = true) {
        Eigen::Matrix<ScalarT, Eigen::Dynamic, 1> vec_e = Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>::Map(&vec[0], vec.size());
        writeEigenMatrixToFile<Eigen::Matrix<ScalarT, Eigen::Dynamic, 1> >(file_name, vec_e, binary);
    }

    inline size_t getFileSizeInBytes(const std::string &file_name) {
        return std::ifstream(file_name, std::ifstream::ate | std::ifstream::binary).tellg();
    }

    size_t readRawDataFromFile(const std::string &file_name, void * data_ptr, size_t num_bytes = 0);

    void writeRawDataToFile(const std::string &file_name, const void * data_ptr, size_t num_bytes);
}
