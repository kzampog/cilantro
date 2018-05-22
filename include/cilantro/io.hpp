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
        tinyply::PlyFile file(ss);

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
        tinyply::PlyProperty::Type point_type, normal_type, color_type;
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
        std::vector<float> vertex_data_f;
        std::vector<double> vertex_data_d;
        std::vector<float> normal_data_f;
        std::vector<double> normal_data_d;
        std::vector<uint8_t> color_data;

        if (!has_point) return;

        // Initialize PLY data holders
        if (point_type == tinyply::PlyProperty::Type::FLOAT32) {
            file.request_properties_from_element<float>("vertex", {"x", "y", "z"}, vertex_data_f);
        } else if (point_type == tinyply::PlyProperty::Type::FLOAT64) {
            file.request_properties_from_element<double>("vertex", {"x", "y", "z"}, vertex_data_d);
        }

        if (has_normal && normal_type == tinyply::PlyProperty::Type::FLOAT32) {
            file.request_properties_from_element<float>("vertex", {"nx", "ny", "nz"}, normal_data_f);
        } else if (has_normal && normal_type == tinyply::PlyProperty::Type::FLOAT64) {
            file.request_properties_from_element<double>("vertex", {"nx", "ny", "nz"}, normal_data_d);
        }

        if (has_color && color_type == tinyply::PlyProperty::Type::UINT8) {
            file.request_properties_from_element("vertex", {"red", "green", "blue"}, color_data);
        }

        // Read PLY data
        file.read(ss);

        // Populate cloud
        if (point_type == tinyply::PlyProperty::Type::FLOAT32) {
            points = ConstDataMatrixMap<float,3>(vertex_data_f).template cast<ScalarT>();
        } else  if (point_type == tinyply::PlyProperty::Type::FLOAT64) {
            points = ConstDataMatrixMap<double,3>(vertex_data_d).template cast<ScalarT>();
        }

        if (normal_type == tinyply::PlyProperty::Type::FLOAT32) {
            normals = ConstDataMatrixMap<float,3>(normal_data_f).template cast<ScalarT>();
        } else if (normal_type == tinyply::PlyProperty::Type::FLOAT64) {
            normals = ConstDataMatrixMap<double,3>(normal_data_d).template cast<ScalarT>();
        }

        colors = ConstDataMatrixMap<uint8_t,3>(color_data).cast<float>()*(1.0f/255.0f);
    }

    template <typename ScalarT, typename ScalarOutT = ScalarT>
    void writePointCloudToPLYFile(const std::string &file_name,
                                  const ConstVectorSetMatrixMap<ScalarT,3> &points,
                                  const ConstVectorSetMatrixMap<ScalarT,3> &normals,
                                  const ConstVectorSetMatrixMap<float,3> &colors,
                                  bool binary = true)
    {
        tinyply::PlyFile file;

        std::vector<ScalarOutT> vertex_data(3*points.cols());
        DataMatrixMap<ScalarOutT,3>(vertex_data).eigenMap() = points.template cast<ScalarOutT>();
        file.add_properties_to_element<ScalarT>("vertex", {"x", "y", "z"}, vertex_data);

        std::vector<ScalarOutT> normal_data;
        if (normals.cols() == points.cols()) {
            normal_data.resize(3*normals.cols());
            DataMatrixMap<ScalarOutT,3>(normal_data).eigenMap() = normals.template cast<ScalarOutT>();
            file.add_properties_to_element("vertex", {"nx", "ny", "nz"}, normal_data);
        }

        std::vector<uint8_t> color_data;
        if (colors.cols() == points.cols()) {
            color_data.resize(3*colors.cols());
            DataMatrixMap<uint8_t,3>(color_data).eigenMap() = (255.0f*colors).cast<uint8_t>();
            file.add_properties_to_element("vertex", {"red", "green", "blue"}, color_data);
        }

        // Write to file
        std::filebuf fb;
        fb.open(file_name, std::ios::out | std::ios::binary);
        std::ostream output_stream(&fb);
        file.write(output_stream, binary);
        fb.close();
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
