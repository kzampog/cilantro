#pragma once

#include <fstream>
#include <cilantro/data_containers.hpp>

namespace cilantro {
    template<class Matrix>
    void readEigenMatrixFromFile(const std::string &file_path, Matrix &matrix, bool binary = true) {
        if (binary) {
            std::ifstream in(file_path, std::ifstream::binary);
            typename Matrix::Index rows = 0, cols = 0;
            in.read((char*)(&rows), sizeof(typename Matrix::Index));
            in.read((char*)(&cols), sizeof(typename Matrix::Index));
            matrix.resize(rows, cols);
            in.read((char*)matrix.data(), rows*cols*sizeof(typename Matrix::Scalar));
            in.close();
        } else {
            std::ifstream in(file_path.c_str(), std::ios::in);

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
    void writeEigenMatrixToFile(const std::string &file_path, const Matrix &matrix, bool binary = true) {
        if (binary) {
            std::ofstream out(file_path, std::ofstream::binary);
            typename Matrix::Index rows = matrix.rows(), cols = matrix.cols();
            out.write((char*)(&rows), sizeof(typename Matrix::Index));
            out.write((char*)(&cols), sizeof(typename Matrix::Index));
            out.write((char*)matrix.data(), rows*cols*sizeof(typename Matrix::Scalar));
            out.close();
        } else {
            std::ofstream out(file_path.c_str(), std::ios::out);
            out << matrix << "\n";
            out.close();
        }
    }

    template<typename ScalarT>
    void readVectorFromFile(const std::string &file_path, std::vector<ScalarT> &vec, bool binary = true) {
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> vec_e;
        readEigenMatrixFromFile<Eigen::Matrix<ScalarT,Eigen::Dynamic,1> >(file_path, vec_e, binary);
        vec.resize(vec_e.size());
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1>::Map(&vec[0], vec_e.size()) = vec_e;
    }

    template<typename ScalarT>
    void writeVectorToFile(const std::string &file_path, const std::vector<ScalarT> &vec, bool binary = true) {
        Eigen::Matrix<ScalarT,Eigen::Dynamic,1> vec_e = Eigen::Matrix<ScalarT,Eigen::Dynamic,1>::Map(&vec[0], vec.size());
        writeEigenMatrixToFile<Eigen::Matrix<ScalarT,Eigen::Dynamic,1> >(file_path, vec_e, binary);
    }

    inline size_t getFileSizeInBytes(const std::string &file_path) {
        return std::ifstream(file_path, std::ifstream::ate | std::ifstream::binary).tellg();
    }

    inline size_t readRawDataFromFile(const std::string &file_path, void * data_ptr, size_t num_bytes = 0) {
        size_t num_bytes_to_read = (num_bytes == 0) ? getFileSizeInBytes(file_path) : num_bytes;
        std::ifstream in(file_path, std::ios::in | std::ios::binary);
        in.read((char*)data_ptr, num_bytes_to_read);
        in.close();
        return num_bytes_to_read;
    }

    inline void writeRawDataToFile(const std::string &file_path, const void * data_ptr, size_t num_bytes) {
        std::ofstream out(file_path, std::ios::out | std::ios::binary);
        out.write((char*)data_ptr, num_bytes);
        out.close();
    }
}
