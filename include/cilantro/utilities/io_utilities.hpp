#pragma once

#include <fstream>
#include <cilantro/core/data_containers.hpp>

namespace cilantro {
    template<class Matrix>
    bool readEigenMatrixFromFile(const std::string &file_path, Matrix &matrix, bool binary = true) {
        if (binary) {
            std::ifstream in(file_path, std::ifstream::binary);
            if (!in) return false;
            typename Matrix::Index rows = 0, cols = 0;
            if (!in.read((char*)(&rows), sizeof(typename Matrix::Index))) return false;
            if (!in.read((char*)(&cols), sizeof(typename Matrix::Index))) return false;
            matrix.resize(rows, cols);
            return !!in.read((char*)matrix.data(), rows*cols*sizeof(typename Matrix::Scalar));
        } else {
            std::ifstream in(file_path.c_str(), std::ios::in);
            if (!in) return false;

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
            if (!in) return false;

            // Construct matrix
            size_t n_cols = v.size()/n_rows;
            matrix.resize(n_rows, n_cols);
            for (size_t i = 0; i < n_rows; i++) {
                for (size_t j = 0; j < n_cols; j++) {
                    matrix(i,j) = v[i * n_cols + j];
                }
            }
            return true;
        }
    }

    template<class Matrix>
    bool writeEigenMatrixToFile(const std::string &file_path, const Matrix &matrix, bool binary = true) {
        if (binary) {
            std::ofstream out(file_path, std::ofstream::binary);
            if (!out) return false;
            typename Matrix::Index rows = matrix.rows(), cols = matrix.cols();
            if (!out.write((char*)(&rows), sizeof(typename Matrix::Index))) return false;
            if (!out.write((char*)(&cols), sizeof(typename Matrix::Index))) return false;
            return !!out.write((char*)matrix.data(), rows*cols*sizeof(typename Matrix::Scalar));
        } else {
            std::ofstream out(file_path.c_str(), std::ios::out);
            if (!out) return false;
            return !!(out << matrix << "\n");
        }
    }

    inline bool fileExists(const std::string &file_path) {
        return std::ifstream(file_path).is_open();
    }

    inline size_t getFileSizeInBytes(const std::string &file_path) {
        if (!fileExists(file_path)) return 0;
        return std::ifstream(file_path, std::ifstream::ate | std::ifstream::binary).tellg();
    }

    inline size_t readRawDataFromFile(const std::string &file_path, void * data_ptr, size_t num_bytes = 0) {
        size_t num_bytes_to_read = (num_bytes == 0) ? getFileSizeInBytes(file_path) : num_bytes;
        std::ifstream in(file_path, std::ios::in | std::ios::binary);
        if (!in) return 0;
        in.read((char*)data_ptr, num_bytes_to_read);
        return in.gcount();
    }

    inline bool writeRawDataToFile(const std::string &file_path, const void * data_ptr, size_t num_bytes) {
        std::ofstream out(file_path, std::ios::out | std::ios::binary);
        if (!out) return false;
        return !!out.write((char*)data_ptr, num_bytes);
    }
}
