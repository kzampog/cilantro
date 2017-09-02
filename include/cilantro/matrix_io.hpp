#pragma once

#include <fstream>
#include <Eigen/Dense>

template<class Matrix>
void readEigenMatrixFromFile(const std::string &filename, Matrix &matrix, bool binary = true) {
    if (binary) {
        std::ifstream in(filename, std::ifstream::binary);
        typename Matrix::Index rows = 0, cols = 0;
        in.read((char*)(&rows), sizeof(typename Matrix::Index));
        in.read((char*)(&cols), sizeof(typename Matrix::Index));
        matrix.resize(rows, cols);
        in.read((char*)matrix.data(), rows*cols*sizeof(typename Matrix::Scalar));
        in.close();
    } else {
        std::ifstream in(filename.c_str(), std::ios::in);

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
void writeEigenMatrixToFile(const std::string &filename, const Matrix &matrix, bool binary = true) {
    if (binary) {
        std::ofstream out(filename, std::ofstream::binary);
        typename Matrix::Index rows = matrix.rows(), cols = matrix.cols();
        out.write((char*)(&rows), sizeof(typename Matrix::Index));
        out.write((char*)(&cols), sizeof(typename Matrix::Index));
        out.write((char*)matrix.data(), rows*cols*sizeof(typename Matrix::Scalar));
        out.close();
    } else {
        std::ofstream out(filename.c_str(), std::ios::out);
        out << matrix << "\n";
        out.close();
    }
}

template<typename ScalarT>
void readVectorFromFile(const std::string &filename, std::vector<ScalarT> &vec, bool binary = true) {
    Eigen::Matrix<ScalarT, Eigen::Dynamic, 1> vec_e;
    readEigenMatrixFromFile<Eigen::Matrix<ScalarT, Eigen::Dynamic, 1> >(filename, vec_e, binary);
    vec.resize(vec_e.size());
    Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>::Map(&vec[0], vec_e.size()) = vec_e;
}

template<typename ScalarT>
void writeVectorToFile(const std::string &filename, const std::vector<ScalarT> &vec, bool binary = true) {
    Eigen::Matrix<ScalarT, Eigen::Dynamic, 1> vec_e = Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>::Map(&vec[0], vec.size());
    writeEigenMatrixToFile<Eigen::Matrix<ScalarT, Eigen::Dynamic, 1> >(filename, vec_e, binary);
}
