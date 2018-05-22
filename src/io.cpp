#include <cilantro/io.hpp>

namespace cilantro {
    size_t readRawDataFromFile(const std::string &file_name, void * data_ptr, size_t num_bytes) {
        size_t num_bytes_to_read = (num_bytes == 0) ? getFileSizeInBytes(file_name) : num_bytes;
        std::ifstream in(file_name, std::ios::in | std::ios::binary);
        in.read((char*)data_ptr, num_bytes_to_read);
        in.close();
        return num_bytes_to_read;
    }

    void writeRawDataToFile(const std::string &file_name, const void * data_ptr, size_t num_bytes) {
        std::ofstream out(file_name, std::ios::out | std::ios::binary);
        out.write((char*)data_ptr, num_bytes);
        out.close();
    }
}
