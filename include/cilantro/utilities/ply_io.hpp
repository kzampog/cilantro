#pragma once

#include <set>
#include <fstream>
#include <cilantro/3rd_party/tinyply/tinyply.h>
#include <cilantro/core/data_containers.hpp>

namespace cilantro {
    namespace internal {
        // From tinyply example-utils.hpp
        struct MemoryBuffer : public std::streambuf {
            char * p_start {nullptr};
            char * p_end {nullptr};
            size_t size;

            inline MemoryBuffer(char const * first_elem, size_t size)
                    : p_start(const_cast<char*>(first_elem)), p_end(p_start + size), size(size)
            {
                setg(p_start, p_start, p_end);
            }

            inline pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override {
                if (dir == std::ios_base::cur) gbump(static_cast<int>(off));
                else setg(p_start, (dir == std::ios_base::beg ? p_start : p_end) + off, p_end);
                return gptr() - p_start;
            }

            inline pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
                return seekoff(pos, std::ios_base::beg, which);
            }
        };

        // From tinyply example-utils.hpp
        struct MemoryStream : virtual MemoryBuffer, public std::istream {
            inline MemoryStream(char const * first_elem, size_t size)
                    : MemoryBuffer(first_elem, size), std::istream(static_cast<std::streambuf*>(this))
            {}
        };
    }

    class PLYReader {
    public:
        inline PLYReader(const std::string &file_path, bool preload = true) {
            if (preload) {
                preload_data_(file_path);
                input_stream_.reset(new internal::MemoryStream((char*)preloaded_data_.data(), preloaded_data_.size()));
            } else {
                input_stream_.reset(new std::ifstream(file_path, std::ios::binary));
            }

            ply_file_.parse_header(*input_stream_);
        }

        inline bool elementPropertyExists(const std::string &element_key,
                                          const std::vector<std::string> &property_keys) const
        {
            const auto elements = ply_file_.get_elements();
            ptrdiff_t element_ind = -1;
            for (size_t i = 0; i < elements.size(); i++) {
                if (elements[i].name == element_key) {
                    element_ind = i;
                    break;
                }
            }

            if (element_ind == -1) return false;

            std::set<std::string> el_props;
            for (const auto& p : elements[element_ind].properties) el_props.insert(p.name);

            for (const auto& req_p : property_keys) {
                if (el_props.find(req_p) == el_props.end()) return false;
            }

            return true;
        }

        inline std::shared_ptr<tinyply::PlyData> requestData(const std::string &element_key,
                                                             const std::vector<std::string> &property_keys,
                                                             const uint32_t list_size_hint = 0)
        {
            if (!elementPropertyExists(element_key, property_keys)) return std::shared_ptr<tinyply::PlyData>();
            return ply_file_.request_properties_from_element(element_key, property_keys, list_size_hint);
        }

        inline void readData() {
            ply_file_.read(*input_stream_);
            preloaded_data_.clear();
        }

    private:
        std::vector<uint8_t> preloaded_data_;
        std::unique_ptr<std::istream> input_stream_;
        tinyply::PlyFile ply_file_;

        inline void preload_data_(const std::string &file_path) {
            std::ifstream file(file_path, std::ios::binary);
            file.seekg(0, std::ios::end);
            size_t size_bytes = file.tellg();
            file.seekg(0, std::ios::beg);
            preloaded_data_.resize(size_bytes);
            file.read((char*)preloaded_data_.data(), size_bytes);
        }
    };

    class PLYWriter {
    public:
        inline PLYWriter(const std::string &file_path, bool binary = true) 
                : file_path_(file_path), binary_(binary)
        {}

        inline PLYWriter& addData(const std::string &element_key,
                                  const std::vector<std::string> &property_keys,
                                  const std::shared_ptr<tinyply::PlyData> &data_buffer,
                                  const tinyply::Type list_type = tinyply::Type::INVALID,
                                  const size_t list_count = 0)
        {
            if (!data_buffer) return *this;
            data_buffers_.emplace_back(data_buffer);
            ply_file_.add_properties_to_element(element_key, property_keys, data_buffer->t, data_buffer->count,
                                                data_buffer->buffer.get(), list_type, list_count);
            return *this;
        }

        inline void writeData() {
            std::filebuf fb;
            if (binary_) {
                fb.open(file_path_, std::ios::out | std::ios::binary);
            } else {
                fb.open(file_path_, std::ios::out);
            }
            std::ostream output_stream(&fb);
            ply_file_.write(output_stream, binary_);
        }

    private:
        std::vector<std::shared_ptr<tinyply::PlyData>> data_buffers_;
        tinyply::PlyFile ply_file_;
        std::string file_path_;
        bool binary_;
    };

    template <typename ScalarT, ptrdiff_t EigenDim>
    VectorSet<ScalarT,EigenDim> vectorSetFromPLYDataBuffer(const std::shared_ptr<tinyply::PlyData> &data_buffer, size_t dim) {
        VectorSet<ScalarT,EigenDim> data_matrix;

        if (!data_buffer) return data_matrix;

        switch (data_buffer->t) {
            case tinyply::Type::INT8:
                data_matrix = ConstVectorSetMatrixMap<int8_t,EigenDim>((int8_t *)data_buffer->buffer.get(), dim, data_buffer->count).template cast<ScalarT>();
                break;
            case tinyply::Type::UINT8:
                data_matrix = ConstVectorSetMatrixMap<uint8_t,EigenDim>((uint8_t *)data_buffer->buffer.get(), dim, data_buffer->count).template cast<ScalarT>();
                break;
            case tinyply::Type::INT16:
                data_matrix = ConstVectorSetMatrixMap<int16_t,EigenDim>((int16_t *)data_buffer->buffer.get(), dim, data_buffer->count).template cast<ScalarT>();
                break;
            case tinyply::Type::UINT16:
                data_matrix = ConstVectorSetMatrixMap<uint16_t,EigenDim>((uint16_t *)data_buffer->buffer.get(), dim, data_buffer->count).template cast<ScalarT>();
                break;
            case tinyply::Type::INT32:
                data_matrix = ConstVectorSetMatrixMap<int32_t,EigenDim>((int32_t *)data_buffer->buffer.get(), dim, data_buffer->count).template cast<ScalarT>();
                break;
            case tinyply::Type::UINT32:
                data_matrix = ConstVectorSetMatrixMap<uint32_t,EigenDim>((uint32_t *)data_buffer->buffer.get(), dim, data_buffer->count).template cast<ScalarT>();
                break;
            case tinyply::Type::FLOAT32:
                data_matrix = ConstVectorSetMatrixMap<float,EigenDim>((float *)data_buffer->buffer.get(), dim, data_buffer->count).template cast<ScalarT>();
                break;
            case tinyply::Type::FLOAT64:
                data_matrix = ConstVectorSetMatrixMap<double,EigenDim>((double *)data_buffer->buffer.get(), dim, data_buffer->count).template cast<ScalarT>();
                break;
            default:
                break;
        }

        return data_matrix;
    }

    template <typename ScalarT, ptrdiff_t EigenDim, typename ScalarOutT>
    std::shared_ptr<tinyply::PlyData> PLYDataBufferFromVectorSet(const ConstVectorSetMatrixMap<ScalarT,EigenDim> &data_matrix) {
        std::shared_ptr<tinyply::PlyData> data_buffer(new tinyply::PlyData());
        data_buffer->count = data_matrix.cols();

        if (std::is_same<ScalarOutT,int8_t>::value || std::is_same<ScalarOutT,char>::value) {
            data_buffer->t = tinyply::Type::INT8;
        } else if (std::is_same<ScalarOutT,uint8_t>::value || std::is_same<ScalarOutT,unsigned char>::value) {
            data_buffer->t = tinyply::Type::UINT8;
        } else if (std::is_same<ScalarOutT,int16_t>::value || std::is_same<ScalarOutT,short>::value) {
            data_buffer->t = tinyply::Type::INT16;
        } else if (std::is_same<ScalarOutT,uint16_t>::value || std::is_same<ScalarOutT,unsigned short>::value) {
            data_buffer->t = tinyply::Type::UINT16;
        } else if (std::is_same<ScalarOutT,int32_t>::value || std::is_same<ScalarOutT,int>::value) {
            data_buffer->t = tinyply::Type::INT32;
        } else if (std::is_same<ScalarOutT,uint32_t>::value || std::is_same<ScalarOutT,unsigned int>::value) {
            data_buffer->t = tinyply::Type::UINT32;
        } else if (std::is_same<ScalarOutT,float>::value) {
            data_buffer->t = tinyply::Type::FLOAT32;
        } else if (std::is_same<ScalarOutT,double>::value) {
            data_buffer->t = tinyply::Type::FLOAT64;
        } else {
            data_buffer->t = tinyply::Type::INVALID;
        }

        if (std::is_same<ScalarT,ScalarOutT>::value) {
            data_buffer->buffer = tinyply::Buffer((uint8_t *)data_matrix.data());
        } else {
            data_buffer->buffer = tinyply::Buffer(data_matrix.rows()*data_matrix.cols()*sizeof(ScalarOutT));
            DataMatrixMap<ScalarOutT,EigenDim>((ScalarOutT *)data_buffer->buffer.get(), data_matrix.rows(), data_matrix.cols()).base() = data_matrix.template cast<ScalarOutT>();
        }

        return data_buffer;
    }
}
