#include "ply_reader.h"

using namespace ply_reader;
using namespace std;

//////////////////
// PLY Property //
//////////////////

PlyProperty::PlyProperty(std::istream &is) : isList(false) {
    parse_internal(is);
}

void PlyProperty::parse_internal(std::istream &is) {
    string type;
    is >> type;
    if (type == "list") {
        string countType;
        is >> countType >> type;
        listType = property_type_from_string(countType);
        isList = true;
    }
    propertyType = property_type_from_string(type);
    is >> name;
}

/////////////////
// PLY Element //
/////////////////

PlyElement::PlyElement(std::istream &is) {
    parse_internal(is);
}

void PlyElement::parse_internal(std::istream &is) {
    is >> name >> size;
}

//////////////
// PLY File //
//////////////

PlyFile::PlyFile(std::istream &is) {
    if (!parse_header(is)) {
        throw std::runtime_error("file is not ply or encountered junk in header");
    }
}

bool PlyFile::parse_header(std::istream &is) {
    std::string line;
    bool gotMagic = false;
    while (std::getline(is, line)) {
        std::istringstream ls(line);
        std::string token;
        ls >> token;
        if (token == "ply" || token == "PLY" || token == "") {
            gotMagic = true;
            continue;
        } else if (token == "comment") read_header_text(line, ls, comments, 8);
        else if (token == "format") read_header_format(ls);
        else if (token == "element") read_header_element(ls);
        else if (token == "property") read_header_property(ls);
        else if (token == "obj_info") read_header_text(line, ls, objInfo, 9);
        else if (token == "end_header") break;
        else return false;
    }
    return gotMagic;
}

void PlyFile::read_header_text(std::string line, std::istream &is, std::vector<std::string> &place, int erase) {
    place.push_back((erase > 0) ? line.erase(0, erase) : line);
}

void PlyFile::read_header_format(std::istream &is) {
    std::string s;
    (is >> s);
    if (s == "binary_little_endian") isBinary = true;
    else if (s == "binary_big_endian") isBinary = isBigEndian = true;
}

void PlyFile::read_header_element(std::istream &is) {
    get_elements().emplace_back(is);
}

void PlyFile::read_header_property(std::istream &is) {
    get_elements().back().properties.emplace_back(is);
}

size_t PlyFile::skip_property_binary(const PlyProperty &property, std::istream &is) {
    static std::vector<char> skip(PropertyTable[property.propertyType].stride);
    if (property.isList) {
        size_t listSize = 0;
        size_t dummyCount = 0;
        read_property_binary(property.listType, &listSize, dummyCount, is);
        for (size_t i = 0; i < listSize; ++i) is.read(skip.data(), PropertyTable[property.propertyType].stride);
        return listSize;
    } else {
        is.read(skip.data(), PropertyTable[property.propertyType].stride);
        return 0;
    }
}

void PlyFile::skip_property_ascii(const PlyProperty &property, std::istream &is) {
    std::string skip;
    if (property.isList) {
        int listSize;
        is >> listSize;
        for (int i = 0; i < listSize; ++i) is >> skip;
    } else is >> skip;
}

void PlyFile::read_property_binary(PlyProperty::Type t, void *dest, size_t &destOffset, std::istream &is) {
    static std::vector<char> src(PropertyTable[t].stride);
    is.read(src.data(), PropertyTable[t].stride);

    switch (t) {
        case PlyProperty::Type::INT8:
            ply_cast<int8_t>(dest, src.data(), isBigEndian);
            break;
        case PlyProperty::Type::UINT8:
            ply_cast<uint8_t>(dest, src.data(), isBigEndian);
            break;
        case PlyProperty::Type::INT16:
            ply_cast<int16_t>(dest, src.data(), isBigEndian);
            break;
        case PlyProperty::Type::UINT16:
            ply_cast<uint16_t>(dest, src.data(), isBigEndian);
            break;
        case PlyProperty::Type::INT32:
            ply_cast<int32_t>(dest, src.data(), isBigEndian);
            break;
        case PlyProperty::Type::UINT32:
            ply_cast<uint32_t>(dest, src.data(), isBigEndian);
            break;
        case PlyProperty::Type::FLOAT32:
            ply_cast_float<float>(dest, src.data(), isBigEndian);
            break;
        case PlyProperty::Type::FLOAT64:
            ply_cast_double<double>(dest, src.data(), isBigEndian);
            break;
        case PlyProperty::Type::INVALID:
            throw std::invalid_argument("invalid ply property");
    }
    destOffset += PropertyTable[t].stride;
}

void PlyFile::read_property_ascii(PlyProperty::Type t, void *dest, size_t &destOffset, std::istream &is) {
    switch (t) {
        case PlyProperty::Type::INT8:
            *((int8_t *) dest) = ply_read_ascii<int32_t>(is);
            break;
        case PlyProperty::Type::UINT8:
            *((uint8_t *) dest) = ply_read_ascii<uint32_t>(is);
            break;
        case PlyProperty::Type::INT16:
            ply_cast_ascii<int16_t>(dest, is);
            break;
        case PlyProperty::Type::UINT16:
            ply_cast_ascii<uint16_t>(dest, is);
            break;
        case PlyProperty::Type::INT32:
            ply_cast_ascii<int32_t>(dest, is);
            break;
        case PlyProperty::Type::UINT32:
            ply_cast_ascii<uint32_t>(dest, is);
            break;
        case PlyProperty::Type::FLOAT32:
            ply_cast_ascii<float>(dest, is);
            break;
        case PlyProperty::Type::FLOAT64:
            ply_cast_ascii<double>(dest, is);
            break;
        case PlyProperty::Type::INVALID:
            throw std::invalid_argument("invalid ply property");
    }
    destOffset += PropertyTable[t].stride;
}

void PlyFile::read(std::istream &is) {
    read_internal(is);
}

void PlyFile::read_internal(std::istream &is) {
    std::function<void(PlyProperty::Type t, void *dest, size_t &destOffset, std::istream &is)> read;
    std::function<void(const PlyProperty &property, std::istream &is)> skip;
    if (isBinary) {
        read = [&](PlyProperty::Type t, void *dest, size_t &destOffset, std::istream &is) {
            read_property_binary(t, dest, destOffset, is);
        };
        skip = [&](const PlyProperty &property, std::istream &is) { skip_property_binary(property, is); };
    } else {
        read = [&](PlyProperty::Type t, void *dest, size_t &destOffset, std::istream &is) {
            read_property_ascii(t, dest, destOffset, is);
        };
        skip = [&](const PlyProperty &property, std::istream &is) { skip_property_ascii(property, is); };
    }

    for (auto &element : get_elements()) {
        if (std::find(requestedElements.begin(), requestedElements.end(), element.name) != requestedElements.end()) {
            for (size_t count = 0; count < element.size; ++count) {
                for (auto &property : element.properties) {
                    if (auto &cursor = userDataTable[make_key(element.name, property.name)]) {
                        if (property.isList) {
                            size_t listSize = 0;
                            size_t dummyCount = 0;
                            read(property.listType, &listSize, dummyCount, is);
                            if (cursor->realloc == false) {
                                cursor->realloc = true;
                                resize_vector(property.propertyType, cursor->vector, listSize * element.size,
                                              cursor->data);
                            }
                            for (size_t i = 0; i < listSize; ++i) {
                                read(property.propertyType, (cursor->data + cursor->offset), cursor->offset, is);
                            }
                        } else {
                            read(property.propertyType, (cursor->data + cursor->offset), cursor->offset, is);
                        }
                    } else {
                        skip(property, is);
                    }
                }
            }
        } else continue;
    }
}

/////////////////
// PLY Wrapper //
/////////////////
void PlyFile::read_ply_file(const std::string &filename, ply_reader::PointCloud &poClo) {
    try {
        std::ifstream ss(filename, std::ios::binary);

        PlyFile file(ss);

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
        poClo.num_points_ = vertexCount;

        // Add to vertex list
        for (int i = 0; i < verts.size(); i += 3) {
            Eigen::Vector3f add_pt(verts.data()[i], verts.data()[i + 1], verts.data()[i + 2]);
            poClo.points_.push_back(add_pt);
        }
        // Add to normals list
        for (int i = 0; i < norms.size(); i += 3) {
            Eigen::Vector3f add_norm(norms.data()[i], norms.data()[i + 1], norms.data()[i + 2]);
            poClo.normals_.push_back(add_norm);
        }
        // Add to colors list
        if (colors.size() > 0) {
            for (int i = 0; i < colors.size(); i += 3) {
                Eigen::Vector3f add_col((float) colors.data()[i] / 255.0, (float) colors.data()[i + 1] / 255.0,
                                        (float) colors.data()[i + 2] / 255.0);
                poClo.colors_.push_back(add_col);
            }
        } else {
            for (int i = 0; i < verts.size(); i += 3) {
                Eigen::Vector3f add_col(0.0f, 0.0f, 0.0f);
                poClo.colors_.push_back(add_col);
            }
        }

    } catch (const std::exception &e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }
}