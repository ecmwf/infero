#pragma once

#include <cstring>
#include <memory>
#include <ostream>
#include <string>
#include <fstream>

#include "eckit/log/Log.h"
#include "eckit/exception/Exceptions.h"

using eckit::Log;

namespace infero {

class InferoBuffer
{
public:

    InferoBuffer(char* as_void_ptr, size_t dataSize);

    ~InferoBuffer();

    static InferoBuffer* from_path(const std::string path);

    void* as_void_ptr() const {return reinterpret_cast<void*>(data_);}

    char* as_char_ptr() const {return data_;}

    size_t size() const {return dataSize_;}

private:

    static InferoBuffer* read_from_disk(const std::string path);

private:

    // data (copy)
    char* data_;

    // data size
    size_t dataSize_;

};

}  // namespace infero
