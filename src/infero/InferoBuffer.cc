#include "InferoBuffer.h"

#include "eckit/mpi/Comm.h"

namespace infero {

InferoBuffer::InferoBuffer(char *data, size_t dataSize): data_(nullptr), dataSize_(dataSize){

    // takes ownership of data_
    data_ = new char[ dataSize_ ];
    memcpy(data_, data, dataSize);
}

InferoBuffer::~InferoBuffer(){
    delete [] data_;
}

InferoBuffer* InferoBuffer::from_path(const std::string path){

    InferoBuffer* _buffr = nullptr;

#ifdef HAVE_MPI

    Log::info() << "mpi size = " << eckit::mpi::comm().size()
                << ", rank = "   << eckit::mpi::comm().rank()
                << std::endl;

    size_t _buffr_size;
    char* _buffr_data = nullptr;

    // rank 0 reads data from disk
    if (eckit::mpi::comm().rank() == 0){
        _buffr = InferoBuffer::read_from_disk(path);
        _buffr_size = _buffr->size();
        _buffr_data = reinterpret_cast<char*>(_buffr->data());
        Log::info() << "Rank 0 has read file: " << path
                    << ", of size: " << _buffr_size
                    << std::endl;
    }

    // Rank 0 broadcasts size
    eckit::mpi::comm().broadcast(_buffr_size, 0);

    // All other ranks make space for data buffer
    if (eckit::mpi::comm().rank() != 0){
        _buffr_data = new char[_buffr_size];
    }

    // Rank 0 broadcasts buffer data
    eckit::mpi::comm().broadcast(_buffr_data, _buffr_data+_buffr_size, 0);

    // All other ranks build their buffer
    if (eckit::mpi::comm().rank() != 0){
        _buffr = new InferoBuffer(_buffr_data, _buffr_size);

        Log::info() << "Rank " << eckit::mpi::comm().rank()
                    << " has received buffer size " << _buffr_size
                    << std::endl;
    }

#else

    // rank 0 reads data from disk
    _buffr = InferoBuffer::read_from_disk(path);

#endif

    return _buffr;
}

InferoBuffer* InferoBuffer::read_from_disk(const std::string path){

    // read model from path
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size <= 0)
        throw eckit::FailedSystemCall("File " + path +
                                      " has size " + std::to_string(size),
                                      Here());

    char* buffer = new char[ static_cast<size_t>(size) ];
    if (file.read(buffer, size))
    {
        Log::info() << "File " + path + " read."
                    << " Size: " << std::to_string(size)
                    << std::endl;
    }

    InferoBuffer* Buffr = new InferoBuffer( buffer, static_cast<size_t>(size));

    // delete tmp buffer
    delete [] buffer;

    return Buffr;

}

} // infero namespace
