#ifndef INPUT_DATA_H
#define INPUT_DATA_H

#include <vector>
#include <string>
#include <memory>


class InputData;

typedef std::unique_ptr<InputData> InputDataPtr;

class InputData
{

public:

    InputData(std::string filename);

    static InputDataPtr from_numpy(std::string filename);

    // get data
    float* get_data();

    // get size
    size_t get_size();

public:

    std::vector<size_t> shape;

private:

    size_t mSize;
    std::vector<float> mDataVector;

};

#endif
