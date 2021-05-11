#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <memory>

class Prediction;

typedef std::unique_ptr<Prediction> PredictionPtr;

// ML prediction
class Prediction
{

public:

    Prediction(std::vector<float> data_tensor,
               std::vector<int64_t> shape);

    // save to disk as npy
    void to_numpy(std::string filename);

    // return the flat size
    int size_flat() const;

    // ptr to raw data
    float *data() const;

    int n_rows() const;

    int n_cols() const;

public:

    std::vector<float> data_tensor;

private:

    std::vector<int64_t> shape;
    std::vector<long unsigned int> shape_static;

};