/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include <iostream>

#include <algorithm>

#include "infero/models/InferenceModelTorch.h"
#include "infero/infero_utils.h"

namespace infero {


static InferenceModelBuilder<InferenceModelTorch> torchBuilder;

eckit::LocalConfiguration InferenceModelTorch::defaultConfig() {
    static eckit::LocalConfiguration config;
    // empty defaults..
    return config;
}


InferenceModelTorch::InferenceModelTorch(const eckit::Configuration& conf) : 
    InferenceModel(conf, InferenceModelTorch::defaultConfig()){

    // read/bcast model by mpi (when possible)
    broadcast_model(modelPath());

    // Load the model    
    try {
        eckit::Log::info() << "Loading model from: " << modelPath() << std::endl;
        torch_module_ = torch::jit::load(modelPath());
        eckit::Log::info() << "Model loaded" << std::endl;
    } catch (const c10::Error& e) {
        eckit::Log::error() << "Error loading the model: " << modelPath() << " - " << e.what() << std::endl;
        throw eckit::SeriousBug("Error loading the model");
    }

}


std::string InferenceModelTorch::name() const {
    return "InferenceModelTorch";
}


void InferenceModelTorch::print(std::ostream& os) const {
    os << "InferenceModelTorch";
}

void InferenceModelTorch::infer_impl(eckit::linalg::TensorFloat& tIn, eckit::linalg::TensorFloat& tOut,
                                        std::string input_name, std::string output_name) {

    // Load the input tensor
    auto shape = tIn.shape();
    std::vector<int64_t> shape_int64(shape.size());
    std::copy(shape.begin(), shape.end(), shape_int64.begin());
    
    torch::IntArrayRef shape_ref(shape_int64.data(), shape_int64.size());
    torch::Tensor input = torch::from_blob(tIn.data(), shape_ref);

    // Pass input tensor to the model
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    // Run inference
    at::Tensor out_tensor;
    try {
        auto output = torch_module_.forward(inputs);
        out_tensor = output.toTensor();
    } catch (const c10::Error& e) {
        eckit::Log::error() << "Error running inference" << std::endl;
    }

    // --- Copy data from the output tensor ---
    at::IntArrayRef sizes = out_tensor.sizes();
    std::vector<int64_t> out_shape(sizes.begin(), sizes.end());
    if (tOut.layout() == eckit::linalg::TensorFloat::Layout::ColMajor) {

         // Torch uses Left (C) tensor layouts, so we need to convert
         eckit::linalg::TensorFloat tLeft(out_tensor.data_ptr<float>(),
                                          utils::convert_shape<int64_t, size_t>(out_shape),
                                          eckit::linalg::TensorFloat::Layout::RowMajor);
         eckit::linalg::TensorFloat tRight = tLeft.transformRowMajorToColMajor();
         tOut = tRight;
    } else {
         // Torch uses Left (C) tensor layouts, so we can copy straight into memory of tOut
         memcpy(tOut.data(), out_tensor.data_ptr<float>(), out_tensor.numel() * sizeof(float));
    }

}

void InferenceModelTorch::infer_mimo_impl(std::vector<eckit::linalg::TensorFloat*> &tIn, std::vector<const char*> &input_names,
                                            std::vector<eckit::linalg::TensorFloat*> &tOut, std::vector<const char*> &output_names) {

    eckit::Log::info() << "Running Torch inference (mimo).." << std::endl;

    // Pack input tensors into a map
    std::unordered_map<std::string, torch::Tensor> inputs;
    for (size_t i=0; i<tIn.size(); i++) {
        eckit::Log::debug() << "Input " << i << ": " << input_names[i] << std::endl;
        
        // Load the input tensor
        auto shape = tIn[i]->shape();
        std::vector<int64_t> shape_int64(shape.size());
        std::copy(shape.begin(), shape.end(), shape_int64.begin());
        
        torch::IntArrayRef shape_ref(shape_int64.data(), shape_int64.size());
        torch::Tensor input_tensor = torch::from_blob(tIn[i]->data(), shape_ref);

        inputs[input_names[i]] = input_tensor;
    }

    // Vector of inputs (as required by forward method)
    std::vector<torch::jit::IValue> inputs_vector;
    inputs_vector.push_back(inputs);

    // Run inference
    auto output = torch_module_.forward(inputs_vector).toGenericDict();

    // copy output
    size_t NOutputs = output_names.size();
    eckit::Timing t_start(statistics_.timer());
    for (size_t i=0; i<NOutputs; i++){

        auto output_tensor = output.at(output_names[i]).toTensor();
        float* data_ptr = output_tensor.data_ptr<float>();

        if (tOut[i]->layout() == eckit::linalg::TensorFloat::Layout::ColMajor) {

            Log::info() << i << "-th Output Tensor needs right-layout. " << "Transforming left to right.." << std::endl;

            // Torch uses Left (C) tensor layouts, so we need to convert
            TensorFloat tLeft(data_ptr, tOut[i]->shape(), eckit::linalg::TensorFloat::Layout::RowMajor);  // wrap data
            TensorFloat tRight = tLeft.transformRowMajorToColMajor();
            *tOut[i] = tRight;

        } else {

            // Torch uses Left (C) tensor layouts, so we can copy straight into memory of tOut
            memcpy(tOut[i]->data(), data_ptr, tOut[i]->size() * sizeof(float));
        }
    }
    statistics_.oTensorLayoutTiming_ += eckit::Timing{statistics_.timer()} - t_start;

}


} // namespace infero