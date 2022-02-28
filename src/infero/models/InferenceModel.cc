/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include <vector>
#include <string>

#include "eckit/exception/Exceptions.h"
#include "eckit/config/LocalConfiguration.h"

#ifdef HAVE_MPI
#include "eckit/mpi/Comm.h"
#endif

#include "infero/models/InferenceModel.h"

#ifdef HAVE_ONNX
#include "infero/models/InferenceModelONNX.h"
#endif

#ifdef HAVE_TF_C
#include "infero/models/InferenceModelTFC.h"
#endif

#ifdef HAVE_TFLITE
#include "infero/models/InferenceModelTFlite.h"
#endif

#ifdef HAVE_TENSORRT
#include "infero/models/InferenceModelTRT.h"
#endif

using namespace eckit;

namespace infero {

InferenceModel::InferenceModel(const eckit::Configuration& conf) :
    modelBuffer_{size_t(0)} {
}

InferenceModel::~InferenceModel() {

    print_statistics();

    if(isOpen_){
        close();
    }
}

void InferenceModel::open()  {

    // soft check: multiple open() allowed
    if (isOpen_){
        Log::info() << "INFO: Inference model already open.. " << std::endl;
    } else {
        isOpen_ = true;
    }
}

void InferenceModel::infer(linalg::TensorFloat& tIn, linalg::TensorFloat& tOut, std::string input_name, std::string output_name)
{

    // Input Tensor re-ordering as needed
    eckit::Timing t_start(statistics_.timer_);
    eckit::linalg::TensorFloat input_tensor;

    if (tIn.isRight()) {
        Log::info() << "Input Tensor has right-layout, but left-layout is needed. "
                    << "Transforming to left.." << std::endl;
        input_tensor = tIn.transformRightToLeftLayout();
    } else {

        // TODO: this still makes a copy (for now)
        input_tensor = tIn;
    }
    statistics_.iTensorLayoutTiming_ += eckit::Timing{statistics_.timer_} - t_start;

    // do the actual inference..
    eckit::Timing start_infer(statistics_.timer_);
    infer_impl(input_tensor, tOut, input_name, output_name);
    statistics_.inferenceTiming_ += eckit::Timing{statistics_.timer_} - start_infer;


}

void InferenceModel::infer_impl(linalg::TensorFloat& tIn, linalg::TensorFloat& tOut, std::string input_name, std::string output_name)
{
    NOTIMP;
}

// inference for models with multiple inputs and outputs
void InferenceModel::infer_mimo(std::vector<eckit::linalg::TensorFloat*> &tIn, std::vector<const char*> &input_names,
                                std::vector<eckit::linalg::TensorFloat*> &tOut, std::vector<const char*> &output_names)
{
    // Take copy of the input tensors
    std::vector<eckit::linalg::TensorFloat*> inputTensors(tIn.begin(), tIn.end());

    // For each tensor that needs re-ordering, do it into a copy
    std::vector<std::unique_ptr<eckit::linalg::TensorFloat>> temporaryCopies;

    eckit::Timing t_start(statistics_.timer_);
    for (int i = 0; i < inputTensors.size(); ++i) {
        if (inputTensors[i]->isRight()) {

            Log::info() << i << "-th Input Tensor has right-layout, "
                        << "but left-layout is needed. Transforming to left.." << std::endl;

            temporaryCopies.emplace_back(new eckit::linalg::TensorFloat(inputTensors[i]->transformRightToLeftLayout()));
            inputTensors[i] = temporaryCopies.back().get();
        }
    }
    statistics_.iTensorLayoutTiming_ += eckit::Timing{statistics_.timer_} - t_start;

    // do the actual inference..
    eckit::Timing start_infer(statistics_.timer_);
    Log::info() << "doing inference.." << std::endl;
    infer_mimo_impl(inputTensors, input_names, tOut, output_names);
    statistics_.inferenceTiming_ += eckit::Timing{statistics_.timer_} - start_infer;

}

// inference for models with multiple inputs and outputs
void InferenceModel::infer_mimo_impl(std::vector<eckit::linalg::TensorFloat*>& tIn, std::vector<const char*>& input_names,
                                     std::vector<eckit::linalg::TensorFloat*>& tOut, std::vector<const char*>& output_names)
{
    NOTIMP;
}

void InferenceModel::close() {

    // soft check: multiple close() allowed
    if (!isOpen_){
        Log::info() << "INFO: Inference model already closed.. " << std::endl;
    } else {
        isOpen_ = false;
    }
}

void InferenceModel::broadcast_model(const std::string path) {

#ifdef HAVE_MPI
    modelBuffer_ = eckit::mpi::comm().broadcastFile(path, 0);
#endif
}

VecPairStr InferenceModel::RequiredEnvVariables_()
{
    // by default, no env variables required
    return VecPairStr();
}

void InferenceModel::readEnvConfig_()
{
    envConfig_.reset(new eckit::LocalConfiguration);

    // read environment variables
    for (auto& var: RequiredEnvVariables_()){

        const char* value_ = getenv(var.first.c_str());
        if (value_){
            Log::info() << var << ": " << value_ << std::endl;
            envConfig_->set(var.first, value_);
        } else {
            Log::info() << var << " not found, using default: "
                        << var.second << std::endl;
            envConfig_->set(var.first, var.second);
        }
    }
}


void InferenceModel::print_statistics()
{
    Log::info() << statistics() << std::endl;
}


//-------------------------------------------------------------------------------------------------


InferenceModelFactory::InferenceModelFactory() {}

InferenceModelFactory::~InferenceModelFactory() {}

InferenceModelFactory& InferenceModelFactory::instance() {
    static InferenceModelFactory theinstance;
    return theinstance;
}

void InferenceModelFactory::enregister(const std::string& name,
                                       const InferenceModelBuilderBase& builder) {
    std::lock_guard<std::mutex> lock(mutex_);
    ASSERT(builders_.find(name) == builders_.end());
    builders_.emplace(std::make_pair(name, std::ref(builder)));
}

void InferenceModelFactory::deregister(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = builders_.find(name);
    ASSERT(it != builders_.end());
    builders_.erase(it);
}

InferenceModel* InferenceModelFactory::build(const std::string& name,
                                             const eckit::Configuration& config) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = builders_.find(name);
    if (it == builders_.end()) {
        std::ostringstream ss;
        ss << "Builder not found for backend " << name;
        throw SeriousBug(ss.str(), Here());
    }

    return it->second.get().make(config);
}

InferenceModelBuilderBase::InferenceModelBuilderBase(const std::string& name) :
    name_(name) {
    InferenceModelFactory::instance().enregister(name, *this);
}

InferenceModelBuilderBase::~InferenceModelBuilderBase() {
    InferenceModelFactory::instance().deregister(name_);
}






}  // namespace infero
