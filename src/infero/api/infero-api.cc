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
#include <vector>
#include <memory>

#include "infero/api/infero.h"
#include "infero/models/InferenceModel.h"

#include "eckit/config/YAMLConfiguration.h"
#include "eckit/runtime/Main.h"
#include "eckit/exception/Exceptions.h"

#ifdef HAVE_MPI
  #include "eckit/io/SharedBuffer.h"
  #include "eckit/mpi/Comm.h"
#endif

using namespace eckit;
using namespace std;
using namespace infero;

using eckit::linalg::TensorDouble;
using eckit::linalg::TensorFloat;

//---------------------------------------------------------------------------------------------------

/* Error handling */

static std::string g_current_error_str;
static infero_failure_handler_t g_failure_handler = nullptr;
static void* g_failure_handler_context = nullptr;

/** Returns the error string */
const char* infero_error_string(int err) {
    switch (err) {
    case INFERO_SUCCESS:
        return "Success";
    case INFERO_ERROR_GENERAL_EXCEPTION:
    case INFERO_ERROR_UNKNOWN_EXCEPTION:
        return g_current_error_str.c_str();
    default:
        return "<unknown>";
    };
}

int innerWrapFn(std::function<int()> f) {
    return f();
}

int innerWrapFn(std::function<void()> f) {
    f();
    return INFERO_SUCCESS;
}

/** Wraps API functions and properly set errors to be reported
 * to the C interface */
template <typename FN>
int wrapApiFunction(FN f) {

    try {
        return innerWrapFn(f);
    } catch (Exception& e) {
        Log::error() << "Caught exception on C-C++ API boundary: " << e.what() << std::endl;
        g_current_error_str = e.what();
        if (g_failure_handler) {
            g_failure_handler(g_failure_handler_context, INFERO_ERROR_GENERAL_EXCEPTION);
        }
        return INFERO_ERROR_GENERAL_EXCEPTION;
    } catch (std::exception& e) {
        Log::error() << "Caught exception on C-C++ API boundary: " << e.what() << std::endl;
        g_current_error_str = e.what();
        if (g_failure_handler) {
            g_failure_handler(g_failure_handler_context, INFERO_ERROR_GENERAL_EXCEPTION);
        }
        return INFERO_ERROR_GENERAL_EXCEPTION;
    } catch (...) {
        Log::error() << "Caught unknown on C-C++ API boundary" << std::endl;
        g_current_error_str = "Unrecognised and unknown exception";
        if (g_failure_handler) {
            g_failure_handler(g_failure_handler_context, INFERO_ERROR_UNKNOWN_EXCEPTION);
        }
        return INFERO_ERROR_UNKNOWN_EXCEPTION;
    }

    ASSERT(false);
}

int infero_set_failure_handler(infero_failure_handler_t handler, void* context) {
    return wrapApiFunction([handler, context] {
        g_failure_handler = handler;
        g_failure_handler_context = context;
        eckit::Log::info() << "Infero setting failure handler fn." << std::endl;
    });
}

// ----------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

// tensors to be used for inference input/output
struct infero_tensors_t {
    infero_tensors_t() {}
    ~infero_tensors_t() noexcept(false) {}
    std::vector<TensorFloat*> tensors;
    std::vector<char*> input_names;
};

// model handle
struct infero_handle_t {
    infero_handle_t(InferenceModel* mod) : impl_(mod) {}
    ~infero_handle_t() noexcept(false) {}
    std::unique_ptr<InferenceModel> impl_;
};

int infero_initialise(int argc, char** argv){
    return wrapApiFunction([argc, argv]{
        eckit::Main::initialise(argc, argv);

        static bool initialised = false;

        if (initialised) {
            Log::warning() << "Initialising Infero library twice" 
                           << std::endl;
        }

        if (!initialised) {
            eckit::Main::initialise(1, const_cast<char**>(argv));
            initialised = true;
        }

    });
}

int infero_create_handle_from_yaml_str(const char str[], infero_handle_t** h) {
    return wrapApiFunction([str, h]{
        std::string str_(str);
        eckit::YAMLConfiguration cfg(str_);
        *h = new infero_handle_t(InferenceModel::create(cfg.getString("type"), cfg));
        ASSERT((*h)->impl_);

        (*h)->impl_->open();
    });
}

int infero_create_handle_from_yaml_file(const char path[], infero_handle_t** h) {
    return wrapApiFunction([path, h]{
#ifdef HAVE_MPI
        eckit::SharedBuffer buff = eckit::mpi::comm().broadcastFile(path, 0);
        eckit::YAMLConfiguration cfg(buff);
#else
        eckit::YAMLConfiguration cfg(path);
#endif    

        *h = new infero_handle_t(InferenceModel::create(cfg.getString("type"), cfg));
        ASSERT((*h)->impl_);

        (*h)->impl_->open();
    });
}

int infero_open_handle(infero_handle_t* h) {
    return wrapApiFunction([h]{
        h->impl_->open();
    });
}


int infero_close_handle(infero_handle_t* h) {
    return wrapApiFunction([h]{
        h->impl_->close();
    });
}


int infero_delete_handle(infero_handle_t* h) {
    return wrapApiFunction([h]{
        delete h;
    });
}


// run a ML engine for inference
int infero_inference_double(infero_handle_t* h,
                            int rank1, 
                            const double data1[], 
                            const int shape1[], 
                            int rank2,
                            double data2[], 
                            const int shape2[]) {

    return wrapApiFunction([]{
        std::cout << "infero_inference_double() - NOTIMP" << std::endl;
        NOTIMP;
    });
   
}

// run a ML engine for inference
int infero_inference_double_ctensor(infero_handle_t* h, 
                                    int rank1, 
                                    const double data1[], 
                                    const int shape1[], 
                                    int rank2,
                                    double data2[], 
                                    const int shape2[]) {    
    return wrapApiFunction([]{
        std::cout << "infero_inference_double_ctensor() "
                  << "- used for c-style input tensors - NOTIMP" << std::endl;   
        NOTIMP;   
    });

}


// run a ML engine for inference
int infero_inference_float(infero_handle_t* h, 
                           int rank1, 
                           const float data1[], 
                           const int shape1[], 
                           int rank2,
                           float data2[], 
                           const int shape2[]) {

    return wrapApiFunction([h, rank1, data1, shape1, rank2, data2, shape2]{
        ASSERT(h);

        std::cout << "infero_inference_float()" << std::endl;

        std::vector<size_t> shape1_vec(shape1,shape1+rank1);
        std::vector<size_t> shape2_vec(shape2,shape2+rank2); 
        TensorFloat* tIn(new TensorFloat(data1, shape1_vec, true));
        TensorFloat* tOut(new TensorFloat(data2, shape2_vec, true));

        h->impl_->infer(*tIn, *tOut);

        delete tIn;
        delete tOut;
   
   });
}

// run a ML engine for inference
int infero_inference_float_ctensor(infero_handle_t* h, 
                                   int rank1, 
                                   const float data1[], 
                                   const int shape1[], 
                                   int rank2,
                                   float data2[], 
                                   const int shape2[]) {

    return wrapApiFunction([h, rank1, data1, shape1, rank2, data2, shape2]{                                   

        ASSERT(h);

        std::cout << "infero_inference_float_ctensor() - used for c-style input tensors" << std::endl;

        std::vector<size_t> shape1_vec(shape1,shape1+rank1);
        std::vector<size_t> shape2_vec(shape2,shape2+rank2);
        TensorFloat* tIn(new TensorFloat(data1, shape1_vec, false));
        TensorFloat* tOut(new TensorFloat(data2, shape2_vec, false));

        h->impl_->infer(*tIn, *tOut);

        delete tIn;
        delete tOut;

    });
}



// run a ML engine for inference
int infero_inference_float_mimo(infero_handle_t* h,
                                int nInputs,
                                const char** iNames, 
                                const int* iRanks, 
                                const int** iShape, 
                                const float** iData,
                                int nOutputs,
                                const char** oNames, 
                                const int* oRanks, 
                                const int** oShape, 
                                float** oData){

    return wrapApiFunction([h,
                            nInputs, 
                            iNames, 
                            iRanks, 
                            iShape, 
                            iData, 
                            nOutputs, 
                            oNames, 
                            oRanks, 
                            oShape, 
                            oData]{
        ASSERT(h);

        std::cout << "infero_inference_float_mimo()" << std::endl;

        // loop over INPUT tensors
        ASSERT(nInputs >= 1);
        std::vector<TensorFloat*> inputData(static_cast<size_t>(nInputs));
        std::vector<const char*>  inputNames(static_cast<size_t>(nInputs));
        for (size_t i=0; i<static_cast<size_t>(nInputs); i++){

            // rank
            size_t rank = static_cast<size_t>(*(iRanks+i));
            ASSERT(rank >= 1);

            // shape
            std::vector<size_t> shape_(rank);
            for (size_t rr=0; rr<rank; rr++){
                shape_[rr] = static_cast<size_t>(*(*(iShape+i)+rr));
            }

            // name and data
            inputNames[i] = *(iNames+i);
            inputData[i] = new TensorFloat(*(iData+i), shape_, true);
        }

        // loop over OUTPUT tensors
        ASSERT(nOutputs >= 1);
        std::vector<TensorFloat*> outputData(static_cast<size_t>(nOutputs));
        std::vector<const char*>  outputNames(static_cast<size_t>(nOutputs));
        for (size_t i=0; i<static_cast<size_t>(nOutputs); i++){

            // rank
            size_t rank = static_cast<size_t>(*(oRanks+i));
            ASSERT(rank >= 1);

            // shape
            std::vector<size_t> shape_(rank);
            for (size_t rr=0; rr<rank; rr++){
                shape_[rr] = static_cast<size_t>(*(*(oShape+i)+rr));
            }

            // name and data
            outputNames[i] = *(oNames+i);
            outputData[i] = new TensorFloat(*(oData+i), shape_, true);
        }


        // mimo inference
        h->impl_->infer_mimo(inputData, inputNames, outputData, outputNames);

        // delete memory for input tensors
        for (size_t i=0; i<static_cast<size_t>(nInputs); i++){
            delete inputData[i];
        }

        // delete memory for output tensors
        for (size_t i=0; i<static_cast<size_t>(nOutputs); i++){
            delete outputData[i];
        }

    });
}


// run a ML engine for inference
int infero_inference_float_mimo_ctensor(infero_handle_t* h,
                                        int nInputs,
                                        const char** iNames,
                                        const int* iRanks, 
                                        const int** iShape, 
                                        const float** iData,
                                        int nOutputs,
                                        const char** oNames, 
                                        const int* oRanks, 
                                        const int** oShape, 
                                        float** oData) {

    return wrapApiFunction([h,
                            nInputs, 
                            iNames, 
                            iRanks, 
                            iShape, 
                            iData, 
                            nOutputs, 
                            oNames, 
                            oRanks, 
                            oShape, 
                            oData]{
        ASSERT(h);

        std::cout << "infero_inference_float_mimo()" << std::endl;

        // loop over INPUT tensors
        ASSERT(nInputs >= 1);
        std::vector<TensorFloat*> inputData(static_cast<size_t>(nInputs));
        std::vector<const char*>  inputNames(static_cast<size_t>(nInputs));
        for (size_t i=0; i<static_cast<size_t>(nInputs); i++){

            // rank
            size_t rank = static_cast<size_t>(*(iRanks+i));
            ASSERT(rank >= 1);

            // shape
            std::vector<size_t> shape_(rank);
            for (size_t rr=0; rr<rank; rr++){
                shape_[rr] = static_cast<size_t>(*(*(iShape+i)+rr));
            }

            // name and data
            inputNames[i] = *(iNames+i);
            inputData[i] = new TensorFloat(*(iData+i), shape_, false);
        }

        // loop over OUTPUT tensors
        ASSERT(nOutputs >= 1);
        std::vector<TensorFloat*> outputData(static_cast<size_t>(nOutputs));
        std::vector<const char*>  outputNames(static_cast<size_t>(nOutputs));
        for (size_t i=0; i<static_cast<size_t>(nOutputs); i++){

            // rank
            size_t rank = static_cast<size_t>(*(oRanks+i));
            ASSERT(rank >= 1);

            // shape
            std::vector<size_t> shape_(rank);
            for (size_t rr=0; rr<rank; rr++){
                shape_[rr] = static_cast<size_t>(*(*(oShape+i)+rr));
            }

            // name and data
            outputNames[i] = *(oNames+i);
            outputData[i] = new TensorFloat(*(oData+i), shape_, false);
        }


        // mimo inference
        h->impl_->infer_mimo(inputData, inputNames, outputData, outputNames);

        // delete memory for input tensors
        for (size_t i=0; i<static_cast<size_t>(nInputs); i++){
            delete inputData[i];
        }

        // delete memory for output tensors
        for (size_t i=0; i<static_cast<size_t>(nOutputs); i++){
            delete outputData[i];
        }

    });
}

int infero_finalise(){    
    return wrapApiFunction([]{
        // nothing to do here..
   });    
}

#ifdef __cplusplus
}
#endif
