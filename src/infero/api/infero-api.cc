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
#include <string>
#include <map>
#include <any>
#include <algorithm>

#include "eckit/runtime/Main.h"
#include "eckit/config/YAMLConfiguration.h"
#include "eckit/exception/Exceptions.h"
#include "eckit/filesystem/LocalPathName.h"

#include "eckit/io/SharedBuffer.h"
#include "eckit/mpi/Comm.h"

#include "infero/api/infero.h"
#include "infero/models/InferenceModel.h"


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
static bool infero_initialised = false;


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

// model handle
struct infero_handle_t {
    infero_handle_t(InferenceModel* mod) : impl_(mod) {}
    ~infero_handle_t() noexcept(false) {}
    std::unique_ptr<InferenceModel> impl_;
};

int infero_initialise(int argc, char** argv){
    return wrapApiFunction([argc, argv]{

        if (!infero_initialised) {
            eckit::Main::initialise(argc, argv);
            infero_initialised = true;
        } else {
            throw eckit::UnexpectedState("Initialising Infero library twice!", Here());
        }

    });
}

int infero_create_handle_from_yaml_str(const char* str, infero_handle_t** h) {
    return wrapApiFunction([str, h]{
        std::string str_(str);
        eckit::YAMLConfiguration cfg(str_);
        *h = new infero_handle_t(InferenceModelFactory::instance().build(cfg.getString("type"), cfg));

        ASSERT(*h);
        ASSERT((*h)->impl_);        

    });
}

int infero_create_handle_from_yaml_file(const char* path, infero_handle_t** h) {
    return wrapApiFunction([path, h]{
        eckit::SharedBuffer buff = eckit::mpi::comm().broadcastFile(path, 0);
        eckit::YAMLConfiguration cfg(buff);
        *h = new infero_handle_t(InferenceModelFactory::instance().build(cfg.getString("type"), cfg));
        ASSERT(*h);
        ASSERT((*h)->impl_);

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
    return wrapApiFunction([&h]{
        if (h){
            delete h;
            h = nullptr;
        }
    });
}


// run a ML engine for inference
int infero_inference_float(infero_handle_t* h, 
                           int rank1, 
                           const float data1[], 
                           const int shape1[],
                           int layout1,
                           int rank2,
                           float data2[], 
                           const int shape2[],
                           int layout2) {

    return wrapApiFunction([h, rank1, data1, shape1, layout1, rank2, data2, shape2, layout2]{
        ASSERT(h);

        std::vector<size_t> shape1_vec(shape1,shape1+rank1);
        std::vector<size_t> shape2_vec(shape2,shape2+rank2); 
        TensorFloat* tIn(new TensorFloat(const_cast<float*>(data1), shape1_vec, static_cast<TensorFloat::Layout>(layout1)));
        TensorFloat* tOut(new TensorFloat(data2, shape2_vec, static_cast<TensorFloat::Layout>(layout2)));

        h->impl_->infer(*tIn, *tOut);

        delete tIn;
        delete tOut;
   
   });
}


// run a ML engine for inference
int infero_inference_double(infero_handle_t* h,
                            int rank1, 
                            const double data1[], 
                            const int shape1[], 
                            int layout1,
                            int rank2,
                            double data2[], 
                            const int shape2[],
                            int layout2) {

    return wrapApiFunction([]{
        std::cout << "infero_inference_double() - NOTIMP" << std::endl;
        NOTIMP;
    });
   
}


// run a ML engine for inference
int infero_inference_float_mimo(infero_handle_t* h,
                                int nInputs,
                                const char** iNames, 
                                const int* iRanks, 
                                const int** iShape, 
                                const float** iData,
                                int iLayout,
                                int nOutputs,
                                const char** oNames, 
                                const int* oRanks, 
                                const int** oShape, 
                                float** oData,
                                int oLayout ){

    return wrapApiFunction([h,
                            nInputs, 
                            iNames, 
                            iRanks, 
                            iShape, 
                            iData,
                            iLayout,
                            nOutputs, 
                            oNames, 
                            oRanks, 
                            oShape, 
                            oData,
                            oLayout]{
        ASSERT(h);

        std::cout << "infero_inference_float_mimo()" << std::endl;

        // loop over INPUT tensors
        ASSERT(nInputs >= 1);
        std::map<std::string,TensorFloat*> imap;
        for (size_t i=0; i<static_cast<size_t>(nInputs); i++){

            // rank
            size_t rank = static_cast<size_t>(*(iRanks+i));
            ASSERT(rank >= 1);

            // shape
            std::vector<size_t> shape_(rank);
            for (size_t rr=0; rr<rank; rr++){
                shape_[rr] = static_cast<size_t>(*(*(iShape+i)+rr));
            }
            imap.insert(make_pair(*(iNames+i), new TensorFloat(const_cast<float*>(*(iData+i)), shape_, static_cast<TensorFloat::Layout>(iLayout))));
        }

        // loop over OUTPUT tensors
        ASSERT(nOutputs >= 1);
        std::map<std::string,TensorFloat*> omap;
        for (size_t i=0; i<static_cast<size_t>(nOutputs); i++){

            // rank
            size_t rank = static_cast<size_t>(*(oRanks+i));
            ASSERT(rank >= 1);

            // shape
            std::vector<size_t> shape_(rank);
            for (size_t rr=0; rr<rank; rr++){
                shape_[rr] = static_cast<size_t>(*(*(oShape+i)+rr));
            }

            omap.insert(make_pair(*(oNames+i), new TensorFloat(*(oData+i), shape_, static_cast<TensorFloat::Layout>(oLayout))));
        }

        // mimo inference
        h->impl_->infer_mimo(imap, omap);

        std::for_each(imap.begin(),imap.end(),
                        [](auto& item){ 
                            delete item.second;
                            item.second=nullptr;
                        });

        std::for_each(omap.begin(),omap.end(),
                        [](auto& item){ 
                            delete item.second;
                            item.second=nullptr;
                        });

    });
}


// run a ML engine for inference
int infero_inference_double_mimo(infero_handle_t* h,
                                int nInputs,
                                const char** iNames, 
                                const int* iRanks, 
                                const int** iShape, 
                                const double** iData,
                                int iLayout,
                                int nOutputs,
                                const char** oNames, 
                                const int* oRanks, 
                                const int** oShape, 
                                double** oData,
                                int oLayout ){
    return wrapApiFunction([]{
        std::cout << "infero_inference_double_mimo() - NOTIMP" << std::endl;
        NOTIMP;
    });
}



int infero_inference_float_map(infero_handle_t* h, void* imap_any_ptr, void* omap_any_ptr){

    return wrapApiFunction([h, imap_any_ptr, omap_any_ptr]{

        ASSERT(h);
        ASSERT(imap_any_ptr);
        ASSERT(omap_any_ptr);

        std::map<std::string,std::any>* imap_any = static_cast<std::map<std::string,std::any>*>(imap_any_ptr);
        std::map<std::string, TensorFloat*> imap;
        for (const auto& item: *imap_any) {
            imap.insert(make_pair(item.first, static_cast<TensorFloat*>(std::any_cast<void*>(item.second)) ));
        }

        std::map<std::string,std::any>* omap_any = static_cast<std::map<std::string,std::any>*>(omap_any_ptr);
        std::map<std::string, TensorFloat*> omap;
        for (const auto& item: *omap_any) {
            omap.insert(make_pair(item.first, static_cast<TensorFloat*>(std::any_cast<void*>(item.second))  ));
        }        

        h->impl_->infer_mimo(imap, omap);

        });
}

int infero_inference_double_map(infero_handle_t* h, void* imap_any_ptr, void* omap_any_ptr){

    return wrapApiFunction([]{
        std::cout << "infero_inference_double_map() - NOTIMP" << std::endl;
        NOTIMP;
    });
}


int infero_print_statistics(infero_handle_t* h){
    return wrapApiFunction([h]{
        h->impl_->print_statistics();
    });
}


int infero_print_config(infero_handle_t* h){
    return wrapApiFunction([h]{
        h->impl_->print_config();
    });
}


int infero_finalise(){    
    return wrapApiFunction([]{

        if (!infero_initialised) {
            throw eckit::UnexpectedState("Infero library not initialised!", Here());
        } else {
            infero_initialised = false;
        }
   });    
}


#ifdef __cplusplus
}
#endif
