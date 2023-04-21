/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <fstream>
#include <mutex>
#include <map>

#include "eckit/config/Configuration.h"
#include "eckit/config/LocalConfiguration.h"
#include "eckit/linalg/Tensor.h"
#include "eckit/log/Log.h"
#include "eckit/io/SharedBuffer.h"

#include "ModelStatistics.h"


using eckit::Log;

namespace infero {

using ModelParams_t = std::map<std::string,std::string>;


/// Interface for an inference model
class InferenceModel {

    using TensorMap = std::map<std::string, eckit::linalg::TensorFloat*>;

public:

    InferenceModel(const eckit::Configuration& conf);

    virtual ~InferenceModel();

    virtual std::string name() const;

    /// opens the engine
    virtual void open();

    /// run the inference
    virtual void infer(eckit::linalg::TensorFloat& tIn, eckit::linalg::TensorFloat& tOut,
                       std::string input_name = "", std::string output_name = "");

    /// MIMO (Multi Input Multi Output) inference 
    virtual void infer_mimo(TensorMap& iMap, TensorMap& oMap);

    /// closes the engine
    virtual void close();    

    void print_statistics();

    void print_config();

    ModelStatistics& statistics(){ return statistics_; }

protected:

    virtual void infer_mimo(std::vector<eckit::linalg::TensorFloat*> &tIn, std::vector<const char*> &input_names,
                        std::vector<eckit::linalg::TensorFloat*> &tOut, std::vector<const char*> &output_names);

    virtual void infer_impl(eckit::linalg::TensorFloat& tIn, eckit::linalg::TensorFloat& tOut,
                            std::string input_name = "", std::string output_name = "");

    virtual void infer_mimo_impl(std::vector<eckit::linalg::TensorFloat*> &tIn, std::vector<const char*> &input_names,
                                 std::vector<eckit::linalg::TensorFloat*> &tOut, std::vector<const char*> &output_names);



    /// print the model
    virtual void print(std::ostream& os) const = 0;

    friend std::ostream& operator<<(std::ostream& os, InferenceModel& obj) {
        obj.print(os);
        return os;
    }

    virtual void broadcast_model(const std::string path);

    /// default model Configuration
    virtual ModelParams_t defaultParams_();

    /// implementation-specific default configuration
    virtual ModelParams_t implDefaultParams_();

    /// Assemble the Configuration (defaults + user + env)
    void readConfig_(const eckit::Configuration& conf);

    /// env-variables configuration
    std::unique_ptr<eckit::LocalConfiguration> ModelConfig_;

    /// Inference model buffer
    eckit::SharedBuffer modelBuffer_;

    /// Stats
    ModelStatistics statistics_;

protected:            

    bool isOpen_;

};


//-------------------------------------------------------------------------------------------------

// fwd declaration
class InferenceModelBuilderBase;


// factory (registers/deregisters builders and calls "build")
class InferenceModelFactory {

public: // methods

    static InferenceModelFactory& instance();

    void enregister(const std::string& name, const InferenceModelBuilderBase& builder);
    void deregister(const std::string& name);

    InferenceModel* build(const std::string& name, const eckit::Configuration& config) const;

private: // methods

    // Only one instance can be built, inside instance()
    InferenceModelFactory();
    ~InferenceModelFactory();

private: // members

    mutable std::mutex mutex_;

    std::map<std::string, std::reference_wrapper<const InferenceModelBuilderBase>> builders_;
};


// base builder
class InferenceModelBuilderBase {
public: // methods

    // Only instantiate from subclasses
    InferenceModelBuilderBase(const std::string& name);
    virtual ~InferenceModelBuilderBase();

    virtual InferenceModel* make(const eckit::Configuration& config) const = 0;

public: // members
    std::string name_;
};


// a concrete builder for a specific InferenceModel type
template <typename T>
class InferenceModelBuilder : public InferenceModelBuilderBase {
public: // methods
    InferenceModelBuilder() : InferenceModelBuilderBase(T::type()) {}
    
    ~InferenceModelBuilder() override {}

    InferenceModel* make(const eckit::Configuration& config) const override {
        return new T(config);
    }
};













}  // namespace infero
