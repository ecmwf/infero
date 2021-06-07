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

#include "eckit/parser/YAMLParser.h"

#include "infero/MLTensor.h"


using namespace std;
using namespace eckit::linalg;


namespace infero {

// Minimal interface for a runtime engine
class MLEngine {

public:

    // a ML Engine configuration
    class Configuration {

    public:

        Configuration(const std::string& path, const std::string& type) :
            path_(path),
            type_(type){}

        static Configuration from_yaml(std::string yml_str){
            eckit::Value val = eckit::YAMLParser::decodeString(yml_str);
            std::string path_ = val["path"].as<std::string>();
            std::string type_ = val["type"].as<std::string>();

            return Configuration(path_, type_);
        }

        const std::string& path() const { return path_;}
        const std::string& type() const { return type_;}

    private:

        std::string path_;
        std::string type_;
    };

public:

    MLEngine(std::string model_filename) : mModelFilename(model_filename) {}

    virtual ~MLEngine();

    // run the inference
    virtual std::unique_ptr<infero::MLTensor> infer(std::unique_ptr<infero::MLTensor>& input_sample) = 0;

    // create concrete engines
    static std::unique_ptr<MLEngine> create(std::string choice, std::string model_path);

    static int create_handle(std::string choice, std::string model_path);

    static void close_handle(int handle_id);

    static std::unique_ptr<MLEngine>& get_model(int handle_id);

    friend std::ostream& operator<<(std::ostream& os, MLEngine& obj) {
        obj.print(os);
        return os;
    }


protected:
    virtual void print(std::ostream& os) const {}

protected:

    std::string mModelFilename;

    static int gid_;
    static std::map<int, std::unique_ptr<MLEngine>> map_;
};


}  // namespace infero
