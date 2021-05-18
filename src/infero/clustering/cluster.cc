/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include "eckit/option/CmdArgs.h"
#include "eckit/option/SimpleOption.h"
#include "eckit/runtime/Main.h"
#include "eckit/log/Log.h"

#include "infero/MLTensor.h"
#include "infero/clustering/Clustering.h"

using namespace eckit;
using namespace eckit::option;


using namespace infero;


void usage(const std::string&){

    Log::info() << std::endl
                << "-------------------------------" << std::endl
                << "Machine Learning clustering tool" << std::endl
                << "-------------------------------" << std::endl
                << std::endl
                << "Runs a clustering algorithm on a ML prediction"
                << std::endl;
}


int main(int argc, char** argv) {

    Main::initialise(argc, argv);
    std::vector<Option*> options;

    options.push_back(new SimpleOption<std::string>("input", "Path to input file"));
    options.push_back(new SimpleOption<std::string>("clustering", "Clustering [dbscan, ...]"));

    CmdArgs args(&usage, options, 0, 0, true);

    // input path
    std::string input_path = args.getString("input","data.npy");
    std::string clustering = args.getString("clustering","dbscan");
    std::string output_path = args.getString("output","clusters.json");

    // input data
    std::unique_ptr<infero::MLTensor> inputT = infero::MLTensor::from_file(input_path);

    // run clustering
    std::unique_ptr<Clustering> cluster = Clustering::create(clustering);

    int err = cluster->run(inputT);
    if(err){
        Log::error() << "Clustering Failed!" << std::endl;
        return EXIT_FAILURE;
    } else {
        cluster->print_summary();

        // write to JSON
        cluster->write_json(output_path);
    }

    return EXIT_SUCCESS;
}
