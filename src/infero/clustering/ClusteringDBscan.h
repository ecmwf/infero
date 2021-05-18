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

#include "DBSCAN/dbscan.h"

#include "infero/clustering/Clustering.h"


// min number of clusters
static const unsigned int DBSCAN_MIN_N_CLUSTERS = 1;

// dbscan point distance in cluster
static constexpr float DBSCAN_EPS = 10.0;

// min point val to be considered for clustering
static constexpr float DBSCAN_MIN_VAL = 0.6;

using namespace infero;


// runs a DBScan algorithm
class ClusteringDBscan: public Clustering
{

public:

    ClusteringDBscan();

    // run clustering
    virtual int run(std::unique_ptr<MLTensor>& prediction);


private:

    // read the prediction
    virtual std::vector<Point> readPrediction(std::unique_ptr<MLTensor>& prediction);


private:

    float min_threshold;

};
