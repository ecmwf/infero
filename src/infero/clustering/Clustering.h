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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "infero/MLTensor.h"


using namespace infero;

// a cluster point
struct ClusterPoint {

    float x;
    float y;
    int cid;

    ClusterPoint() {}

    ClusterPoint(float x, float y, int cid) : x(x), y(y), cid(cid) {}
};


// cluster types
typedef std::vector<ClusterPoint> ClusterPoints;
typedef std::map<int, ClusterPoints> ClusterMap;
typedef std::pair<int, ClusterPoints> ClusterPair;


// generic clustering algorithm
class Clustering {

public:
    Clustering();

    virtual ~Clustering();

    // run the clustering
    virtual int run(std::unique_ptr<MLTensor>& prediction) = 0;

    // summary of clustering
    virtual void print_summary();

    // write JSON
    virtual int write_json(std::string filename);

    static std::unique_ptr<Clustering> create(std::string choice);

public:
    // cluster centers
    std::vector<ClusterPoint> cluster_centers;

protected:
    // labelled points
    //   - x1,y1,cid1
    //   - x2,y2,cid1
    //   - .....
    //   - xn,yn,cidm
    std::vector<ClusterPoint> points;

    // cluster centers
    virtual void calculate_cluster_centers();
};
