/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include <fstream>
#include <sstream>

#include "eckit/exception/Exceptions.h"
#include "eckit/log/JSON.h"
#include "eckit/log/Log.h"

#include "infero/clustering/Clustering.h"
#include "infero/clustering/ClusteringDBscan.h"

using namespace eckit;


Clustering::Clustering() {
    cluster_centers.resize(0);
    points.resize(0);
}


Clustering::~Clustering() {}

void Clustering::calculate_cluster_centers() {

    // loop over cluster points
    // and fill a map < cluster ID, vec<points> >
    ClusterMap cluster;
    for (const auto& pt : points) {

        auto it = cluster.find(pt.cid);
        if (it != cluster.end()) {

            it->second.push_back(ClusterPoint(pt.x, pt.y, pt.cid));
        }
        else {

            ClusterPoints vv{ClusterPoint(pt.x, pt.y, pt.cid)};
            ClusterPair _new_clsr(pt.cid, vv);
            cluster.emplace(_new_clsr);
        }
    }


    // average point coords for each cluster ID
    float x, y;
    for (const auto& k : cluster) {

        x = y = 0;

        for (auto& p : k.second) {
            x += p.x;
            y += p.y;
        }

        if (k.second.size()) {
            x /= k.second.size();
            y /= k.second.size();
        }

        this->cluster_centers.push_back(ClusterPoint(x, y, k.first));
    }
}

// print clusters
void Clustering::print_summary() {

    Log::info() << "\n*** centers *** " << std::endl;
    for (const auto& clust_ctr : cluster_centers) {
        printf("%2d) x = %8.3f, y = %8.3f\n", clust_ctr.cid, clust_ctr.x, clust_ctr.y);
    }

    Log::info() << std::endl;
}


int Clustering::write_json(std::string filename) {
    std::stringstream s;
    JSON json_out(s, JSON::Formatting::indent(2));

    json_out.startObject();
    json_out.startList();
    for (const auto& c : cluster_centers) {
        json_out.startList();
        json_out << c.x << c.y;
        json_out.endList();
    }
    json_out.endList();
    json_out.endObject();

    std::ofstream fout(filename);

    if (fout) {

        Log::info() << "Writing to JSON file " << filename << std::endl;

        fout << s.str();
        return 0;
    }

    Log::error() << "Failed to open output file " << filename << std::endl;
    return -1;
}

Clustering* Clustering::create(std::string choice) {
    if (choice == "dbscan") {
        Log::info() << "creating ClusteringDBscan.. " << std::endl;
        return new ClusteringDBscan;
    }
    else {
        throw BadValue("Invalid Clustering choice" + std::string(choice), Here());
    }
}
