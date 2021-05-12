/*
 * (C) Copyright 1996- ECMWF.
 * 
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include "clustering_dbscan.h"
#include "infero/clustering/ClusteringDBscan.h"


ClusteringDBscan::ClusteringDBscan():
    min_threshold(DBSCAN_MIN_VAL)
{

}

int ClusteringDBscan::run(PredictionPtr& prediction)
{

    // read point data
    vector<Point> _points = readPrediction(prediction);

    // constructor
    DBSCAN ds(DBSCAN_MIN_N_CLUSTERS,
              DBSCAN_EPS,
              _points);

    // main loop
    ds.run();

    // fill up the cluster vector structure
    for(int i=0; i < ds.getTotalPointSize(); i++)
    {

        int cid = ds.m_points[i].clusterID;
        float x = ds.m_points[i].x;
        float y = ds.m_points[i].y;

        this->points.push_back(ClusterPoint(x,y,cid));
    }

    // calc cluster centers
    this->calculate_cluster_centers();

    return 0;

}



//read and ingest the prediction
std::vector<Point> ClusteringDBscan::readPrediction(PredictionPtr& prediction)
{

    std::vector<Point> _points;

    int val_count = 0;
    for (int irow=0; irow<prediction->n_rows(); irow++){
        for (int icol=0; icol<prediction->n_cols(); icol++){

            if (prediction->data_tensor[val_count] > min_threshold){
                Point p;
                p.clusterID = UNCLASSIFIED;
                p.x = irow;
                p.y = icol;
                p.z = 0;
                _points.push_back(p);
            }

            val_count++;
        }
    }

    return _points;
}
