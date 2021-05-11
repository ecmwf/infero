#ifndef CLUSTERING_DBSCAN_H
#define CLUSTERING_DBSCAN_H

#include "clustering.h"
#include "contrib/DBSCAN/dbscan.h"

// min number of clusters
static const unsigned int DBSCAN_MIN_N_CLUSTERS = 1;

// dbscan point distance in cluster
static constexpr float DBSCAN_EPS = 10.0;

// min point val to be considered for clustering
static constexpr float DBSCAN_MIN_VAL = 0.6;


// runs a DBScan algorithm
class ClusteringDBscan: public Clustering
{

public:

    ClusteringDBscan();

    // run clustering
    virtual int run(PredictionPtr& prediction);


private:

    // read the prediction
    virtual std::vector<Point> readPrediction(PredictionPtr& prediction);


private:

    float min_threshold;

};

#endif // CLUSTERING_DBSCAN_H
