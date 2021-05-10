#ifndef CLUSTERING_H
#define CLUSTERING_H

#include "prediction.h"
#include <vector>
#include <map>
#include <string>
#include <memory>


// a cluster point
struct ClusterPoint{

    float x;
    float y;
    int cid;

    ClusterPoint(){}

    ClusterPoint(float x, float y, int cid):
        x(x), y(y), cid(cid){}
};


// cluster types
typedef std::vector<ClusterPoint> ClusterPoints;
typedef std::map<int, ClusterPoints> ClusterMap;
typedef std::pair<int, ClusterPoints> ClusterPair;


class Clustering;
typedef std::unique_ptr<Clustering> ClusteringPtr;


// generic clustering algorithm
class Clustering
{

public:

    Clustering();

    // run the clustering
    virtual int run(PredictionPtr& prediction) = 0;

    // summary of clustering
    virtual void print_summary();

    // write JSON
    virtual int write_json(std::string filename);

    static ClusteringPtr create(std::string choice);

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
    virtual void _calc_cluster_centers();

};

#endif // CLUSTERING_H
