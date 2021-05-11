#include "clustering.h"
#include "clustering/clustering_dbscan.h"
#include "eckit/log/Log.h"
#include "eckit/log/JSON.h"

#include <fstream>
#include <sstream>

using namespace eckit;


Clustering::Clustering()
{
    cluster_centers.resize(0);
    points.resize(0);
}


void Clustering::_calc_cluster_centers()
{

    // loop over cluster points
    // and fill a map < cluster ID, vec<points> >
    ClusterMap _clust;
    for (const auto& pt: points){

        auto _it = _clust.find(pt.cid);
        if (_it != _clust.end()){

            _it->second.push_back(ClusterPoint(pt.x, pt.y, pt.cid));

        } else {

            ClusterPoints vv{ClusterPoint(pt.x, pt.y, pt.cid)};
            ClusterPair _new_clsr(pt.cid, vv);
            _clust.emplace(_new_clsr);
        }
    }


    // average point coords for each cluster ID
    float _x, _y;
    for (const auto& k: _clust){

        _x = _y = 0;

        for (auto& p: k.second){
            _x += p.x;
            _y += p.y;
        }

        if (k.second.size()){
            _x /= k.second.size();
            _y /= k.second.size();
        }

        this->cluster_centers.push_back(ClusterPoint(_x, _y, k.first));

    }

}

// print clusters
void Clustering::print_summary()
{

    Log::info() << "\n*** centers *** " << std::endl;

    for (const auto& clust_ctr: cluster_centers){

        printf("%2d) x = %8.3f, y = %8.3f\n",
               clust_ctr.cid,
               clust_ctr.x,
               clust_ctr.y);
    }

    Log::info() << std::endl;

}


int Clustering::write_json(std::string filename)
{
   std::stringstream s;
   JSON json_out(s, JSON::Formatting::indent(2));

   json_out.startObject();
   json_out.startList();
   for (const auto& c: cluster_centers){
       json_out.startList();
       json_out << c.x << c.y;
       json_out.endList();
   }
   json_out.endList();
   json_out.endObject();

   std::ofstream fout( filename );

   if( fout ) {

       Log::info() << "Writing to JSON file "
                   << filename
                   << std::endl;

       fout << s.str();
       return 0;

   }

   Log::error() << "Failed to open output file "
                << filename
                << std::endl;
   return -1;
}

ClusteringPtr Clustering::create(std::string choice)
{
    if (choice.compare("dbscan") == 0){

        Log::info() << "creating ClusteringDBscan.. "
                  << std::endl;

        return ClusteringPtr(new ClusteringDBscan);

    } else {

        Log::error() << "Invalid Clustering choice "
                     << choice
                     << std::endl;

        return nullptr;
    }
}
