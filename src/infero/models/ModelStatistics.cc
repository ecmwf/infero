#include "eckit/log/Log.h"

#include "ModelStatistics.h"

using eckit::Log;

namespace infero {

ModelStatistics::ModelStatistics()
{

}

void ModelStatistics::encode(eckit::Stream &s) const
{
    s << iTensorLayoutTiming_;
    s << inferenceTiming_;
    s << oTensorLayoutTiming_;
    s << totalTiming_;
}

void ModelStatistics::report(std::ostream &out, const char *indent) const
{

    out << std::endl
        << "========== Infero Model Statistics ========== "
        << std::endl;

    reportTime(out, "STATS: Time to reorder Input",
               iTensorLayoutTiming_, indent);

    reportTime(out, "STATS: Time to run Inference", inferenceTiming_, indent);

    reportTime(out, "STATS: Time to reorder Output",
               oTensorLayoutTiming_, indent);

    reportTime(out, "STATS: Total Time", totalTiming_, indent);

}

} // namespace infero
