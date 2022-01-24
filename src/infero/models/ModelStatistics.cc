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
}

void ModelStatistics::report(std::ostream &out, const char *indent) const
{

    out << std::endl
        << "========== Infero Model Statistics ========== "
        << std::endl;

    reportTime(out, "INFERO-STATS: Time to copy/reorder Input ",
               iTensorLayoutTiming_, indent);

    reportTime(out, "INFERO-STATS: Time to execute inference  ", inferenceTiming_, indent);

    reportTime(out, "INFERO-STATS: Time to copy/reorder Output",
               oTensorLayoutTiming_, indent);

    reportTime(out, "INFERO-STATS: Total Time", calcTotalTime(), indent);

}

eckit::Timing ModelStatistics::calcTotalTime() const
{

    eckit::Timing totalTiming_;

    totalTiming_ += iTensorLayoutTiming_;
    totalTiming_ += inferenceTiming_;
    totalTiming_ += oTensorLayoutTiming_;

    return totalTiming_;
}

} // namespace infero
