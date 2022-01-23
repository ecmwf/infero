#ifndef ModelStatistics_H
#define ModelStatistics_H

#include "eckit/log/Statistics.h"
#include <iostream>

namespace infero {

class ModelStatistics : public eckit::Statistics
{

public:

    ModelStatistics();

    eckit::Timing inferenceTiming_;
    eckit::Timing iTensorLayoutTiming_;
    eckit::Timing oTensorLayoutTiming_;

    void encode(eckit::Stream &s) const;

    void report(std::ostream &out, const char *indent = "") const;

    friend std::ostream &operator<<(std::ostream &s, const ModelStatistics &x) {
        x.report(s);
        return s;
    }

private:

    eckit::Timing calcTotalTime() const;
};

} // namespace infero

#endif // ModelStatistics_H
