
#include "eckit/config/LocalConfiguration.h"

namespace infero {

/// @brief A configurable object
class Configurable {

public:

    Configurable(const eckit::Configuration& userConfig, const eckit::Configuration& defaults = eckit::LocalConfiguration() );

    virtual ~Configurable();

    const eckit::LocalConfiguration& config() const;

    friend std::ostream& operator<<(std::ostream& oss, const Configurable& obj);

private:

    // internal copy of config
    eckit::LocalConfiguration config_;

};

} // namespace infero