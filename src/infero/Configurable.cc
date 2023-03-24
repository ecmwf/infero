
#include "eckit/exception/Exceptions.h"
#include "eckit/log/Log.h"

#include "infero/Configurable.h"

namespace infero {

// Setup from config
Configurable::Configurable(const eckit::Configuration& userConfig, const eckit::Configuration& defaults) {

    config_ = eckit::LocalConfiguration();

    // check that userConfig contains valid keys
    for (const auto& key: userConfig.keys()) {

        try {
            ASSERT(defaults.has(key));
        } catch (eckit::Exception& e) {
            eckit::Log::error() << "Error! model configuration contains an invalid key: " << std::string{key} << std::endl;
            throw;
        }
    }

    // assign user configuration
    for (const auto& key: defaults.keys()) {
        if (userConfig.has(key)) {
            config_.set(key, userConfig.getString(key));
        } else {
            config_.set(key, defaults.getString(key));
        }
    }
}


Configurable::~Configurable() {
}


const eckit::LocalConfiguration& Configurable::config() const {
    return config_;
}


std::ostream& operator<<(std::ostream& oss, const Configurable& obj) {
    oss << obj.config();
    return oss;
}

} // namespace infero