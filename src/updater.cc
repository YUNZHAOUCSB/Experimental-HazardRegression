/**
 * Copyright (c) 2015 by Contributors
 * \author Ziqi Liu
 */

#include "sgd/sgd_updater.h"
#include "hazard/updater.h"

namespace hazard {

DMLC_REGISTER_PARAMETER(SGDUpdaterParam);

    Updater* Updater::Create(const std::string& type) {
        if (type == "sgd_dynamic") {
            return new SGDUpdater();
        } else {
            LOG(FATAL) << "unknown updater type: " << type;
        }
        return NULL;
    }

}  //namespace hazard