/**
 * Copyright (c) 2015 by Contributors
 * \author Ziqi Liu
 */

#include "sgd/sgd_learner.h"
#include "hazard/learner.h"

namespace hazard {

DMLC_REGISTER_PARAMETER(SGDLearnerParam);

Learner* Learner::Create(const std::string& type) {
    if (type == "sgd") {
        return new SGDLearner();
    } else {
        LOG(FATAL) << "unknown learner type: " << type;
    }
    return NULL;
}

}  // namespace difacto