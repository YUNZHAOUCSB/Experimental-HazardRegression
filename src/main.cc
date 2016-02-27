/**
 * Copyright (c) 2016 by Contributors
 * \author Ziqi Liu
 */
#include "hazard/learner.h"
#include "dmlc/parameter.h"

namespace hazard {
struct HazardParam : public dmlc::Parameter<HazardParam> {
    std::string task;
    std::string learner;

    DMLC_DECLARE_PARAMETER(HazardParam) {
        DMLC_DECLARE_FIELD(learner).set_default("sgd");
        DMLC_DECLARE_FIELD(task).set_default("train");
    }
};
DMLC_REGISTER_PARAMETER(HazardParam);

void WarnUnknownKWArgs(const HazardParam& param, const KWArgs& remain) {
    if (remain.empty()) return;
    LOG(WARNING) << "unrecognized keyword argument for task " << param.task;
    for (auto kw : remain) {
        LOG(WARNING) << "  " << kw.first << " = " << kw.second;
    }
}
}  //namespace hazard

int main(int argc, char *argv[]) {
    if(argc < 2) {
        LOG(ERROR) << "usage: hazard key1=val1 key2=val2 ...";
        return 0;
    }
    using namespace hazard;

    KWArgs kwargs;
    for(int i=1; i<argc; i++) {
        char name[256], val[256];
        if(sscanf(argv[i], "%[^=]=%[^\n]", name, val) == 2) {
            kwargs.push_back(std::make_pair(name, val));
        }
    }
    HazardParam param;
    auto kwargs_remain = param.InitAllowUnknown(kwargs);

    if(param.task == "train") {
        Learner* learner = Learner::Create(param.learner);
        WarnUnknownKWArgs(param, learner->Init(kwargs_remain));
        learner->Run();
        delete learner;
    }
    else if(param.task == "predict") {
        LOG(FATAL) << "TODO";
    }
    else {
        LOG(FATAL) << "unknown task: " << param.task;
    }

    return 0;
}