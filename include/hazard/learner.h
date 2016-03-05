/**
 * Copyright (c) 2015 by Contributors
 * \author Ziqi Liu
 */

#ifndef HAZARD_LEARNER_H_
#define HAZARD_LEARNER_H_

#include <string>
#include "hazard/base.h"

namespace hazard {
    class Learner {
    public:
        static Learner* Create(const std::string& type);
        Learner() {}
        virtual ~Learner() {}
        virtual KWArgs Init(const std::string& task, const KWArgs& kwargs) = 0;
        virtual void Run(const std::string& task) = 0;
    };
}

#endif