/**
 * Copyright (c) 2015 by Contributors
 * \author Ziqi Liu
 */

#ifndef HAZARD_UPDATER_H_
#define HAZARD_UPDATER_H_

#include <string>
#include "hazard/base.h"

namespace hazard {
    class Updater {
    public:
        static Updater* Create(const std::string& type);
        Updater() {}
        virtual ~Updater() {}
        virtual KWArgs Init(const KWArgs& kwargs) = 0;
        virtual void Update() = 0;
    };
}

#endif