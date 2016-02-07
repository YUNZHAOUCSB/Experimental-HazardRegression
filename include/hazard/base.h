/**
 * Copyright (c) 2015 by Contributors
 * \author Ziqi Liu
 */

#ifndef HAZARD_BASE_H_
#define HAZARD_BASE_H_

#include <string>
#include <vector>

namespace hazard {
    typedef float real_t;
    typedef uint32_t feaid_t;
    typedef uint32_t time_t;
    typedef std::vector<std::pair<std::string, std::string>> KWArgs;
#define DEFAULT_NTHREADS 4
}

#endif