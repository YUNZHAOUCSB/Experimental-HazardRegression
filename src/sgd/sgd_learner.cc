/**
 * Copyright (c) 2016 by Contributors
 * \author Ziqi Liu
 */

#include <omp.h>
#include "./sgd_learner.h"

namespace hazard {

void SGDLearner::CalcLoss(const dmlc::RowBlock<feaid_t>& data) {
    loss_.clear();
#pragma omp parallel for num_threads(param_.nthreads)
    for (size_t i=0; i<data.size; i++) {
        const dmlc::Row<feaid_t>& d = data[i];
        uint8_t label = (uint8_t)d.label;
        time_t rcensor, lcensor;
        rcensor = d.index[0]; lcensor = d.index[1];
        std::pair<real_t, real_t> res;
        for(size_t j=2; j<d.length; j++) {
            feaid_t feaid = d.index[j];
            if (!feat_set_.count(feaid)) continue;
            if(label) {
                res = updater_->CHazardFea(feaid, rcensor);
                std::get<0>(loss_[i]) += res.first;
                std::get<1>(loss_[i]) += res.second;
                res = updater_->CHazardFea(feaid, lcensor);
                std::get<2>(loss_[i]) += res.first;
                std::get<3>(loss_[i]) += res.second;
            }
            else {
                res = updater_->CHazardFea(feaid, rcensor);
                std::get<0>(loss_[i]) += res.first;
                std::get<1>(loss_[i]) += res.second;
            }
        }
    }
}

inline std::pair<real_t, real_t> SGDLearner::GenGrad(uint8_t label,
                                                     size_t x) {
    if (label == 0) {
        return std::make_pair(std::get<1>(loss_[x]), 0.0f);
    }
    else {
        real_t sr = exp(-std::get<0>(loss_[x]));
        real_t sl = exp(-std::get<2>(loss_[x]));
        real_t denom = sl - sr;
        sr *= -std::get<1>(loss_[x]) * 1.0f/denom;
        sl *= std::get<3>(loss_[x]) * 1.0f/denom;
        return std::make_pair(sr, sl);
    }
}

void SGDLearner::CalcGrad(const dmlc::RowBlock<feaid_t>& data) {
    gradients_.Clear();
    for (size_t i=0; i<data.size; i++) {
        const dmlc::Row<feaid_t>& d = data[i];
        uint8_t label = (uint8_t)d.label;
        time_t rcensor, lcensor;
        rcensor = d.index[0]; lcensor = d.index[1];
        std::pair<real_t, real_t> grad = GenGrad(label, i);
#pragma omp parallel for num_threads(param_.nthreads)
        for (size_t j=2; j<d.length; j++) {
            feaid_t feaid = d.index[j];
            if (!feat_set_.count(feaid)) continue;
            auto entry = gradients_[feaid];
            entry[rcensor] += grad.first;
            if (label) entry[lcensor] += grad.second;
        }
    }
}

}