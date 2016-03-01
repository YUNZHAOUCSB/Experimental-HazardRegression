/**
 * Copyright (c) 2016 by Contributors
 * \author Ziqi Liu
 */

#include "./sgd_learner.h"
#include <mutex>

namespace hazard {

void SGDLearner::CalcLoss(const dmlc::RowBlock<feaid_t>& data) {
    loss_.clear();
    std::mutex m;
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
            m.lock();
            updater_->Exist(feaid);
            m.unlock();
            if(label) {
                res = updater_->CHazardFea(feaid, rcensor);
                std::get<0>(loss_[i]) += res.first;
                std::get<1>(loss_[i]) += res.second;
                res = updater_->CHazardFea(feaid, lcensor);
                std::get<2>(loss_[i]) += res.first;
                std::get<3>(loss_[i]) += res.second;
            } else {
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
        real_t rcumuhr = std::get<0>(loss_[x]);
        real_t rhr = std::get<1>(loss_[x]);
        real_t lcumuhr = std::get<2>(loss_[x]);
        real_t lhr = std::get<3>(loss_[x]);
        real_t log_interval = LogMinus(lcumuhr, rcumuhr);
        real_t log_r = -rcumuhr + std::log(rhr);
        real_t log_l = -lcumuhr + std::log(lhr);
        real_t rgrad = -std::exp(log_r - log_interval);
        real_t lgrad = std::exp(log_l - log_interval);
        if (rhr == 0.0f) rgrad = 0.0f;
        if (lhr == 0.0f) lgrad = 0.0f;
        return std::make_pair(rgrad, lgrad);
    }
}

void SGDLearner::CalcGrad(const dmlc::RowBlock<feaid_t>& data) {
    std::mutex m;
    gradients_.Clear();
#pragma omp parallel for num_threads(param_.nthreads)
    for (size_t i=0; i<data.size; i++) {
        const dmlc::Row<feaid_t>& d = data[i];
        uint8_t label = (uint8_t)d.label;
        time_t rcensor, lcensor;
        rcensor = d.index[0]; lcensor = d.index[1];
        std::pair<real_t, real_t> grad = GenGrad(label, i);
        for (size_t j=2; j<d.length; j++) {
            feaid_t feaid = d.index[j];
            if (!feat_set_.count(feaid)) continue;
            m.lock();
            SGDEntry& entry = gradients_[feaid];
            entry[rcensor] += grad.first;
            if (label) entry[lcensor] += grad.second;
            m.unlock();
        }
    }
}

}
