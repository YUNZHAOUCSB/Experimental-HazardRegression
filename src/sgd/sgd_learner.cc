/**
 * Copyright (c) 2015 by Contributors
 * \author Ziqi Liu
 */

#include <omp.h>
#include "./sgd_learner.h"

namespace hazard {

void SGDLearner::CalcLoss(const dmlc::RowBlock<feaid_t>& data) {
    loss_.clear();
    //loss_.resize(data.size);
#pragma omp parallel for num_threads(param_.nthreads)
    for (size_t i=0; i<data.size; i++) {
        const dmlc::Row<feaid_t>& d = data[i];
        time_t censor = d.index[0];
        uint8_t label = (uint8_t)d.label;
        std::pair<real_t, real_t> res;
        for(size_t j=1; j<d.length; j++) {
            feaid_t feaid = d.index[j];
            if(label) {
                res = updater_->CHazardFea(
                    feaid, std::max(param_.starttime, censor-param_.lcensor));
                std::get<0>(loss_[i]) += res.first;
                std::get<2>(loss_[i]) += res.second;
                res = updater_->CHazardFea(feaid, censor);
                std::get<1>(loss_[i]) += res.first;
                std::get<3>(loss_[i]) += res.second;
            }
            else {
                res = updater_->CHazardFea(feaid, censor);
                std::get<0>(loss_[i]) += res.first;
                std::get<2>(loss_[i]) += res.second;
            }
        }
    }
}

inline std::pair<real_t, real_t> SGDLearner::GenGrad(uint8_t label,
                                              time_t censor,
                                              size_t x) {
    if (label == 0) {
        return std::make_pair(std::get<2>(loss_[x]), 0.0f);
    }
    else {
        real_t sl = exp(-std::get<0>(loss_[x]));
        real_t sr = exp(-std::get<1>(loss_[x]));
        real_t denom = sl - sr;
        sl *= std::get<2>(loss_[x]) * 1.0f/denom;
        sr *= -std::get<3>(loss_[x]) * 1.0f/denom;
        return std::make_pair(sl, sr);
    }
}

void SGDLearner::CalcGrad(const dmlc::RowBlock<feaid_t>& data) {
    model_gradient_.Clear();
    for (size_t i=0; i<data.size; i++) {
        const dmlc::Row<feaid_t>& d = data[i];
        time_t censor = d.index[0];
        uint8_t label = (uint8_t)d.label;
#pragma omp parallel for num_threads(param_.nthreads)
        for (size_t j=1; j<d.length; j++) {
            feaid_t feaid = d.index[j];
            std::pair<real_t, real_t> grad = GenGrad(label, censor, i);
            auto entry = model_gradient_[feaid];
            entry[censor] += grad.first;
            if(label) entry[std::max(param_.starttime, censor - param_.lcensor)]
                          += grad.second;
        }
    }
}

}