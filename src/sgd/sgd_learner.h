/**
 * Copyright (c) 2015 by Contributors
 * \author Ziqi Liu
 */

#ifndef HAZARD_SGD_SGD_LEARNER_H_
#define HAZARD_SGD_SGD_LEARNER_H_

#include "./sgd_param.h"
#include "./sgd_updater.h"
#include "hazard/learner.h"
#include "hazard/base.h"
#include "data/batch_iter.h"
#include "dmlc/data.h"
#include "data/row_block.h"

namespace hazard {
class SGDLearner : public Learner {
public:
    SGDLearner() {}
    ~SGDLearner(){
        delete updater_;
    }
    KWArgs Init(const KWArgs& kwargs) override {
        KWArgs remain = param_.InitAllowUnknown(kwargs);
        updater_ = (SGDUpdater*)Updater::Create(param_.updater_type);
        remain = updater_->Init(remain, param_);
        loss_.clear();
        loss_.resize(param_.batch_size);
        return remain;
    }
    void Run() override {
        for (uint32_t epoch=0; epoch < param_.max_num_epochs; epoch++) {
            RunEpoch(epoch, "training");
            RunEpoch(epoch, "validating");
        }
    }
    void RunEpoch(uint32_t epoch, std::string type) {
        std::string filename = "training" == type ? param_.data_in :
            param_.val_data;
        BatchIter reader(filename, param_.data_format,
                         0, 1, param_.batch_size);
        while(reader.Next()) {
            const dmlc::RowBlock<feaid_t>& data = reader.Value();
            CalcLoss(data);
            CalcGrad(data);
            updater_->Update(model_gradient_);
        }
    }
    std::pair<real_t, real_t> GenGrad(uint8_t label,
                                      time_t censor,
                                      size_t x);
    void CalcLoss(const dmlc::RowBlock<feaid_t>& data);
    void CalcGrad(const dmlc::RowBlock<feaid_t>& data);
private:
    SGDLearnerParam param_;
    SGDUpdater* updater_;
    SGDModel model_gradient_;
    std::vector<std::tuple<real_t, real_t, real_t, real_t>> loss_;
};  // class SGDLearner

}
#endif