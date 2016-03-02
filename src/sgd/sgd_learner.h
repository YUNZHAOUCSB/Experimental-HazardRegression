/**
 * Copyright (c) 2016 by Contributors
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
#include <unordered_set>
#include <set>

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
        // construct feat_set_ and updater_.cumu_cnt_
        std::unordered_map<feaid_t, size_t> feat_cnt;
        std::set<time_t> ordinal;
        BatchIter reader(param_.data_in, param_.data_format,
                         0, 1, param_.batch_size);
        while(reader.Next()) {
            const dmlc::RowBlock<feaid_t>& data = reader.Value();
            for (size_t i=0; i<data.size; i++) {
                const dmlc::Row<feaid_t>& d = data[i];
                uint8_t label = (uint8_t)d.label;
                ordinal.insert(d.index[0]);
                if (label) ordinal.insert(d.index[1]);
                for (size_t j = 2; j<d.length; j++) {
                    feaid_t feaid = d.index[j];
                    feat_cnt[feaid] += 1;
                }
            }
        }
        //construct cumu_cnt_
        auto it = ordinal.cbegin();
        auto it0 = ordinal.cbegin();
        updater_->starttime_ = *it;
        updater_->cumu_cnt_[*it] = 0;
        it++;
        for (; it!=ordinal.cend(); it++, it0++) {
            updater_->cumu_cnt_[*it] = updater_->cumu_cnt_[*(it0)] + 1;
        }
        //construct feat_set_, init updater_->model_
        for (auto f : feat_cnt) {
            feaid_t feaid = f.first;
            if (f.second >= param_.feat_thresh) {
                feat_set_.insert(feaid);
                updater_->Exist(feaid);
            }
        }
        // init loss_
        loss_.clear();
        loss_.resize(param_.batch_size);

        if(!param_.model_in.empty()) {
            ReadModel();
        }

        return remain;
    }

    inline real_t LogMinus(real_t cumul, real_t cumur) {
        cumul = -cumul; cumur = -cumur;
        real_t tmp = std::exp(cumur - cumul);
        CHECK_LT(tmp, 1.0f);
        return cumul + std::log(1.0f - tmp);
    }

    void CalcRes(const dmlc::RowBlock<feaid_t>& data, std::string type) {
        real_t res = 0.0f;
#pragma omp parallel for reduction(+:res) num_threads(param_.nthreads)
        for (size_t i=0; i<data.size; i++) {
            const dmlc::Row<feaid_t>& d = data[i];
            uint8_t label = (uint8_t) d.label;
            if (label) {
                res += -LogMinus(std::get<2>(loss_[i]), std::get<0>(loss_[i]));
            } else {
                res += std::get<0>(loss_[i]);
            }
        }
        if(type == "training") train_loss_ += res;
        else val_loss_ += res;
    }

    void PrintRes(size_t epoch) {
        LOG(INFO) << "Iter\t" << epoch << "\tTraining\t"
            << train_loss_ << "\tValidation\t" << val_loss_
                  << "\n";
    }

    void InitEpoch(size_t epoch) {
        train_loss_ = 0.0f; val_loss_ = 0.0f;
        updater_->InitEpoch(epoch);
    }

    void ReadModel() {
        updater_->ReadModel(param_.model_in);
    }

    void SaveModel(size_t epoch) {
        char name[256];
        sprintf(name, "model_%lu", epoch);
        FILE* f = fopen(name, "w");
        updater_->SaveModel(f);
        fclose(f);
    }

    void Complete(size_t epoch) {
        PrintRes(epoch);
        SaveModel(epoch);
    }

    void Run() override {
        for (uint32_t epoch=1; epoch <= param_.max_num_epochs; epoch++) {
            InitEpoch(epoch);
            RunEpoch(epoch, "training");
            RunEpoch(epoch, "validation");
            Complete(epoch);
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
            CalcRes(data, type);
            if ("training" == type) {
                CalcGrad(data);
                updater_->Update(gradients_);
            }
        }
    }

    std::pair<real_t, real_t> GenGrad(uint8_t label, size_t x);
    void CalcLoss(const dmlc::RowBlock<feaid_t>& data);
    void CalcGrad(const dmlc::RowBlock<feaid_t>& data);

private:
    SGDLearnerParam param_;
    /** \brief updater specified */
    SGDUpdater* updater_;
    /** \brief gradients computed in each minibatch */
    SGDModel gradients_;
    /**
     *  \brief
     *   hazard cumulative at right censoring time
     *   hazard rate at right censoring time
     *   hazard cumulative at left censoring time
     *   hazard rate at left censoring time
    */
    std::vector<std::tuple<real_t, real_t, real_t, real_t>> loss_;
    /**
     *  \brief whether a feature in active set
     */
    std::unordered_set<feaid_t> feat_set_;
    real_t train_loss_;
    real_t val_loss_;
};  // class SGDLearner

}
#endif
