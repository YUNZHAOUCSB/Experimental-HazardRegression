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
#include <iomanip>

namespace hazard {
class SGDLearner : public Learner {
public:
    SGDLearner() {}
    ~SGDLearner(){
        delete updater_;
    }

    KWArgs Init(const std::string& task, const KWArgs& kwargs) override {
        KWArgs remain = param_.InitAllowUnknown(kwargs);
        updater_ = (SGDUpdater*)Updater::Create(param_.updater_type);
        remain = updater_->Init(remain, param_);
        // construct feat_set_ and updater_.cumu_cnt_
        std::unordered_map<feaid_t, size_t> feat_cnt;
        std::unordered_map<feaid_t, std::set<time_t>> ordinal;
        BatchIter reader(param_.data_in, param_.data_format,
                         0, 1, param_.batch_size);
        while(reader.Next()) {
            const dmlc::RowBlock<time_t>& data = reader.Value();
            for (size_t i=0; i<data.size; i++) {
                const dmlc::Row<time_t>& d = data[i];
                uint8_t label = (uint8_t)d.label;
                if(updater_->endtime_ < d.index[0]) updater_->endtime_ = d.index[0];
                for (size_t j = 2; j<d.length; j++) {
                    feaid_t feaid = (feaid_t)d.index[j];
                    feat_cnt[feaid] += 1;
                    updater_->Build(feaid, (time_t)d.index[0]);
                    if (label) updater_->Build(feaid, (time_t)d.index[1]);
                }
            }
        }
        // if model_in set init_hrate = 0.0f
//        if (!param_.model_in.empty()) {
//            updater_->SetHrate(0.0f);
//        }
        //construct feat_set_, init updater_->model_
        for (auto f : feat_cnt) {
            feaid_t feaid = f.first;
            if (f.second >= param_.feat_thresh) {
                feat_set_.insert(feaid);
                //updater_->Exist(feaid, ordinal[feaid]);
            }
        }
        //construct cumu_cnt_
//        for (auto e : ordinal) {
//            feaid_t feaid = e.first;
//            if (!feat_set_.count(feaid)) continue;
//            std::set<time_t>& s = e.second;
//            auto it = s.cbegin(); auto it0 = s.cbegin();
//            updater_->cumu_cnt_[feaid][*it] = 0;
//            it++;
//            for (; it!=s.cend(); it++, it0++) {
//                updater_->cumu_cnt_[feaid][*it] = updater_->cumu_cnt_[feaid][*it0] + 1;
//            }
//        }
        // init loss_
        loss_.clear();
        loss_.resize(param_.batch_size);

        if (task == "predict" && param_.model_in.empty()) {
            LOG(FATAL) << "model_in = NULL";
        }
        if(!param_.model_in.empty()) {
            ReadModel();
        }

        return remain;
    }

    inline real_t LogMinus(real_t cumul, real_t cumur, std::string type) {
        cumul = -cumul; cumur = -cumur;
        real_t tmp = std::exp(cumur - cumul);
        if (tmp >= 1.0 && type != "training") return 0.0;
        CHECK_LT(tmp, 1.0f);
        return cumul + std::log(1.0f - tmp);
    }

    void CalcRes(const dmlc::RowBlock<time_t>& data, std::string type) {
        real_t res = 0.0f;
#pragma omp parallel for reduction(+:res) num_threads(param_.nthreads)
        for (size_t i=0; i<data.size; i++) {
            const dmlc::Row<time_t>& d = data[i];
            uint8_t label = (uint8_t) d.label;
            if (label) {
                res += -LogMinus(std::get<1>(loss_[i]), std::get<0>(loss_[i]), type);
            } else {
                res += std::get<0>(loss_[i]);
            }
        }
        if(type == "training") train_loss_ += res;
        else val_loss_ += res;
    }

    void PrintRes(size_t epoch) {
	std::cout << std::fixed;
	std::cout << std::setprecision(6);
        std::cout << "Iter\t" << epoch << "\tTraining\t"
                  << train_loss_*1.0/(real_t)n_train_ << "\tValidation\t"
                  << val_loss_*1.0/(real_t)n_val_
                  << "\n";
    }

    void InitEpoch(size_t epoch) {
        n_train_ = 0; n_val_ = 0;
        train_loss_ = 0.0f; val_loss_ = 0.0f;
        updater_->InitEpoch(epoch);
    }

    void ReadModel() {
        updater_->ReadModel(param_.model_in);
    }

    void SaveModel(size_t epoch) {
        char name[256];
        sprintf(name, "%s_%lu", param_.model_out.c_str(), epoch);
        FILE* f = fopen(name, "w");
        updater_->SaveModel(f);
        fclose(f);
    }

    void Check() {
        size_t c=0;
        for(auto e : updater_->model_.model_map_) {
            SGDEntry& entry = e.second;
            c += entry.w.size();
        }
        printf("%lu\n", c);
    }

    void Complete(size_t epoch, const std::string& task) {
        PrintRes(epoch);
        if (task == "train" && !param_.model_out.empty()) SaveModel(epoch);
    }

    void Run(const std::string& task) override {
        if (task == "predict") param_.max_num_epochs = 1;
        for (uint32_t epoch=1; epoch <= param_.max_num_epochs; epoch++) {
            InitEpoch(epoch);
            RunEpoch(epoch, "training", task);
            RunEpoch(epoch, "validation", task);
            Complete(epoch, task);
        }
    }

    void RunEpoch(uint32_t epoch, const std::string& type, const std::string& task) {
        std::string filename = "training" == type ? param_.data_in :
            param_.val_data;
        BatchIter reader(filename, param_.data_format,
                         0, 1, param_.batch_size);
        while(reader.Next()) {
            const dmlc::RowBlock<time_t>& data = reader.Value();
            if (type == "training") n_train_ += data.size;
            else n_val_ += data.size;
            CalcLoss(data);
            CalcRes(data, type);
            if ("training" == type && task == "train") {
                CalcGrad(data);
                updater_->Update(gradients_);
            }
        }
    }

    //std::pair<real_t, real_t> GenGrad(uint8_t label, size_t x);
    void CalcLoss(const dmlc::RowBlock<time_t>& data);
    void CalcGrad(const dmlc::RowBlock<time_t>& data);

private:
    SGDLearnerParam param_;
    /** \brief updater specified */
    SGDUpdater* updater_;
    /** \brief gradients computed in each minibatch */
    SGDModel gradients_;
    /**
     *  \brief
     *   hazard cumulative at right censoring time
     *   hazard cumulative at left censoring time
    */
    std::vector<std::tuple<real_t, real_t>> loss_;
    /**
     *  \brief whether a feature in active set
     */
    std::unordered_set<feaid_t> feat_set_;
    real_t train_loss_;
    real_t val_loss_;
    size_t n_train_;
    size_t n_val_;
};  // class SGDLearner

}
#endif
