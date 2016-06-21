/**
 * Copyright (c) 2016 by Contributors
 * \author Ziqi Liu
 */

#include "./sgd_learner.h"
#include <mutex>

namespace hazard {

void SGDLearner::CalcLoss1(const dmlc::RowBlock<time_t>& data) {
    loss1_.clear();
    loss1_.resize(param_.batch_size);
#pragma omp parallel for num_threads(param_.nthreads)
    for (size_t i=0; i<data.size; i++) {
        const dmlc::Row<time_t>& d = data[i];
        uint8_t label = (uint8_t)d.label;
        time_t rcensor, lcensor;
        rcensor = (time_t)d.index[0]; lcensor = (time_t)d.index[1];
        real_t res;
        for(size_t j=2; j<d.length; j++) {
            feaid_t feaid = (feaid_t)d.index[j];
            if (!feat_set_.count(feaid)) continue;
            if(label) {
                res = updater_->CHazardFea(feaid, rcensor);
                std::get<0>(loss1_[i]) += res;
                res = updater_->CHazardFea(feaid, lcensor);
                std::get<1>(loss1_[i]) += res;
            }
        }
    }
}

void SGDLearner::CalcLoss2(const dmlc::RowBlock<time_t>& data) {
    loss1_.clear();
    loss1_.resize(param_.batch_size);
#pragma omp parallel for num_threads(param_.nthreads)
    for (size_t i=0; i<data.size; i++) {
        const dmlc::Row<time_t>& d = data[i];
        uint8_t label = (uint8_t)d.label;
        time_t rcensor, lcensor;
        rcensor = (time_t)d.index[0]; lcensor = (time_t)d.index[1];
        real_t res;
        for(size_t j=2; j<d.length; j++) {
            feaid_t feaid = (feaid_t)d.index[j];
            if (!feat_set_.count(feaid)) continue;
            if(label) {
                res = updater_->CHazardFea1(feaid, rcensor);
                std::get<0>(loss1_[i]) += res;
                res = updater_->CHazardFea1(feaid, lcensor);
                std::get<1>(loss1_[i]) += res;
            }
        }
    }
}
void SGDLearner::CalcLoss(const dmlc::RowBlock<time_t>& data) {
    loss_.clear();
    loss_.resize(param_.batch_size);
#pragma omp parallel for num_threads(param_.nthreads)
    for (size_t i=0; i<data.size; i++) {
        const dmlc::Row<time_t>& d = data[i];
        uint8_t label = (uint8_t)d.label;
        time_t rcensor, lcensor;
        rcensor = (time_t)d.index[0]; lcensor = (time_t)d.index[1];
        real_t res;
        for(size_t j=2; j<d.length; j++) {
            feaid_t feaid = (feaid_t)d.index[j];
            if (!feat_set_.count(feaid)) continue;
            if(label) {
                res = updater_->CHazardFea(feaid, rcensor);
                std::get<0>(loss_[i]) += res;
                res = updater_->CHazardFea(feaid, lcensor);
                std::get<1>(loss_[i]) += res;
            } else {
                res = updater_->CHazardFea(feaid, rcensor);
                std::get<0>(loss_[i]) += res;
            }
        }
    }
}

//inline std::pair<real_t, real_t> SGDLearner::GenGrad(uint8_t label,
//                                                     size_t x) {
//    if (label == 0) {
//        return std::make_pair(std::get<1>(loss_[x]), 0.0f);
//    }
//    else {
//        real_t rcumuhr = std::get<0>(loss_[x]);
//        real_t rhr = std::get<1>(loss_[x]);
//        real_t lcumuhr = std::get<2>(loss_[x]);
//        real_t lhr = std::get<3>(loss_[x]);
//        real_t log_interval = LogMinus(lcumuhr, rcumuhr, "training");
//        real_t log_r = -rcumuhr + std::log(rhr);
//        real_t log_l = -lcumuhr + std::log(lhr);
//        real_t rgrad = -std::exp(log_r - log_interval);
//        real_t lgrad = std::exp(log_l - log_interval);
//        if (rhr == 0.0f) rgrad = 0.0f;
//        if (lhr == 0.0f) lgrad = 0.0f;
//        return std::make_pair(rgrad, lgrad);
//    }
//}

void SGDLearner::AveGrad(const dmlc::RowBlock<time_t>& data) {
    std::mutex m;
    for(size_t i=0;i<data.size;i++) {
        const dmlc::Row<time_t>& d=data[i];
        uint8_t label=(uint8_t)d.label;
        time_t rcensor, lcensor;
        rcensor=(time_t)d.index[0]; lcensor=(time_t)d.index[1];
        real_t gg=0.0;
        if(label) {
            real_t cumur=std::get<0>(loss1_[i]);
            real_t cumul=std::get<1>(loss1_[i]);
            real_t tmp = cumur - cumul;
            if(tmp<param_.epsilon) gg=-1.0/(param_.epsilon);
            else gg = 1.0/(1.0-std::exp(tmp));
        }
        if(label) {
#pragma omp parallel for num_threads(param_.nthreads)
            for (size_t j=2; j<d.length; j++) {
                feaid_t feaid = (feaid_t)d.index[j];
                if (!feat_set_.count(feaid)) continue;
                m.lock();
                SGDEntry& mentry = updater_->model_[feaid];
                m.unlock();
                auto it = mentry.w.cbegin();
                auto it_next = mentry.w.cbegin();
                it_next++;
                time_t t;
                for (; it_next!=mentry.w.cend() && it_next->first <=lcensor;
                     it++, it_next++) {
                    t = it->first;
                    std::get<1>(mentry[t]) += it_next->first - t;
                }
                for (; it_next!=mentry.w.cend() && it_next->first <= rcensor;
                     it++, it_next++) {
                    t = it->first;
                    std::get<1>(mentry[t]) += gg*(it_next->first-t);
					//printf("%e\n", std::get<1>(mentry[t]));
                }
            }
        }
        else {
#pragma omp parallel for num_threads(param_.nthreads)
            for (size_t j=2; j<d.length; j++) {
                feaid_t feaid = (feaid_t)d.index[j];
                if (!feat_set_.count(feaid)) continue;
                m.lock();
                SGDEntry& mentry = updater_->model_[feaid];
                m.unlock();
                CHECK_GT(mentry.Size(),1);
                auto it = mentry.w.cbegin();
                auto it_next = mentry.w.cbegin();
                it_next++;
                CHECK_LE(it->first, rcensor);
                time_t t;
                for (; it_next!=mentry.w.cend() && it_next->first <= rcensor
                         ; it++, it_next++) {
                    t = it->first;
                    std::get<1>(mentry[t]) += it_next->first - t;
                }
            }
        }
    }
}


void SGDLearner::CalcGrad(const dmlc::RowBlock<time_t>& data) {
    std::mutex m;
    gradients_.Clear();
    for (size_t i=0; i<data.size; i++) {
        const dmlc::Row<time_t>& d = data[i];
        uint8_t label = (uint8_t)d.label;
        time_t rcensor, lcensor;
        rcensor = (time_t)d.index[0]; lcensor = (time_t)d.index[1];
        real_t gg = 0.0;
        real_t gg1 = 0.0;
        if(label) {
			real_t cumur=std::get<0>(loss_[i]);
			real_t cumul=std::get<1>(loss_[i]);
			real_t tmp = cumur - cumul;
			if(tmp<param_.epsilon) gg=-1.0/(param_.epsilon);
			else gg = 1.0/(1.0-std::exp(tmp));
        }
        if(label) {
#pragma omp parallel for num_threads(param_.nthreads)
            for (size_t j=2; j<d.length; j++) {
                feaid_t feaid = (feaid_t)d.index[j];
                if (!feat_set_.count(feaid)) continue;
                m.lock();
                SGDEntry& mentry = updater_->model_[feaid];
                SGDEntry& entry = gradients_[feaid];
                m.unlock();
                auto it = mentry.w.cbegin();
                auto it_next = mentry.w.cbegin();
                it_next++;
                time_t t;
                for (; it_next!=mentry.w.cend() && it_next->first <=lcensor;
                     it++, it_next++) {
                    t = it->first;
                    std::get<0>(entry[t]) += it_next->first- t;
                }
                for (; it_next!=mentry.w.cend() && it_next->first <= rcensor;
                     it++, it_next++) {
                    t = it->first;
                    std::get<0>(entry[t]) += (gg)*(it_next->first-t);
                }
            }
        }
        else {
#pragma omp parallel for num_threads(param_.nthreads)
            for (size_t j=2; j<d.length; j++) {
                feaid_t feaid = (feaid_t)d.index[j];
                if (!feat_set_.count(feaid)) continue;
                m.lock();
                SGDEntry& mentry = updater_->model_[feaid];
                SGDEntry& entry = gradients_[feaid];
                m.unlock();
                CHECK_GT(mentry.Size(),1);
                auto it = mentry.w.cbegin();
                auto it_next = mentry.w.cbegin();
                it_next++;
                CHECK_LE(it->first, rcensor);
                time_t t;
                for (; it_next!=mentry.w.cend() && it_next->first <= rcensor
                         ; it++, it_next++) {
                    t = it->first;
                    std::get<0>(entry[t]) += it_next->first-t;
                }
            }
        }
    }
}

void SGDLearner::CalcVRGrad(const dmlc::RowBlock<time_t>& data) {
    CalcLoss2(data);
    std::mutex m;
    gradients_.Clear();
    for (size_t i=0; i<data.size; i++) {
        const dmlc::Row<time_t>& d = data[i];
        uint8_t label = (uint8_t)d.label;
        time_t rcensor, lcensor;
        rcensor = (time_t)d.index[0]; lcensor = (time_t)d.index[1];
        real_t gg = 0.0;
        real_t gg1 = 0.0;
        if(label) {
			real_t cumur=std::get<0>(loss_[i]);
			real_t cumul=std::get<1>(loss_[i]);
			real_t tmp = cumur - cumul;
			if(tmp<param_.epsilon) gg=-1.0/(param_.epsilon);
			else gg = 1.0/(1.0-std::exp(tmp));

			cumur=std::get<0>(loss1_[i]);
			cumul=std::get<1>(loss1_[i]);
			tmp = cumur - cumul;
			if(tmp<param_.epsilon) gg1=-1.0/(param_.epsilon);
			else gg1 = 1.0/(1.0-std::exp(tmp));
			//printf("%e\n",gg1);
        }
        if(label) {
#pragma omp parallel for num_threads(param_.nthreads)
            for (size_t j=2; j<d.length; j++) {
                feaid_t feaid = (feaid_t)d.index[j];
                if (!feat_set_.count(feaid)) continue;
                m.lock();
                SGDEntry& mentry = updater_->model_[feaid];
                SGDEntry& entry = gradients_[feaid];
                m.unlock();
                auto it = mentry.w.cbegin();
                auto it_next = mentry.w.cbegin();
                it_next++;
                time_t t;
                for (; it_next!=mentry.w.cend() && it_next->first <=lcensor;
                     it++, it_next++) {
                    t = it->first;
                    std::get<0>(entry[t]) += (std::get<1>(mentry[t])*1.0/n_train_);//*1.0/data.size;
                    //std::get<0>(entry[t]) += it_next->first- t;
                }
                for (; it_next!=mentry.w.cend() && it_next->first <= rcensor;
                     it++, it_next++) {
                    t = it->first;
                    std::get<0>(entry[t]) += ((gg-gg1)*(it_next->first-t) + std::get<1>(mentry[t])*1.0/n_train_);//*1.0/data.size;
                    //std::get<0>(entry[t]) += (gg)*(it_next->first-t);// + std::get<1>(mentry[t])*1.0/n_train_;
                }
            }
        }
        else {
#pragma omp parallel for num_threads(param_.nthreads)
            for (size_t j=2; j<d.length; j++) {
                feaid_t feaid = (feaid_t)d.index[j];
                if (!feat_set_.count(feaid)) continue;
                m.lock();
                SGDEntry& mentry = updater_->model_[feaid];
                SGDEntry& entry = gradients_[feaid];
                m.unlock();
                CHECK_GT(mentry.Size(),1);
                auto it = mentry.w.cbegin();
                auto it_next = mentry.w.cbegin();
                it_next++;
                CHECK_LE(it->first, rcensor);
                time_t t;
                for (; it_next!=mentry.w.cend() && it_next->first <= rcensor
                         ; it++, it_next++) {
                    t = it->first;
                    std::get<0>(entry[t]) += (std::get<1>(mentry[t])*1.0/n_train_);//*1.0/data.size;
                    //std::get<0>(entry[t]) += it_next->first-t;//std::get<1>(mentry[t])*1.0/n_train_;
                }
            }
        }
    }
}

}
