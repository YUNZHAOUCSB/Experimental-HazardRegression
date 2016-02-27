/**
 * Copyright (c) 2016 by Contributors
 * \author Ziqi Liu
 */

#include <omp.h>
#include "./sgd_updater.h"
#include "data/batch_iter.h"
namespace hazard {
KWArgs SGDUpdater::Init(const KWArgs& kwargs,
                        const SGDLearnerParam& sgdlparam) {
    nthreads_ = sgdlparam.nthreads;
    KWArgs remain = param_.InitAllowUnknown(kwargs);
    return remain;
}

void SGDUpdater::InitEpoch(size_t epoch) {
    param_.eta = param_.lr * 1.0/pow(epoch, param_.decay);
}

std::pair<real_t, real_t> SGDUpdater::CHazardFea(feaid_t feaid, time_t censor) {
    SGDEntry& entry = model_[feaid];
    if(entry.Size() == 0)
        entry[starttime_] = param_.init_hrate;
    real_t hcumulative = 0.0f;
    real_t hrate = 0.0f;
    time_t k;
    entry.GreastLowerBound(censor, k);
    hrate = entry[k];
    auto it = entry.w.begin();
    auto it_next = entry.w.upper_bound(it->first);
    for (; it_next!=entry.w.cend() && it_next->first <= k; it++, it_next++) {
            hcumulative += (it_next->first - it->first)*it->second;
    }
    hcumulative += (censor - it->first) * it->second;
    return std::make_pair(hcumulative, hrate);
}

inline real_t SGDUpdater::SoftThresh(real_t w) {
    //soft thresholding
    real_t lrl1 = param_.eta * param_.l1;
    if(w > lrl1)  w -= lrl1;
    else if(w < -lrl1) w += lrl1;
    else w = 0.0f;
    //project w back to constraint set \geq 0
    w = std::max(0.0f, w);
    return w;
}

void SGDUpdater::CalcFldpX(SGDEntry& model_entry, std::vector<real_t>& x
               , std::vector<time_t>& k) {
    size_t i = 0;
    for (auto e : model_entry.w) {
        k[i] = e.first;
        x[i++] = e.second;
    }
}
void SGDUpdater::CalcFldpW(SGDEntry& model_entry, std::vector<real_t>& w) {
    size_t i = 0;
    std::map<time_t, real_t>::iterator it = model_entry.w.begin();
    std::map<time_t, real_t>::iterator it_next = model_entry.w.upper_bound(it->first);
    for (; it_next!=model_entry.w.cend(); it++, it_next++) {
        w[i++] = cumu_cnt_[it_next->first] - cumu_cnt_[it->first];
    }
    w[i] = cumu_cnt_.size() - cumu_cnt_[it->first];
}

void SGDUpdater::FLSAIsotonic(feaid_t feaid) {
    SGDEntry& model_entry = model_[feaid];
    std::vector<real_t> x(model_entry.Size());
    std::vector<time_t> k(x.size());
    std::vector<real_t> w(x.size());
    CalcFldpX(model_entry, x, k);
    CalcFldpW(model_entry, w);
    IsotonicDp(x.data(), x.size(), param_.l2*param_.eta, 5000, w.data());
    StoreChanges(feaid, x, k);
}

void SGDUpdater::StoreChanges(feaid_t feaid, std::vector<real_t>& x
                            , std::vector<time_t>& k) {
    SGDEntry& entry = model_[feaid];
    entry.w.clear();
    size_t i = 0;
    real_t prev, cur;
    prev = SoftThresh(x[i]); entry[k[i]] =  prev; i++;
    for (; i<x.size(); i++) {
        cur = SoftThresh(x[i]);
        if (prev != cur) {
            entry[k[i]] = cur;
            prev = cur;
        }
    }
}

void SGDUpdater::IsotonicDp(real_t* x,
                            size_t seq_len,
                            real_t lambda2,
                            size_t init_buf_sz,
                            real_t* obs_wts) {
    if (lambda2 < 1e-10) {
        LOG(ERROR) << "FLSAIsotonic: l2's must be strictly positive\n";
    }

    if (seq_len < 2) {
        return;
    }
    real_t back_pointers[seq_len*2];

    size_t check_freq = 40;

    if (init_buf_sz < 30 * check_freq) {
        LOG(ERROR) << "FLSAIsotonic : initial buffer size is too small\n";
    }

    Msg msg;
    msg.FirstMsg(init_buf_sz, x[0]*obs_wts[0], -0.5*obs_wts[0], lambda2, back_pointers);

    real_t * bp = back_pointers + 2;
    real_t * x0 = x + 1;
    int check_msg = check_freq - 1;

    real_t* wt = obs_wts + 1;
    for (size_t j = 1; j < seq_len; ++j, bp += 2, ++x0, ++wt, --check_msg) {
        msg.UpdMsgOpt((*x0)*(*wt), -0.5*(*wt), lambda2, bp);
        if (!check_msg) {
            check_msg = check_freq - 1;
            msg.ShiftMsg(check_freq);
        }
    }

    real_t last_msg_max = msg.MaxMsg();

    msg.BackTrace(x, seq_len, back_pointers, last_msg_max);
}

void SGDUpdater::UpdateGradient(feaid_t feaid, SGDEntry& grad_entry) {
    SGDEntry& model_entry = model_[feaid];
    for (auto e : grad_entry.w) {
        time_t tt = e.first;
        real_t val = e.second;
        time_t k;
        if (model_entry.GreastLowerBound(tt, k)) {
            model_entry[tt] -= param_.eta * val;
        }
        else {
            model_entry[tt] = model_entry[k];
            model_entry[tt] -= param_.eta * val;
        }
    }
}

void SGDUpdater::Update(SGDModel& grad) {
//#pragma omp parallel for num_threads(nthreads_)
    for (auto g : grad.model_map_) {
        feaid_t feaid = g.first;
        UpdateGradient(feaid, g.second);
        FLSAIsotonic(feaid);
    }
}

} //namespace hazard