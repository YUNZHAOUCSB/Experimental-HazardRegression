/**
 * Copyright (c) 2015 by Contributors
 * \author Ziqi Liu
 */

#include <omp.h>
#include "./sgd_updater.h"
#include "data/batch_iter.h"
namespace hazard {
KWArgs SGDUpdater::Init(const KWArgs& kwargs,
                        const SGDLearnerParam& sgdlparam) {
    starttime_ = sgdlparam.starttime;
    nthreads_ = sgdlparam.nthreads;
    KWArgs remain = param_.InitAllowUnknown(kwargs);
    BatchIter reader(sgdlparam.data_in, sgdlparam.data_format,
                     0, 1, sgdlparam.batch_size);
    std::map<time_t, char> ordinal;
    while(reader.Next()) {
        const dmlc::RowBlock<feaid_t>& data = reader.Value();
        for (size_t i=0; i<data.size; i++) {
            const dmlc::Row<feaid_t>& d = data[i];
            ordinal[d.index[0]] = 1;
            if(d.label) ordinal[std::max(starttime_, d.index[0]-sgdlparam.lcensor)] = 1;
        }
    }
    ordinal_.resize(ordinal.size());
    size_t ind = 0;
    for (auto o : ordinal) {
        ordinal_[ind++] = o.first;
    }
    return remain;
}

std::pair<real_t, real_t> SGDUpdater::CHazardFea(feaid_t feaid, time_t censor) {
    SGDEntry& entry = model_[feaid];
    if(entry.Size() == 0)
        entry[starttime_] = param_.initval;
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


void SGDUpdater::SoftThresh(real_t& w, real_t grad, real_t lr, real_t l1) {
    w -= lr*grad;
    real_t lrl1 = lr*l1;
    if(w > lrl1)  w -= lrl1;
    else if(w < -lrl1) w += lrl1;
    else w = 0.0f;
    w = std::max(0.0f, w);
}

void SGDUpdater::FLSAIsotonic(feaid_t feaid) {
    std::vector<real_t> vct(ordinal_.size());
    RestoreOrdinal(feaid, vct);
    IsotonicDp(vct.data(), vct.size(), param_.l2, 5000);
    StoreChanges(feaid, vct);
}

void SGDUpdater::RestoreOrdinal(feaid_t feaid, std::vector<real_t>& vct) {
    SGDEntry& entry = model_[feaid];
    auto it = entry.w.begin();
    auto it_next = entry.w.upper_bound(it->first);
    for(size_t i=0; i<vct.size(); i++) {
        if(it_next == entry.w.end()) {
            vct[i] = it->second;
        }
        else if(ordinal_[i] < it_next->first) {
            vct[i] = it->second;
        }
        else {
            vct[i] = it_next->second;
            it = it_next;
            it_next++;
        }
    }
}

void SGDUpdater::StoreChanges(feaid_t feaid, std::vector<real_t>& vct) {
    SGDEntry& entry = model_[feaid];
    entry.w.clear();
    entry[ordinal_[0]] = vct[0];
    for(size_t i=1; i<vct.size(); i++) {
        if (vct[i-1] != vct[i]) {
            entry[ordinal_[i]] = vct[i];
        }
    }
}

void SGDUpdater::IsotonicDp(real_t* x, int seq_len,
                            real_t lambda2, int init_buf_sz) {
    if (lambda2 < 1e-10) {
        LOG(ERROR) << "FLSAIsotonic: l2's must be strictly positive\n";
    }

    if (seq_len < 2) {
        return;
    }
    real_t back_pointers[seq_len*2];

    int check_freq = 40;

    if (init_buf_sz < 30 * check_freq) {
        LOG(ERROR) << "FLSAIsotonic : initial buffer size is too small\n";
    }

    Msg msg;
    msg.FirstMsg(init_buf_sz, x[0], -0.5, lambda2, back_pointers);

    //int max_msg_sz = 0;

    real_t * bp = back_pointers + 2;
    real_t * x0 = x + 1;
    int check_msg = check_freq - 1;

    for (int j = 1; j < seq_len; ++j, bp += 2, ++x0, --check_msg) {
        msg.UpdMsgOpt((*x0), -0.5, lambda2, bp);
        if (!check_msg) {
            check_msg = check_freq - 1;
            msg.ShiftMsg(check_freq);
        }
    }

    real_t last_msg_max = msg.MaxMsg();

    msg.BackTrace(x, seq_len, back_pointers, last_msg_max);
}

void SGDUpdater::Update(SGDModel& grad) {
    for (auto g : grad.model_map_) {
        feaid_t feaid = g.first;
        SGDEntry& entry = g.second;
        SGDEntry& model_entry = model_[feaid];
        if(model_entry.Size() == 0)
             model_entry[starttime_] = param_.initval;
//#pragma omp parallel for num_threads(nthreads_)
        for (auto e : entry.w) {
            time_t tt = e.first;
            real_t val = e.second;
            time_t k;
            if(model_entry.GreastLowerBound(tt, k)) {
                SoftThresh(model_entry[tt], val, param_.lr, param_.l1);
            }
            else {
                model_entry[tt] = model_entry[k];
                SoftThresh(model_entry[tt], val, param_.lr, param_.l1);
            }
        }
    }

//#pragma omp parallel for num_threads(nthreads_)
    for (auto g : grad.model_map_) {
        feaid_t feaid = g.first;
        FLSAIsotonic(feaid);
    }
}

}