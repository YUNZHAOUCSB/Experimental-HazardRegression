/**
 * Copyright (c) 2016 by Contributors
 * \author Ziqi Liu
 */

#include "./sgd_updater.h"
#include "data/batch_iter.h"
#include <fstream>

namespace hazard {
KWArgs SGDUpdater::Init(const KWArgs& kwargs,
                        const SGDLearnerParam& sgdlparam) {
    nthreads_ = sgdlparam.nthreads;
    KWArgs remain = param_.InitAllowUnknown(kwargs);
    if(!param_.debug.empty())
    debug_ = fopen(param_.debug.c_str(), "w");
    return remain;
}

void SGDUpdater::InitEpoch(size_t epoch) {
    param_.eta = param_.lr * 1.0/pow(epoch, param_.decay);
}

void SGDUpdater::ReadModel(std::string name) {
    for (auto it=model_.model_map_.begin(); it!=model_.model_map_.end(); it++) {
        SGDEntry& entry = it->second;
        for (auto it1=entry.w.begin();it1!=entry.w.end();it1++) {
            it1->second = 0.0;
        }
    }
    std::ifstream infile(name, std::ifstream::in);
    std::string line;
    feaid_t feaid; time_t tid; real_t val;
    while(std::getline(infile, line)) {
        std::stringstream l(line);
        std::string tmp;
        std::getline(l, tmp, '\t');
        std::stringstream(tmp) >> feaid;
        SGDEntry& entry = model_[feaid];
        while(std::getline(l, tmp, '\t')) {
            std::stringstream item(tmp);
            std::string stmp;
            std::getline(item, stmp, ':');
            std::stringstream(stmp) >> tid;
            std::getline(item, stmp, ':');
            std::stringstream(stmp) >> val;
            auto pit = entry.w.upper_bound(tid);
            tid = pit->first;
            if(pit!=entry.w.end()) entry[tid] = val;
        }
    }
    for(auto it=model_.model_map_.begin(); it!=model_.model_map_.end(); it++) {
        SGDEntry& entry = it->second;
        real_t prev=0.0;
        for (auto it1 = entry.w.begin(); it1!=entry.w.end(); it1++) {
            if(it1->second < prev) {
                it1->second = prev;
            }
            else {
                prev = it1->second;
            }
        }
    }
}

void SGDUpdater::SaveModel(FILE* f) {
    std::vector<feaid_t> feats(model_.Size());
    size_t i=0;
    for (auto e : model_.model_map_) {
        feats[i++] = e.first;
    }
    std::sort(feats.begin(), feats.end());
    for (i=0; i<feats.size(); i++) {
        feaid_t feaid = feats[i];
        SGDEntry& entry = model_[feaid];
        fprintf(f, "%d", feaid);
        for (auto o : entry.w) {
	    if(o.second != 0.0)
            fprintf(f, "\t%e:%e", o.first, o.second);
        }
        fprintf(f, "\n");
    }
}

real_t SGDUpdater::CHazardFea(feaid_t feaid, time_t censor) {
    SGDEntry& entry = model_[feaid];
    real_t hcumulative = 0.0;
    time_t k;
    auto it = entry.w.begin();
    auto it_next = entry.w.upper_bound(it->first);
    entry.GreastLowerBound(censor, k);
    if(it->first >= censor) return censor*it->second;
    hcumulative += it->first*it->second;
    for (; it_next!=entry.w.cend() && it_next->first <= k; it++, it_next++) {
            hcumulative += (it_next->first - it->first)*it_next->second;
    }
    if(it_next != entry.w.cend())
    hcumulative += (censor - it->first) * it_next->second;
    return hcumulative;
}

void SGDUpdater::Pool (real_t* y, real_t* w, int i, int j) {
    int k;
    real_t s0=0, s1=0;

    for (k=i; k<=j; k++) {s1 += y[k]*w[k]; s0 += w[k];}
    s1 /= s0;
    for (k=i; k<=j; k++) y[k] = s1;
}

void SGDUpdater::PAVA (real_t* y, real_t* w, int n) {
    if (n <= 1) return;
    int npools;
    n--;

    /* keep passing through the array until pooling is not needed */
    do {
        int i = 0;
        npools = 0;
        while (i < n) {
            int k = i;
            /* starting at y[i], find longest non-increasing sequence y[i:k] */
            while (k < n && y[k] >= y[k+1])  k++;
            if (y[i] != y[k]) {Pool(y, w, i, k); npools++;}
            i = k+1;
        }
    } while (npools > 0);
}

inline real_t SGDUpdater::SoftThresh(real_t w) {
    //soft thresholding
    real_t lrl1 = param_.eta * param_.l1;
    if(w > lrl1)  w -= lrl1;
    else if(w < -lrl1) w += lrl1;
    else w = 0.0f;
    //project w back to constraint set \geq 0
    w = std::max(0.0, w);
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
    void SGDUpdater::CalcFldpW(SGDEntry& model_entry, std::vector<real_t>& w, feaid_t feaid) {
    size_t i = 0;
    std::map<time_t, real_t>::iterator it = model_entry.w.begin();
    std::map<time_t, real_t>::iterator it_next = model_entry.w.upper_bound(it->first);
    for (; it_next!=model_entry.w.cend(); it++, it_next++) {
        w[i++] = cumu_cnt_[feaid][it_next->first] - cumu_cnt_[feaid][it->first];
    }
    w[i] = cumu_cnt_[feaid].size() - cumu_cnt_[feaid][it->first];
    CHECK_GT(w[i],0);
}

void SGDUpdater::ProxOperators(feaid_t feaid) {
    SGDEntry& model_entry = model_[feaid];
    std::vector<real_t> x(model_entry.Size());
    std::vector<time_t> k(x.size());
    std::vector<real_t> w(x.size(),1.0);
    CalcFldpX(model_entry, x, k);
    //CalcFldpW(model_entry, w, feaid);
    if (param_.flsa == true)
        IsotonicFLSA(x.data(), x.size(), param_.l2*param_.eta, 5000, w.data());
    else
        PAVA(x.data(), w.data(), x.size());
    StoreChanges(feaid, x, k);
}

void SGDUpdater::StoreChanges(feaid_t feaid, std::vector<real_t>& x
                            , std::vector<time_t>& k) {
    SGDEntry& entry = model_[feaid];
    entry.w.clear();
    real_t cur;
    for (size_t i=0; i<x.size(); i++) {
        cur = SoftThresh(x[i]);
        entry[k[i]] = cur;
    }
}

void SGDUpdater::IsotonicFLSA(real_t* x,
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
    // write model_entry
//    for (auto e : grad_entry.w) {
//        time_t tt = e.first;
//        time_t k;
//        if (!model_entry.GreastLowerBound(tt, k)) {
//            model_entry[tt] = model_entry[k];
//        }
//    }
    // don't write model_entry, but update grad_entry
    time_t ss = (model_entry.w.begin())->first;
    time_t ee = (model_entry.w.rbegin())->first;
    if (param_.concave_penalty2) {
        for (auto e : grad_entry.w) {
            time_t tt = e.first;
            auto it = model_entry.w.find(tt);
	    auto backit = it;
            CHECK_NE(it, model_entry.w.end());
            real_t val = it->second;
            auto prev = --backit; auto next = ++backit;
            // add concave_penalty2 to grad_entry
            real_t g;
            if (tt > ss) {
                g = 1.0f/(val - prev->second + param_.epsilon2);
                CHECK_GE(val, prev->second);
                grad_entry[tt] += param_.lconcave2 * g;
            }
            if (tt < ee) {
		g = -1.0f/(next->second - val + param_.epsilon2);
		CHECK_GE(next->second, val);
                grad_entry[tt] += param_.lconcave2 * g;
            }
        }
    }
    // update model_entry
    for (auto e : grad_entry.w) {
        time_t tt = e.first;
        real_t val = e.second;
        real_t temp = model_entry[tt];
        // update gradient
        model_entry[tt] -= param_.eta * val;
        // update concave_penalty1
        if (param_.concave_penalty1) {
            model_entry[tt] -= param_.eta * param_.lconcave1 *
                (1.0/(temp+param_.epsilon1));
        }
        if(!param_.debug.empty() && feaid==0) {
            val = param_.eta*val + param_.eta*param_.lconcave1*(1.0/temp+param_.epsilon1);
            val = -val;
            fprintf(debug_, "%e:%e\t", tt, val);
        }
    }
    if(!param_.debug.empty() && feaid == 0) fprintf(debug_, "\n\n\n");
}

void SGDUpdater::Update(SGDModel& grad) {
    std::vector<feaid_t> feats(grad.Size());
    size_t i=0;
    for (auto g : grad.model_map_) {
        feaid_t feaid = g.first;
        feats[i++] = feaid;
    }
#pragma omp parallel for num_threads(nthreads_)
    for (i=0; i<feats.size(); i++) {
        feaid_t feaid = feats[i];
        UpdateGradient(feaid, grad[feaid]);
        ProxOperators(feaid);
    }
}

} //namespace hazard
