/**
 * Copyright (c) 2016 by Contributors
 * \author Ziqi Liu
 */

#ifndef HAZARD_SGD_SGD_UPDATER_H_
#define HAZARD_SGD_SGD_UPDATER_H_

#include "./sgd_param.h"
#include "hazard/base.h"
#include "hazard/updater.h"
#include <omp.h>
#include <vector>
#include <map>
#include <unordered_map>

namespace hazard {

/** \brief models for each feature
    a tree allowing O(\log n) search,
    insertion and deletion.
*/
struct SGDEntry {
public:
    SGDEntry() {}
    ~SGDEntry() {}

    std::map<time_t, std::tuple<real_t,real_t,real_t,real_t>> w;
    inline size_t Size() {
        return w.size();
    }
    inline std::tuple<real_t,real_t,real_t,real_t>& operator[] (time_t t) {
        return w[t];
    }
    inline bool GreastLowerBound(time_t key, time_t& key_out) {
        auto it = w.lower_bound(key);
        if(it == w.end()) {key_out = (--it)->first; return false;}
        else if(it->first == key) {key_out = key; return true;}
        else {key_out = (--it)->first; return false;}
    }
};

struct SGDModel {
public:
    SGDModel() {}
    ~SGDModel() {}

    inline SGDEntry& operator[] (feaid_t id) {
        return model_map_[id];
    }
    inline void Clear() {
        model_map_.clear();
    }
    inline size_t Size() {
        return model_map_.size();
    }
    inline size_t Count(feaid_t feaid) {
        return model_map_.count(feaid);
    }

    std::unordered_map<feaid_t, SGDEntry> model_map_;
};

class SGDUpdater : public Updater {


public:
    SGDUpdater() {
        starttime_ = 0.0;
        endtime_ = 0.0;
    }
    virtual ~SGDUpdater() {}
    KWArgs Init(const KWArgs& kwargs) override {return kwargs;}
    KWArgs Init(const KWArgs& kwargs,
                const SGDLearnerParam& sgdlparam);
    void InitEpoch(size_t epoch);
    void Update() override {}
    void Update(SGDModel& grad);
    inline void Exist(feaid_t feaid, std::set<time_t>& ord) {
        auto it = ord.cbegin();
        SGDEntry& entry = model_[feaid];
        if(entry.Size() == 0) {
            std::get<0>(entry[*it]) = param_.init_hrate;
        }
    }
    void ResetExpGrad();
    real_t CHazardFea(feaid_t feaid, time_t censor);
    real_t CHazardFea1(feaid_t feaid, time_t censor);
    inline real_t SoftThresh(real_t w);
    void CalcFldpX(SGDEntry&, std::vector<real_t>&, std::vector<time_t>&);
    void CalcFldpW(SGDEntry&, std::vector<real_t>&, feaid_t feaid);
    void FLSA(real_t*, int, real_t, int);
    void Pool(real_t*, real_t*, int i, int j);
    void PAVA(real_t*, real_t*, int);
    void StoreChanges(feaid_t feaid, std::vector<real_t>&, std::vector<time_t>&);
    void ProxOperators(feaid_t feaid);
    void UpdateGradient(feaid_t feaid, SGDEntry& entry);
    void SaveModel(FILE* f);
    void ReadModel(std::string name);
    inline void Build(feaid_t feaid, time_t t) {
        std::get<0>(model_[feaid][t]) = param_.init_hrate;
    }
//    inline void SetHrate(real_t hr) {
//        param_.init_hrate = hr;
//    }

    time_t starttime_;
    time_t endtime_;
    /**
     *  \brief cumulative data count before current time point
     */
    std::unordered_map<feaid_t, std::map<time_t, size_t>> cumu_cnt_;
    SGDModel model_;
private:
    SGDUpdaterParam param_;
    int nthreads_;
    FILE* debug_;
}; //class SGDUpdater

} //namespace hazard

#endif
