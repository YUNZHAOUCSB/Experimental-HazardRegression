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
        n_train_ = 0;
        KWArgs remain = param_.InitAllowUnknown(kwargs);
        updater_ = (SGDUpdater*)Updater::Create(param_.updater_type);
        remain = updater_->Init(remain, param_);
        // construct feat_set_ and updater_.cumu_cnt_
        std::unordered_map<feaid_t, size_t> feat_cnt;
        std::unordered_map<feaid_t, std::set<time_t>> ordinal;
        std::unordered_set<feaid_t> feat_in_pos;
        BatchIter reader(param_.data_in, param_.data_format,
                         0, 1, param_.batch_size);
        while(reader.Next()) {
            const dmlc::RowBlock<time_t>& data = reader.Value();
            n_train_ += data.size;
            for (size_t i=0; i<data.size; i++) {
                const dmlc::Row<time_t>& d = data[i];
                uint8_t label = (uint8_t)d.label;
                if(updater_->endtime_ < d.index[0]) updater_->endtime_ = d.index[0];
                for (size_t j = 2; j<d.length; j++) {
                    feaid_t feaid = (feaid_t)d.index[j];
                    feat_cnt[feaid] += 1;
                    updater_->Build(feaid, (time_t)d.index[0]);
                    if(label) {
                        feat_in_pos.insert(feaid);
                        updater_->Build(feaid, (time_t)d.index[1]);
                    }
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
            if (f.second >= param_.feat_thresh && feat_in_pos.count(feaid)) {
                feat_set_.insert(feaid);
                //updater_->Exist(feaid, ordinal[feaid]);
            }
        }
        for(auto it = updater_->model_.model_map_.begin(); it!=updater_->model_.model_map_.end(); it++) {
            feaid_t feaid = it->first;
            SGDEntry& entry = it->second;
            if(!feat_set_.count(feaid)) {
                    entry.w.clear();
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
        loss1_.clear();
        loss1_.resize(param_.batch_size);

        if (task == "predict" && param_.model_in.empty()) {
            LOG(FATAL) << "model_in = NULL";
        }
        if(!param_.model_in.empty()) {
            ReadModel();
        }

        ind_.resize(n_train_);
        for(int i=0;i<n_train_;i++) ind_[i]=i;

        return remain;
    }

    inline real_t LogMinus(real_t cumul, real_t cumur, std::string type,time_t rt, time_t lt) {
        real_t tmp = cumur - cumul;
        //if(tmp<=param_.epsilon && type=="training") printf("rt=%e,lt=%e,cumur=%e,cumul=%e\n",rt,lt,cumur,cumul);
if(tmp==0.0) n_bad_++;
        if (tmp <= param_.epsilon) {
			//n_bad_++; 
			return cumul-std::log(1.0-std::exp(-param_.epsilon)+std::exp(-param_.epsilon)*(tmp-param_.epsilon));
		}
        return cumul - std::log(1.0f - std::exp(-tmp));
    }

    void CalcRes(const dmlc::RowBlock<time_t>& data, std::string type) {
        real_t res = 0.0f;
        real_t res_pos=0.0;
        real_t res_neg=0.0;
//#pragma omp parallel for reduction(+:res) num_threads(param_.nthreads)
        for (size_t i=0; i<data.size; i++) {
            const dmlc::Row<time_t>& d = data[i];
            uint8_t label = (uint8_t) d.label;
            time_t rt=(time_t)d.index[0]; time_t lt=(time_t)d.index[1];
            if (label) {
                if(type=="validation") n_val_pos_++;
                else if(type=="training") n_train_pos_++;
                else n_test_++;
                res += LogMinus(std::get<1>(loss_[i]), std::get<0>(loss_[i]), type, rt,lt);
            } else {
                if(type=="validation") n_val_neg_++;
                else if(type=="training") n_train_neg_++;
                else n_test_++;
                res += std::get<0>(loss_[i]);
                res_neg+=std::get<0>(loss_[i]);
            }
        }
        res_pos=res-res_neg;
        if(type == "training") {train_loss_ += res; train_pos_loss_+=res_pos;
            train_neg_loss_+=res_neg;}
        else if(type=="validation") {val_loss_ += res; val_pos_loss_+=res_pos; val_neg_loss_+=res_neg;}
        else test_loss_ +=res;
    }

    void PrintRes(size_t epoch) {
	std::cout << std::fixed;
	std::cout << std::setprecision(6);
        std::cout << "Iter\t" << epoch << "\tTraining\t"
                  << train_loss_*1.0/(real_t)n_train_ << "\tValidation\t"
                  << val_loss_*1.0/(real_t)n_val_
				  << "\tTest\t" << test_loss_*1.0/(real_t)n_test_
                  << "\tParameters\t" << CheckSparsity()
				  << "\tval_pos\t" << val_pos_loss_*1.0/(real_t)n_val_pos_
				  << "\tval_neg\t" << val_neg_loss_*1.0/(real_t)n_val_neg_
//            << "\ttrain_pos\t" << train_pos_loss_*1.0/(real_t)n_train_pos_
//            << "\ttrain_neg\t" << train_neg_loss_*1.0/(real_t)n_train_neg_
            << "\tbad\t" << n_bad_
                  << "\n";
    }

    void InitEpoch(size_t epoch) {
        n_val_ = 0; n_test_ = 0;
        train_loss_ = 0.0f; val_loss_ = 0.0f; test_loss_ = 0.0f;
		n_train_pos_=0;n_val_pos_=0;n_train_neg_=0;n_val_neg_=0;
		train_pos_loss_=0.0;val_pos_loss_=0.0;train_neg_loss_=0.0;val_neg_loss_=0.0;
		n_bad_ = 0;
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

    size_t CheckSparsity() {
      size_t c=0;
      for(auto e : updater_->model_.model_map_) {
        SGDEntry& entry = e.second;
        if(entry.Size()<=1) continue;
        auto it=entry.w.begin();
        real_t prev = std::get<0>(it->second);
        if(prev!=0.0) c++;
        it++;
        for (;it!=entry.w.end();it++) {
            if(std::get<0>(it->second)!=prev) {c++;prev=std::get<0>(it->second);}
        }
      }
      return c;
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
            //RunEpoch(epoch, "test", task);
            Complete(epoch, task);
        }
    }

    void ResetExpGrad() {
        updater_->ResetExpGrad();
    }
    void SVRG(const std::string& filename) {
        ResetExpGrad();
		BatchIter reader(filename, param_.data_format,
				0, 1, param_.batch_size);
		while(reader.Next()) {
			const dmlc::RowBlock<time_t>& data = reader.Value();
			CalcLoss1(data);
			AveGrad(data);
		}
        for(auto it=updater_->model_.model_map_.begin();it!=updater_->model_.model_map_.end();it++) {
            SGDEntry& entry = it->second;
            for(auto it1=entry.w.begin();it1!=entry.w.end();it1++) {
                std::get<2>(it1->second) = std::get<0>(it1->second);
            }
        }
    }

    void RunEpoch(uint32_t epoch, const std::string& type, const std::string& task) {
//        std::string filename = "training" == type ? param_.data_in :
//            param_.val_data;
        std::string filename;
        if(type == "training") filename = param_.data_in;
        else if(type == "validation") filename = param_.val_data;
        else filename = param_.test_data;
        if(filename.empty()) return;
		if((epoch%2==0) && task == "train" && type == "training") {
		//if(task == "train" && type == "training") {
			SVRG(filename);
			BatchIter reader(filename, param_.data_format,
					0, 1, param_.batch_size);
            while(reader.Next()) {
                const dmlc::RowBlock<time_t>& data = reader.Value();
                CalcLoss(data);
                CalcRes(data, type);
                if ("training" == type && task == "train") {
                    CalcVRGrad(data);
                    updater_->Update(gradients_);
                }
            }
        }
//        if("training"==type) {
//            while(i<param_.batch_size) {
//                dmlc::data::RowBlockContainer<time_t> bl;
//                for(int k=i;k<j;k++) {
//                    bl.Push(data_[ind_[k]]);
//                }
//                CalcLoss(bl.GetBlock());
//                CalcRes(bl.GetBlock(),type);
//                CalcGrad(bl.GetBlock());
//                updater_->Update(gradients_);
//                i=j;j=std::min(i+1024,(int)param_.batch_size);
//            }
//        }
        else {
			BatchIter reader(filename, param_.data_format,
					0, 1, param_.batch_size);
            while(reader.Next()) {
                const dmlc::RowBlock<time_t>& data = reader.Value();
				if (type == "validation") n_val_ += data.size;
                if (type == "test") n_test_ += data.size;
                CalcLoss(data);
                CalcRes(data, type);
                if ("training" == type && task == "train") {
                    CalcGrad(data);
                    updater_->Update(gradients_);
                }
            }
        }
    }

    //std::pair<real_t, real_t> GenGrad(uint8_t label, size_t x);
    void CalcLoss(const dmlc::RowBlock<time_t>& data);
    void CalcLoss1(const dmlc::RowBlock<time_t>& data);
    void CalcLoss2(const dmlc::RowBlock<time_t>& data);
    void CalcGrad(const dmlc::RowBlock<time_t>& data);
    void CalcVRGrad(const dmlc::RowBlock<time_t>& data);
    void AveGrad(const dmlc::RowBlock<time_t>& data);

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
    std::vector<std::tuple<real_t, real_t>> loss1_;
    /**
     *  \brief whether a feature in active set
     */
    dmlc::RowBlock<time_t> data_;
    std::unordered_set<feaid_t> feat_set_;
    std::vector<int> ind_;
    real_t train_loss_;
    real_t val_loss_;
    real_t test_loss_;
    size_t n_train_;
    size_t n_val_;
    size_t n_test_;
	real_t train_pos_loss_;
	real_t val_pos_loss_;
	size_t n_train_pos_;
	size_t n_val_pos_;
	real_t train_neg_loss_;
	real_t val_neg_loss_;
	size_t n_train_neg_;
	size_t n_val_neg_;
	size_t n_bad_;
    size_t cur_data_;
};  // class SGDLearner

}
#endif
