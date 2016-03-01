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

    std::map<time_t, real_t> w;
    inline size_t Size() {
        return w.size();
    }
    inline real_t& operator[] (time_t t) {
        return w[t];
    }
    inline bool GreastLowerBound(time_t key, time_t& key_out) {
        auto it = w.lower_bound(key);
        if(it->first == key) {key_out = key; return true;}
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
    struct MsgElt {
        // the location of the knot
        real_t x_;
        // the sign variable that tells us
        // whether this was a left or right end-point of the
        // segment
        bool sgn_;
        // a delta which can be used to reconstruct the function
        // if we move from the first knot to the last or from
        // the last to the first
        real_t lin_;
        real_t quad_;
        real_t const_;
    };

    class Msg {
    public:
        std::vector<MsgElt> buf_;
        int start_idx_;
        int len_;
        MsgElt init_knot_;
        MsgElt end_knot_;

        void BackTrace(real_t * x_hat, size_t seq_len, real_t * back_pointers,
                       real_t last_msg_max) {
            real_t z = x_hat[seq_len - 1] = last_msg_max;
            real_t * x0 = x_hat + seq_len - 2;

            real_t * bp = back_pointers + (2 * (seq_len - 2));
            for (int idx = seq_len-1; idx; --idx, bp -= 2, --x0) {
                if (z < bp[0]) {
                    z = *x0 = bp[0];
                } else if (z > bp[1]) {
                    z = *x0 = bp[1];
                } else {
                    *x0 = z;
                }
            }
        }
        void FirstMsg(size_t init_sz, real_t lin, real_t quad,
                      real_t lambda2, real_t * bp) {
            len_ = 3;
            start_idx_ = (int)init_sz / 2;

            buf_.resize(init_sz);

            bp[0] = -1.0/0.0;
            bp[1] = (-lambda2 - lin) / (2.0 * quad);

            // update the message and add the new knots
            MsgElt * k0 = &buf_[start_idx_];
            k0->x_ = -1.0/0.0;
            k0->sgn_ = true;
            k0->lin_ = lin;
            k0->quad_ = quad;
            k0->const_ = 0.0;
            init_knot_ = *k0;

            real_t const_shift = bp[1] * lin + bp[1] * bp[1] * quad + lambda2 * bp[1];
            k0 = &buf_[start_idx_+2];
            k0->x_ = 1.0/0.0;
            k0->sgn_ = false;
            k0->lin_ = -lambda2;
            k0->quad_ = 0.0;
            k0->const_ = const_shift;
            end_knot_ = *k0;


            k0 = &buf_[start_idx_+1];
            k0->x_ = bp[1];
            k0->sgn_ = false;
            k0->lin_ = lin + lambda2;
            k0->const_ = 0.0;
            k0->quad_ = quad;

            len_ = 3;
        }
        void UpdMsgOpt(real_t lin, real_t quad, real_t lambda2,
                       real_t * bp) {
            // Assumes that buf_[0] and buf_[1] contain the end-point knots and
            // that we do not need to create them.

            buf_[start_idx_].x_ = -1.0/0.0;
            buf_[start_idx_ + len_ - 1].x_ = 1.0/0.0;

            bp[0] = -1.0/0.0;

            real_t lin_right = lin, quad_right = quad;
            int new_knot_end = -2;
            int end_idx = start_idx_ + len_ - 1;

            real_t neg_lam2 = -lambda2;

            int start_idx = start_idx_;
            int knot_idx = end_idx;
            MsgElt * k = &(end_knot_);
            real_t x1 = (knot_idx == start_idx+1)
                ? init_knot_.x_
                : buf_[knot_idx-1].x_;
            real_t cur_const = end_knot_.const_;

            if (k->sgn_) {
                lin_right -= k->lin_;
                quad_right -= k->quad_;
            } else {
                lin_right += k->lin_;
                quad_right += k->quad_;
            }

            real_t hit_x = (neg_lam2 - lin_right) / (2.0 * quad_right);
            if (hit_x > x1) {
                // place a knot here
                new_knot_end = knot_idx + 1;
                bp[1] = hit_x;
            } else {
                --knot_idx;

                for (k = &buf_[knot_idx]; ; --knot_idx, --k) {
                    x1 = (knot_idx == start_idx+1) ? init_knot_.x_ : k[-1].x_;

                    if (k->sgn_) {
                        lin_right -= k->lin_;
                        quad_right -= k->quad_;
                    } else {
                        lin_right += k->lin_;
                        quad_right += k->quad_;
                    }

                    hit_x = (neg_lam2 - lin_right) / (2.0 * quad_right);
                    if (hit_x > x1) {
                        // place a knot here
                        new_knot_end = knot_idx + 1;
                        bp[1] = hit_x;
                        cur_const = k->const_;
                        break;
                    }
                }
            }

            init_knot_.lin_ += lin;
            init_knot_.quad_ += quad;

            MsgElt * k0 = &buf_[new_knot_end-1];
            k0->x_ = bp[1];
            k0->sgn_ = false;
            k0->lin_ = lin_right + lambda2;
            k0->quad_ = quad_right;
            k0->const_ = cur_const;

            end_knot_.const_ = cur_const + lambda2 * bp[1] + bp[1] * lin_right
                + bp[1] * bp[1] * quad_right;
            //real_t const_shift = bp[1] * lin + bp[1] * bp[1] * quad + lambda2 * bp[1];

            len_ = 1 + new_knot_end - start_idx_;
        }

        void ShiftMsg(int check_freq) {
            if (len_ > (int)buf_.size() - 20 * check_freq) {
                std::vector<MsgElt> new_buf(buf_.size() * 3);
                std::vector<MsgElt> * old_buf = &buf_;

                int new_start = (new_buf.size() / 4);
                int old_start = start_idx_;

                for (int k = 0; k < len_; ++k) {
                    new_buf[k + new_start] = (*old_buf)[k + old_start];
                }
                buf_.swap(new_buf);
                start_idx_ = new_start;
            }

            if (start_idx_ < 5 * check_freq) {
                //Rprintf("shift message 2\n");
                int new_start = ((buf_.size() - len_)/2);
                int old_start = start_idx_;
                std::vector<MsgElt> * buf = &buf_;

                for (int k = len_ - 1; k >= 0; --k) {
                    (*buf)[k + new_start] = (*buf)[k + old_start];
                }
                start_idx_ = new_start;

            } else if (start_idx_ + len_ > (int)buf_.size() - 5 * check_freq) {
                //Rprintf("shift message 3\n");
                int new_start = ((buf_.size() - len_)/2);
                int old_start = start_idx_;
                std::vector<MsgElt> * buf = &buf_;

                for (int k = 0; k < len_; ++k) {
                    (*buf)[k + new_start] = (*buf)[k + old_start];
                }
                start_idx_ = new_start;
            }
        }
        real_t MaxMsg() {
            const std::vector<MsgElt>& buf = buf_;
            real_t lin_left = 0.0, quad_left = 0.0;

            int last_idx = start_idx_ + len_ - 1;

            for (int knot_idx = start_idx_, m = len_; m; --m, ++knot_idx) {
                bool end_knot = (knot_idx == last_idx);
                const MsgElt& k = (knot_idx == start_idx_) ?
                    init_knot_ :
                    (end_knot ? end_knot_ : buf[knot_idx]);
                real_t x1 = (knot_idx == last_idx-1) ? end_knot_.x_ : buf[knot_idx + 1].x_;

                if (k.sgn_) {
                    lin_left += k.lin_;
                    quad_left += k.quad_;
                } else {
                    lin_left -= k.lin_;
                    quad_left -= k.quad_;
                }

                if (quad_left == 0.0) {
                    continue;
                }

                real_t hit_x = -lin_left / (2.0 * quad_left);
                if (hit_x < x1) {
                    return(hit_x);
                }
            }

            LOG(ERROR) << "FLSA::MaxMsg : failed to maximize message\n";
            return -1.0/0.0;
        }
    }; // class Msg

public:
    SGDUpdater() {}
    virtual ~SGDUpdater() {}
    KWArgs Init(const KWArgs& kwargs) override {return kwargs;}
    KWArgs Init(const KWArgs& kwargs,
                const SGDLearnerParam& sgdlparam);
    void InitEpoch(size_t epoch);
    void Update() override {}
    void Update(SGDModel& grad);
    inline void Exist(feaid_t feaid) {
        SGDEntry& entry = model_[feaid];
        if(entry.Size() == 0)
            entry[starttime_] = param_.init_hrate;
    }
    inline bool NonExist(feaid_t feaid) {
        return !model_.Count(feaid);
    }
    std::pair<real_t, real_t> CHazardFea(feaid_t feaid,
                                         time_t censor);
    inline real_t SoftThresh(real_t w);
    void CalcFldpX(SGDEntry&, std::vector<real_t>&, std::vector<time_t>&);
    void CalcFldpW(SGDEntry&, std::vector<real_t>&);
    void UpdateGradient(feaid_t feaid, SGDEntry& entry);
    void IsotonicDp(real_t*, size_t, real_t, size_t, real_t*);
    void StoreChanges(feaid_t feaid, std::vector<real_t>&, std::vector<time_t>&);
    void FLSAIsotonic(feaid_t feaid);
    void SaveModel(FILE* f);

    time_t starttime_;
    /**
     *  \brief cumulative data count before current time point
     */
    std::unordered_map<time_t, size_t> cumu_cnt_;
private:
    SGDModel model_;
    SGDUpdaterParam param_;
    int nthreads_;
}; //class SGDUpdater

} //namespace hazard

#endif
