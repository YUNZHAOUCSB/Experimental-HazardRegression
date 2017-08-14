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
    //param_.eta = param_.alpha;
}

void SGDUpdater::ReadModel(std::string name) {
    for (auto it=model_.model_map_.begin(); it!=model_.model_map_.end(); it++) {
        SGDEntry& entry = it->second;
        for (auto it1=entry.w.begin();it1!=entry.w.end();it1++) {
            std::get<0>(it1->second) = 0.0;
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
            auto aa=entry.w.lower_bound(tid);
            time_t aat=aa->first;time_t bbt=-1.0;
            aa--;if(aa!=entry.w.end()) bbt=aa->first;
            time_t tt;
            if(std::abs(aat-tid)<std::abs(bbt-tid)) tt=aat;
            else tt=bbt;
            std::get<0>(entry[tt]) = val;
//            auto pit = entry.w.lower_bound(tid);
//            tid = pit->first;
//            if(pit!=entry.w.end()) entry[tid] = val;
        }
    }
    for(auto it=model_.model_map_.begin(); it!=model_.model_map_.end(); it++) {
        SGDEntry& entry = it->second;
        real_t prev=0.0;
        for (auto it1 = entry.w.begin(); it1!=entry.w.end(); it1++) {
            if(std::get<0>(it1->second) == 0.0) {
                std::get<0>(it1->second) = prev;
            }
			prev = std::get<0>(it1->second);
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
          //if(o.second != 0.0)
            fprintf(f, "\t%e:%e", o.first, std::get<0>(o.second));
        }
        fprintf(f, "\n");
    }
}

real_t SGDUpdater::CHazardFea1(feaid_t feaid, time_t censor) {
    SGDEntry& entry = model_[feaid];
    if(entry.Size()<=1) return 0.0;
    real_t hcumulative = 0.0;
    time_t k;
    auto it = entry.w.begin();
    auto it_next = entry.w.upper_bound(it->first);
    entry.GreastLowerBound(censor, k);
    if(it->first >= censor) return 0.0;
    for (; it_next!=entry.w.cend() && it_next->first <= k; it++, it_next++) {
        hcumulative += (it_next->first - it->first)*std::get<2>(it->second);
//		printf("%e += (%e - %e) * %e\n", hcumulative, it_next->first, it->first, std::get<2>(it->second));
    }
    hcumulative += (censor - it->first) * std::get<2>(it->second);
//	printf("%e += (%e - %e) * %e\n", hcumulative, censor, it->first, std::get<2>(it->second));
//	printf("\n\n");

//	for(auto e : model_.model_map_) {
//		for(auto e1 : e.second.w) {
//			printf("%e\n", std::get<2>(e1.second));
//		}
//	}
//	printf("\n---------------------\n");
    return hcumulative;
}

real_t SGDUpdater::CHazardFea(feaid_t feaid, time_t censor) {
    SGDEntry& entry = model_[feaid];
    if(entry.Size()<=1) return 0.0;
    real_t hcumulative = 0.0;
    time_t k;
    auto it = entry.w.begin();
    auto it_next = entry.w.upper_bound(it->first);
    entry.GreastLowerBound(censor, k);
    if(it->first >= censor) return 0.0;
    for (; it_next!=entry.w.cend() && it_next->first <= k; it++, it_next++) {
        hcumulative += (it_next->first - it->first)*std::get<0>(it->second);
    }
    hcumulative += (censor - it->first) * std::get<0>(it->second);
    return hcumulative;
}

void SGDUpdater::ResetExpGrad() {
    for (auto it=model_.model_map_.begin();it!=model_.model_map_.end();it++) {
        SGDEntry& entry = it->second;
        for(auto it1=entry.w.begin();it1!=entry.w.end();it1++) {
            std::get<1>(it1->second) = 0.0;
        }
    }
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
        x[i++] = std::get<0>(e.second);
    }
}
    void SGDUpdater::CalcFldpW(SGDEntry& model_entry, std::vector<real_t>& w, feaid_t feaid) {
    size_t i = 0;
    auto it = model_entry.w.begin();
    auto it_next = model_entry.w.upper_bound(it->first);
    for (; it_next!=model_entry.w.cend(); it++, it_next++) {
        w[i++] = cumu_cnt_[feaid][it_next->first] - cumu_cnt_[feaid][it->first];
    }
    w[i] = cumu_cnt_[feaid].size() - cumu_cnt_[feaid][it->first];
    CHECK_GT(w[i],0);
}

void SGDUpdater::ProxOperators(feaid_t feaid) {
    SGDEntry& model_entry = model_[feaid];
    CHECK_GT(model_entry.Size(),1);
    std::vector<real_t> x(model_entry.Size());
    std::vector<time_t> k(x.size());
    //std::vector<real_t> w(x.size(),1.0);
    CalcFldpX(model_entry, x, k);

    if(param_.flsa == true) {
//        real_t tmp = 0.0;
//        auto it=model_entry.w.begin();
//        auto it_next=model_entry.w.begin();
//        it_next++;
//        for(;it_next!=model_entry.w.end();it_next++,it++) {
//          real_t z=std::get<3>(it->second);
//          tmp += param_.alpha * 1.0/(param_.beta + std::sqrt(z));
//        }
//        tmp=tmp*1.0/(model_entry.Size()-1.0);
//        FLSA(x.data(), x.size()-1, tmp*param_.l2, 5000);

        FLSA(x.data(), x.size()-1, param_.eta*param_.l2, 5000);
    }
    //CalcFldpW(model_entry, w, feaid);
//    if (param_.flsa == true)
//        IsotonicFLSA(x.data(), x.size()-1, param_.l2*param_.eta, 5000, w.data());
//    else
//        PAVA(x.data(), w.data(), x.size()-1);
    StoreChanges(feaid, x, k);
}

void SGDUpdater::StoreChanges(feaid_t feaid, std::vector<real_t>& x
                            , std::vector<time_t>& k) {
    SGDEntry& entry = model_[feaid];
    //entry.w.clear();
    real_t cur=0;
    size_t i=0;
    CHECK_GT(x.size(),1);
    for (; i<x.size()-1; i++) {
        cur = SoftThresh(x[i]);
        std::get<0>(entry[k[i]]) = cur;
    }
    std::get<0>(entry[k[i]]) = cur;
}
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
    };

    class Msg {
    public:
        std::vector<MsgElt> buf_;
        int start_idx_;
        int len_;
        MsgElt init_knot_;
        MsgElt end_knot_;
    };

void InitMsg(Msg * msg, int init_sz, real_t lin, real_t quad,
             real_t lambda2) {
  msg->len_ = 2;
  int i = msg->start_idx_ = init_sz / 2;

  msg->buf_ = std::vector<MsgElt>(init_sz);

  msg->buf_[i].x_ = -1.0/0.0;
  msg->buf_[i].sgn_ = true;
  msg->buf_[i].lin_ = lin;
  msg->buf_[i].quad_ = quad;

  msg->buf_[i+1].x_ = 1.0/0.0;
  msg->buf_[i+1].sgn_ = false;
  msg->buf_[i+1].lin_ = lin;
  msg->buf_[i+1].quad_ = quad;

  msg->init_knot_.x_ = -1.0/0.0;
  msg->init_knot_.sgn_ = true;
  msg->init_knot_.lin_ = lambda2;
  msg->init_knot_.quad_ = 0.0;

  msg->end_knot_.x_ = 1.0/0.0;
  msg->end_knot_.sgn_ = false;
  msg->end_knot_.lin_ = -lambda2;
  msg->end_knot_.quad_ = 0.0;
}
void UpdMsg(Msg * msg, real_t lin, real_t quad, real_t lambda2,
            real_t * bp) {
  std::vector<MsgElt>& buf = msg->buf_;
  int buf_sz = buf.size();
  int last_elt = msg->start_idx_ + msg->len_ - 1;

  real_t lin_left = lin, quad_left = quad;
  int new_knot_start = -3;

  for (int knot_idx = msg->start_idx_, m = msg->len_; m; --m, ++knot_idx) {
    if (knot_idx == last_elt) {
      printf("numerical error.  failed to find back-pointer segment\n");
    }

    const MsgElt& k = buf[knot_idx];
    real_t x1 = buf[knot_idx + 1].x_;
    if (k.sgn_) {
      lin_left += k.lin_;
      quad_left += k.quad_;
    } else {
      lin_left -= k.lin_;
      quad_left -= k.quad_;
    }

    real_t hit_x = (lambda2 - lin_left) / (2.0 * quad_left);
    if (hit_x < x1) {
      // place a knot here
      new_knot_start = knot_idx - 1;

      if (new_knot_start < 0) {
        printf("knot index out of range\n");
      }

      bp[0] = hit_x;
      break;
    }
  }

  real_t lin_right = lin, quad_right = quad;
  int new_knot_end = -2;
  int end_idx = msg->start_idx_ + msg->len_ - 1;

  double neg_lam2 = -lambda2;

  for (int knot_idx = end_idx, m = msg->len_; m; --m, --knot_idx) {
    if (knot_idx == msg->start_idx_) {
      printf("numerical error.  failed to find back-pointer segment\n");
    }
    const MsgElt& k = buf[knot_idx];
    real_t x1 = buf[knot_idx - 1].x_;

    if (k.sgn_) {
      lin_right -= k.lin_;
      quad_right -= k.quad_;
    } else {
      lin_right += k.lin_;
      quad_right += k.quad_;
    }

    real_t hit_x = (neg_lam2 - lin_right) / (2.0 * quad_right);
    if (hit_x > x1) {
      // place a knot here
      new_knot_end = knot_idx + 1;
      bp[1] = hit_x;
      break;
    }
  }

  // update the message and add the new knots
  MsgElt * k0 = &buf[new_knot_start];
  k0->x_ = -1.0/0.0;
  k0->sgn_ = true;
  k0->lin_ = lambda2;
  k0->quad_ = 0.0;
  msg->init_knot_ = *k0;

  k0 = &buf[new_knot_start + 1];
  k0->x_ = bp[0];
  k0->sgn_ = true;
  k0->lin_ = lin_left - lambda2;
  k0->quad_ = quad_left;

  k0 = &buf[new_knot_end];
  k0->x_ = 1.0/0.0;
  k0->sgn_ = false;
  k0->lin_ = -lambda2;
  k0->quad_ = 0.0;
  msg->end_knot_ = *k0;

  k0 = &buf[new_knot_end-1];
  k0->x_ = bp[1];
  k0->sgn_ = false;
  k0->lin_ = lin_right + lambda2;
  k0->quad_ = quad_right;

  msg->start_idx_ = new_knot_start;
  msg->len_ = 1 + new_knot_end - new_knot_start;
}
void UpdMsgOpt(Msg * msg, real_t lambda2, real_t lin, real_t quad,
               real_t * bp) {
  // Assumes that msg->buf_[0] and msg->buf_[1] contain the end-point knots and
  // that we do not need to create them

    std::vector<MsgElt>& buf = msg->buf_;
  //real_t quad = -0.5;

  buf[msg->start_idx_].x_ = -1.0/0.0;
  buf[msg->start_idx_ + msg->len_ - 1].x_ = 1.0/0.0;

  real_t lin_left = lin, quad_left = quad;
  int new_knot_start = -3;

  int knot_idx = msg->start_idx_;
  const MsgElt * k = &(msg->init_knot_);
  real_t x1 = buf[knot_idx + 1].x_;
  if (k->sgn_) {
    lin_left += k->lin_;
    quad_left += k->quad_;
  } else {
    lin_left -= k->lin_;
    quad_left -= k->quad_;
  }

  real_t hit_x = (lambda2 - lin_left) / (2.0 * quad_left);
  if (hit_x < x1) {
    // place a knot here
    new_knot_start = knot_idx - 1;

    bp[0] = hit_x;
  } else {
    ++knot_idx;
    k = &buf[knot_idx];
    for (; ; ++knot_idx, ++k) {
      x1 = k[1].x_;
      if (k->sgn_) {
        lin_left += k->lin_;
        quad_left += k->quad_;
      } else {
        lin_left -= k->lin_;
        quad_left -= k->quad_;
      }

      hit_x = (lambda2 - lin_left) / (2.0 * quad_left);
      if (hit_x < x1) {
        // place a knot here
        new_knot_start = knot_idx - 1;

        bp[0] = hit_x;
        break;
      }
    }
  }

  real_t lin_right = lin, quad_right = quad;
  int new_knot_end = -2;
  int end_idx = msg->start_idx_ + msg->len_ - 1;

  double neg_lam2 = -lambda2;

  knot_idx = end_idx;
  k = &(msg->end_knot_);
  x1 = buf[knot_idx-1].x_;

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
  } else {
    --knot_idx;
    k = &buf[knot_idx];
    for (; ; --knot_idx, --k) {
      x1 = k[-1].x_;

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
        break;
      }
    }
  }

  MsgElt * k0 = &buf[new_knot_start + 1];
  k0->x_ = bp[0];
  k0->sgn_ = true;
  k0->lin_ = lin_left - lambda2;
  k0->quad_ = quad_left;

  k0 = &buf[new_knot_end-1];
  k0->x_ = bp[1];
  k0->sgn_ = false;
  k0->lin_ = lin_right + lambda2;
  k0->quad_ = quad_right;

  msg->start_idx_ = new_knot_start;
  msg->len_ = 1 + new_knot_end - new_knot_start;
}
void ShiftMsg(Msg * msg, int check_freq) {
  if (msg->len_ > msg->buf_.size() - 20 * check_freq) {
      std::vector<MsgElt> new_buf(msg->buf_.size() * 3);
      std::vector<MsgElt> * old_buf = &msg->buf_;

    int new_start = (new_buf.size() / 4);
    int old_start = msg->start_idx_;

    for (int k = 0; k < msg->len_; ++k) {
      new_buf[k + new_start] = (*old_buf)[k + old_start];
    }
    msg->buf_.swap(new_buf);
    msg->start_idx_ = new_start;
  }

  if (msg->start_idx_ < 5 * check_freq) {
    int new_start = ((msg->buf_.size() - msg->len_)/2);
    int old_start = msg->start_idx_;
    std::vector<MsgElt> * buf = &msg->buf_;

    for (int k = msg->len_ - 1; k >= 0; --k) {
      (*buf)[k + new_start] = (*buf)[k + old_start];
    }
    msg->start_idx_ = new_start;

  } else if (msg->start_idx_ + msg->len_ > msg->buf_.size() - 5 * check_freq) {
    int new_start = ((msg->buf_.size() - msg->len_)/2);
    int old_start = msg->start_idx_;
    std::vector<MsgElt> * buf = &msg->buf_;

    for (int k = 0; k < msg->len_; ++k) {
      (*buf)[k + new_start] = (*buf)[k + old_start];
    }
    msg->start_idx_ = new_start;
  }
}
double MaxMsg(const Msg& msg, double * max_val) {
    const std::vector<MsgElt>& buf = msg.buf_;
  int buf_sz = buf.size();
  real_t lin_left = 0.0, quad_left = 0.0;

  int last_idx = msg.start_idx_ + msg.len_ - 1;

  for (int knot_idx = msg.start_idx_, m = msg.len_; m; --m, ++knot_idx) {
    bool end_knot = (knot_idx == last_idx);
    const MsgElt& k = (knot_idx == msg.start_idx_) ?
                       msg.init_knot_ :
                       (end_knot ? msg.end_knot_ : buf[knot_idx]);
    //double x0 = k.x_;
    real_t x1 = (knot_idx == last_idx-1) ? msg.end_knot_.x_ : buf[knot_idx + 1].x_;

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
      if (max_val) {
        *max_val = hit_x * lin_left + hit_x * hit_x * quad_left;
      }
      return(hit_x);
    }
  }

  printf("FLSA::MaxMsg : failed to maximize message\n");
  return -1.0/0.0;
}
void BackTrace(double * x_hat, int seq_len, real_t * back_pointers,
               real_t last_msg_max) {
  real_t z = x_hat[seq_len - 1] = last_msg_max;
  double * x0 = x_hat + seq_len - 2;

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
void SGDUpdater::FLSA(real_t* x, int seq_len, real_t lambda2, int init_buf_sz) {
  if (seq_len < 2) {
    return;
  }
  real_t back_pointers[seq_len*2];

  int check_freq = 40;

  if (init_buf_sz < 30 * check_freq) {
    printf("FLSA_RI : initial buffer size is too small\n");
  }

  Msg msg;
  InitMsg(&msg, init_buf_sz, 0.0, 0.0, lambda2);

  int max_msg_sz = 0;

  double * bp = back_pointers + 2;
  double * x0 = x + 1;
  int check_msg = check_freq - 1;

  UpdMsg(&msg, x[0], -0.5, lambda2, back_pointers);
  for (int j = 1; j < seq_len; ++j, bp += 2, ++x0, --check_msg) {
      UpdMsgOpt(&msg, lambda2, *x0, -0.5, bp);
      if (!check_msg) {
          check_msg = check_freq - 1;
          ShiftMsg(&msg, check_freq);
      }
  }

  real_t last_msg_max = MaxMsg(msg, NULL);

  BackTrace(x, seq_len, back_pointers, last_msg_max);
}

void SGDUpdater::UpdateGradient(feaid_t feaid, SGDEntry& grad_entry) {
    SGDEntry& model_entry = model_[feaid];
    // don't write model_entry, but update grad_entry
    time_t ss = (model_entry.w.begin())->first;
    auto tmpit = model_entry.w.rbegin();
    time_t ee = (++tmpit)->first;
    CHECK_NE(tmpit, model_entry.w.rend());
    if (param_.concave_penalty2) {
        for (auto e : grad_entry.w) {
            time_t tt = e.first;
            auto it = model_entry.w.find(tt);
            auto backit = it;
            CHECK_NE(it, model_entry.w.end());
            real_t val = std::get<0>(it->second);
            auto prev = --backit; backit=it; auto next = ++backit;
            // add concave_penalty2 to grad_entry
            real_t g;
            if (tt > ss && tt <= ee) {
                real_t sgn = val>=std::get<0>(prev->second)?1.0:-1.0;
                if(val==std::get<0>(prev->second)) sgn = 0.0;
                g = sgn*1.0f/(std::abs(val - std::get<0>(prev->second)) + param_.epsilon2);
                std::get<0>(grad_entry[tt]) += param_.lconcave2 * g;
            }
            if (tt < ee) {
                real_t sgn = val>std::get<0>(next->second)?1.0:-1.0;
                if(val==std::get<0>(next->second)) sgn = 0.0;
                g = sgn*1.0f/(std::abs(std::get<0>(next->second) - val) + param_.epsilon2);
                std::get<0>(grad_entry[tt]) += param_.lconcave2 * g;
            }
        }
    }
    // update model_entry
    for (auto e : grad_entry.w) {
        time_t tt = e.first;
        real_t val = std::get<0>(e.second);
        std::tuple<real_t,real_t,real_t,real_t>& a = model_entry[tt];
        real_t temp = std::get<0>(a);
        real_t& z=std::get<3>(a);
        z+=val*val;
        // update gradient
        CHECK_GT(z,0.0);
        //std::get<0>(a) -= param_.alpha * 1.0/(param_.beta+std::sqrt(z)) * val;
        std::get<0>(a) -= param_.eta * val;
        // update concave_penalty1
        if (param_.concave_penalty1) {
            std::get<0>(model_entry[tt]) -= param_.eta * param_.lconcave1 *
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
