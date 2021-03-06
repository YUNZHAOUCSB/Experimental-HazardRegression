/**
 * Copyright (c) 2016 by Contributors
 * \author Ziqi Liu
 */

#ifndef HAZARD_SGD_SGD_PARAM_H_
#define HAZARD_SGD_SGD_PARAM_H_
#include "dmlc/parameter.h"
#include "hazard/base.h"

namespace hazard {
    struct SGDUpdaterParam : public dmlc::Parameter<SGDUpdaterParam> {
        /** \brief the l1 regularizer for \|w\|_1 */
        real_t l1;
        /** \brief the l1 regularizer for \|D^{(1)}w^\top\|_1 */
        real_t l2;
        /** \brief the penalty for concave part \log(w_{ij}) */
        real_t lconcave1;
        /** \brief the penalty for \log(Dw^top_{ij}) */
        real_t lconcave2;
        /** \log (|w| + \epsilon1) */
        real_t epsilon1;
        /** \log (|Dw| + \epsilon2) */
        real_t epsilon2;
        /** \brief init learning rate */
        real_t lr;
        /** \brief learning rate at each epoch */
        real_t eta;
        /** \brief learning rate decay */
        real_t decay;
        /** \brief init attack hazard rate */
        real_t init_hrate;
        /** \brief concave_penalties for each
            coordinate to allow sharper
            infection rates */
        bool concave_penalty1;
        bool concave_penalty2;
        /** whether do flsa */
        bool flsa;
        std::string debug;
        real_t alpha;
        real_t beta;

        DMLC_DECLARE_PARAMETER(SGDUpdaterParam) {
            DMLC_DECLARE_FIELD(l1).set_range(0.0f,1e5f).set_default(1);
            DMLC_DECLARE_FIELD(l2).set_range(0.0f,1e5f).set_default(1);
            DMLC_DECLARE_FIELD(lconcave1).set_range(0.0f,1e5f).set_default(1);
            DMLC_DECLARE_FIELD(lconcave2).set_range(0.0f,1e5f).set_default(1);
            DMLC_DECLARE_FIELD(epsilon1).set_range(1e-5f,1e2f).set_default(1);
            DMLC_DECLARE_FIELD(epsilon2).set_range(1e-5f,1e2f).set_default(1);
            DMLC_DECLARE_FIELD(lr).set_range(0.0f,1e1f).set_default(1);
            DMLC_DECLARE_FIELD(eta).set_range(0.0f,1e1f).set_default(1);
            DMLC_DECLARE_FIELD(decay).set_range(0.0f,1.0f).set_default(1);
            DMLC_DECLARE_FIELD(init_hrate).set_default(1e-1);
            DMLC_DECLARE_FIELD(concave_penalty1).set_default(true);
            DMLC_DECLARE_FIELD(concave_penalty2).set_default(true);
            DMLC_DECLARE_FIELD(flsa).set_default(false);
            DMLC_DECLARE_FIELD(debug).set_default("");
            DMLC_DECLARE_FIELD(alpha).set_default(1e-1);
            DMLC_DECLARE_FIELD(beta).set_default(1e0);
        }
    }; //class SGDUpdaterParam

    struct SGDLearnerParam : public dmlc::Parameter<SGDLearnerParam> {
        /** \brief input data */
        std::string data_in;
        /** \brief evaluation data */
        std::string val_data;
        /** \brief test data */
        std::string test_data;
        /** \breif data format: libsvm */
        std::string data_format;
        /** \brief output model */
        std::string model_out;
        /** \breif input model */
        std::string model_in;
        /** \breif loss function (MLE) */
        std::string loss;
        /** \brief could be sgd or frank wolfe */
        std::string updater_type;
        /** \breif combination of features */
        bool combination;
        /** \brief threshold of feature count for
            eliminating features in advance while
            working with sparsity
        */
		real_t epsilon;
        uint32_t feat_thresh;
        uint32_t max_num_epochs;
        uint32_t batch_size;
        /** \brief number of threads */
        uint32_t nthreads;

        DMLC_DECLARE_PARAMETER(SGDLearnerParam) {
            DMLC_DECLARE_FIELD(data_in);
            DMLC_DECLARE_FIELD(val_data).set_default("");
            DMLC_DECLARE_FIELD(test_data).set_default("");
            DMLC_DECLARE_FIELD(data_format).set_default("libsvm");
            DMLC_DECLARE_FIELD(model_out).set_default("");
            DMLC_DECLARE_FIELD(model_in).set_default("");
            DMLC_DECLARE_FIELD(loss).set_default("mle");
            DMLC_DECLARE_FIELD(updater_type).set_default("sgd");
            DMLC_DECLARE_FIELD(combination).set_default(true);
            DMLC_DECLARE_FIELD(feat_thresh).set_default(10);
            DMLC_DECLARE_FIELD(epsilon).set_default(1e-4);
            DMLC_DECLARE_FIELD(max_num_epochs).set_default(20);
            DMLC_DECLARE_FIELD(batch_size).set_default(1);
            DMLC_DECLARE_FIELD(nthreads).set_default(DEFAULT_NTHREADS);
        }
    };
} //namespace hazard

#endif
