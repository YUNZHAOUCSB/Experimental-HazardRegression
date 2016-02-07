/**
 * Copyright (c) 2015 by Contributors
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
        /** \brief learning rate */
        real_t lr;
        /** \brief learning rate decay */
        real_t decay;

        real_t initval;

        DMLC_DECLARE_PARAMETER(SGDUpdaterParam) {
            DMLC_DECLARE_FIELD(l1).set_range(0.0f,1e2f).set_default(1);
            DMLC_DECLARE_FIELD(l2).set_range(0.0f,1e2f).set_default(1);
            DMLC_DECLARE_FIELD(lr).set_range(0.0f,1e1f).set_default(1);
            DMLC_DECLARE_FIELD(decay).set_range(0.0f,1.0f).set_default(1);
            DMLC_DECLARE_FIELD(initval).set_default(1e-3);
        }
    }; //class SGDUpdaterParam

    struct SGDLearnerParam : public dmlc::Parameter<SGDLearnerParam> {
        std::string data_in;
        std::string val_data;
        std::string data_format;
        std::string model_out;
        std::string model_in;
        std::string loss;
        std::string updater_type;
        uint32_t max_num_epochs;
        uint32_t batch_size;
        uint32_t nthreads;
        time_t lcensor;
        time_t starttime;

        DMLC_DECLARE_PARAMETER(SGDLearnerParam) {
            DMLC_DECLARE_FIELD(data_in);
            DMLC_DECLARE_FIELD(val_data).set_default("");
            DMLC_DECLARE_FIELD(data_format).set_default("libsvm");
            DMLC_DECLARE_FIELD(model_out).set_default("");
            DMLC_DECLARE_FIELD(model_in).set_default("");
            DMLC_DECLARE_FIELD(loss).set_default("mle");
            DMLC_DECLARE_FIELD(max_num_epochs).set_default(20);
            DMLC_DECLARE_FIELD(batch_size).set_default(1);
            DMLC_DECLARE_FIELD(updater_type).set_default("sgd_dynamic");
            DMLC_DECLARE_FIELD(nthreads).set_default(DEFAULT_NTHREADS);
            DMLC_DECLARE_FIELD(lcensor).set_default(10);
            DMLC_DECLARE_FIELD(starttime).set_default(0);
        }
    };
}

#endif