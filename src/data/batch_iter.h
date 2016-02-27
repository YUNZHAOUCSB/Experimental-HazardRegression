/**
 * Copyright (c) 2016 by Contributors
 * \author Ziqi Liu
 */

#ifndef HAZARD_DATA_BATCH_ITER_H_
#define HAZARD_DATA_BATCH_ITER_H_
#include "dmlc/data.h"
#include "data/parser.h"
#include "hazard/base.h"

namespace hazard {

    /**
     * \brief an iterator reads a batch with a given number of examples
     * each time.
     */
    class BatchIter {
    public:
        /**
         * \brief create a batch iterator
         *
         * @param uri filename
         * @param format the data format, support libsvm, rec, ...
         * @param part_index the i-th part to read
         * @param num_parts partition the file into serveral parts
         * @param batch_size the batch size.
         */
        BatchIter(const std::string& uri,
                  const std::string& format,
                  unsigned part_index,
                  unsigned num_parts,
                  unsigned batch_size);

        ~BatchIter() {
            delete parser_;
        }

        /**
         * \brief read the next batch
         */
        bool Next();

        /**
         * \brief get the current batch
         *
         */
        const dmlc::RowBlock<feaid_t>& Value() const {
            return out_blk_;
        }

        /**
         * \brief reset to the file beginning
         */
        void Reset() {
            if (parser_) parser_->BeforeFirst();
        }

    private:
        /**
         * \brief batch_.push(in_blk_(pos:pos+len))
         */
        void Push(size_t pos, size_t len);

        unsigned batch_size_;
        dmlc::data::ParserImpl<feaid_t> *parser_;

        size_t start_, end_;
        dmlc::RowBlock<feaid_t> in_blk_, out_blk_;
        dmlc::data::RowBlockContainer<feaid_t> batch_;
    };

}  // namespace hazard
#endif  // HAZARD_DATA_BATCH_ITER_H_