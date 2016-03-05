/**
 * Copyright (c) 2016 by Contributors
 * \author Ziqi Liu
 */

#include "./batch_iter.h"
#include "data/libsvm_parser.h"

namespace hazard {

    BatchIter::BatchIter(
        const std::string& uri, const std::string& format,
        unsigned part_index, unsigned num_parts,
        unsigned batch_size) {
        batch_size_   = batch_size;
        start_        = 0;
        end_          = 0;
        // create parser
        char const* c_uri = uri.c_str();
        if (format == "libsvm") {
            parser_ = new dmlc::data::LibSVMParser<real_t>(
                dmlc::InputSplit::Create(c_uri, part_index, num_parts, "text"), 1);
        }
        else {
            LOG(FATAL) << "unknown format " << format;
        }
        parser_ = new dmlc::data::ThreadedParser<real_t>(parser_);
    }

    bool BatchIter::Next() {
        batch_.Clear();
        while (batch_.offset.size() < batch_size_ + 1) {
            if (start_ == end_) {
                if (!parser_->Next()) break;
                in_blk_ = parser_->Value();
                start_ = 0;
                end_ = in_blk_.size;
            }

            size_t len = std::min(end_ - start_, batch_size_ + 1 - batch_.offset.size());
            Push(start_, len);
            start_ += len;
        }

        bool binary =  true;
        for (auto f : batch_.value) if (f != 1) { binary = false; break; }
        if (binary) batch_.value.clear();

        out_blk_ = batch_.GetBlock();

        return out_blk_.size > 0;
    }

    void BatchIter::Push(size_t pos, size_t len) {
        if (!len) return;
        CHECK_LE(pos + len, in_blk_.size);
        dmlc::RowBlock<real_t> slice;
        slice.weight = NULL;
        slice.size = len;
        slice.offset  = in_blk_.offset + pos;
        slice.label   = in_blk_.label  + pos;
        slice.index   = in_blk_.index  + in_blk_.offset[pos];
        if (in_blk_.value) {
            slice.value = in_blk_.value  + in_blk_.offset[pos];
        } else {
            slice.value = NULL;
        }
        batch_.Push(slice);
    }

}  // namespace hazard