//
//  add_op.h
//  LightCTR
//
//  Created by SongKuangshi on 2019/5/20.
//  Copyright © 2019 SongKuangshi. All rights reserved.
//

#ifndef add_op_h
#define add_op_h

#include "../aggregate_node.h"
#include "string.h"
#include "../../common/avx.h"

class AddOp : public AggregateNode {
public:
    AddOp() = delete;
    //累加_in_cnt个输入
    AddOp(size_t _in_cnt, size_t _out_cnt = 1) : AggregateNode(_in_cnt, _out_cnt) {
        assert(_in_cnt > 0 && _out_cnt > 0);
    }
    
protected:
    void forward_compute(const std::vector<DAG_Output>& in_outputs) {
        //随意检查前2个输入的shape是否一致
        const size_t len = in_outputs[0].data->size();
        assert(len == in_outputs[1].data->size());
        if (node_output.data == nullptr) {
            node_output.data = std::make_shared<std::vector<float> >(len);
        }
        
        std::memset(node_output.data->data(), 0, len * sizeof(float));
        for(auto& in_output : in_outputs) {
            avx_vecAdd(node_output.data->data(), in_output.data->data(),
                       node_output.data->data(), len);
        }
    }
    
    void backward_compute(const std::vector<DAG_Output>& out_deltas) {
        const size_t len = out_deltas[0].data->size();
        if (node_delta.data == nullptr) {
            node_delta.data = std::make_shared<std::vector<float> >(len);
        }
        
        //加法的梯度，直接等于输入梯度之和的一倍
        std::memset(node_delta.data->data(), 0, len * sizeof(float));
        for(auto& out_delta : out_deltas) {
            avx_vecAdd(node_delta.data->data(), out_delta.data->data(),
                       node_delta.data->data(), len);
        }
    }
};

#endif /* add_op_h */
