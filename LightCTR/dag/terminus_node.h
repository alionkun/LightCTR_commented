//
//  terminus_node.h
//  LightCTR
//
//  Created by SongKuangshi on 2019/5/19.
//  Copyright © 2019 SongKuangshi. All rights reserved.
//

#ifndef terminus_node_h
#define terminus_node_h

#include <vector>
#include "node_abst.h"
#include "../common/avx.h"

//终点节点，没有输出
class TerminusNode : public Autograd_Node_Abst {
public:
    TerminusNode() = delete;
    explicit TerminusNode(size_t _in_cnt) : Autograd_Node_Abst(_in_cnt, 0) {
        assert(_in_cnt > 0);
    }
    
    //调用结束节点，运行DAG的向前模式
    DAG_Output runFlow(bool keep_intermediate = false) {
        init_forward_Flow(keep_intermediate);
        return forward_run().get();
    }
    
protected:
    void forward_compute(const std::vector<DAG_Output>& in_outputs) {
        // compute delta via loss function
    }
    
    void backward_compute(const std::vector<DAG_Output>&) {
        // back propagate delta 
    }
};

#endif /* terminus_node_h */
