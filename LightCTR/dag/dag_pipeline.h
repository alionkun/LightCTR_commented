//
//  dag_pipeline.h
//  LightCTR
//
//  Created by SongKuangshi on 2019/5/5.
//  Copyright © 2019 SongKuangshi. All rights reserved.
//

#ifndef dag_pipeline_h
#define dag_pipeline_h

#include "aggregate_node.h"
#include "source_node.h"
#include "terminus_node.h"

#include "operator/add_op.h"
#include "operator/multiply_op.h"
#include "operator/matmul_op.h"
#include "operator/activations_op.h"
#include "operator/loss_op.h"

// build up pipelines of computation
// or directed acyclic graphs (DAGs) of computation

class DAG_Pipeline {
public:
    
    //构造DAG图
    //用于eval或者predict
    static void addDirectedFlow(std::shared_ptr<Autograd_Node_Abst> source_ptr,
                         std::shared_ptr<Autograd_Node_Abst> terminus_ptr) {
        //链接source_ptr节点到terminus_ptr节点
        terminus_ptr->regist_in_node(source_ptr);
    }
    
    //构造包含梯度子图的DAG
    //用于train
    static void addAutogradFlow(std::shared_ptr<Autograd_Node_Abst> source_ptr,
                                std::shared_ptr<Autograd_Node_Abst> terminus_ptr) {
        //链接source_ptr节点到terminus_ptr节点
        terminus_ptr->regist_in_node(source_ptr);
        //梯度子图的话，反过来
        source_ptr->regist_out_node(terminus_ptr);
    }
    
};


#endif /* dag_pipeline_h */
