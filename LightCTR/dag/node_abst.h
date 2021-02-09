//
//  node_abst.h
//  LightCTR
//
//  Created by SongKuangshi on 2019/5/5.
//  Copyright © 2019 SongKuangshi. All rights reserved.
//

#ifndef node_abst_h
#define node_abst_h

#include <future>
#include "../common/thread_pool.h"

//抽象节点
class Node_Abst {
public:
    //表示一个节点的输出
    class DAG_Output {
    public:
        DAG_Output() {
            node_id = 0;
            data = nullptr;
        }
        DAG_Output(size_t _node_id) : node_id(_node_id) {
        }
        
        //节点的输出均为float向量
        std::shared_ptr<std::vector<float> > data;
        
        size_t node_id;
        //标识是否已经完成计算
        bool deal_flag = 0;
    };
    
    Node_Abst() = delete;
    Node_Abst(size_t _in_cnt, size_t _out_cnt) : dag_threadpool(ThreadPool::Instance()),
    in_cnt(_in_cnt), out_cnt(_out_cnt) {
        static size_t global_node_id = 1; // begin from 1
        
        node_id = global_node_id++;
        node_output.node_id = node_id; // set output node_id
        
        in_nodes.reserve(in_cnt);
    };
    
    //指定输入节点
    inline void regist_in_node(std::shared_ptr<Node_Abst> ptr) {
        assert(ptr != nullptr && in_nodes.size() < in_cnt);
        in_nodes.emplace_back(ptr);
    }
    
    inline const DAG_Output& getOutput() const {
        return node_output;
    }
    
    inline size_t getNodeId() const {
        return node_id;
    }

protected:
    //执行本节点，并返回一个输出的future
    std::future<DAG_Output> forward_run(size_t out_id = 0) {
        //printf("begin to run %ld\n", node_id);

        //所有输入节点都配置好了
        assert(in_nodes.size() == in_cnt);
        
        //本节点已经完成了计算，直接返回结果
        if (node_output.deal_flag) { // return cached intermediate result
            return dag_threadpool.addTask([&]() -> DAG_Output {
                return node_output;
            });
        }
        
        //无锁化进入本节点的计算逻辑
        if(unlikely(CAS32(&forward_flag, 0, 1))) {
            forward_reset();
            auto in_futures = std::make_shared<std::vector<std::future<DAG_Output> > >(in_cnt);
            if (in_cnt > 0) { // source_node in_cnt == 0
                for(size_t i = 0; i < in_cnt; i++) {
                    //首先调用上游节点
                    in_futures->at(i) = in_nodes[i]->forward_run(getNodeId());
                }
            }
            return dag_threadpool.addTask([&, in_futures]() -> DAG_Output {
                std::vector<DAG_Output> in_outputs(in_cnt);
                if (in_cnt > 0) {
                    for(size_t i = 0; i < in_cnt; i++) {
                        //然后读取所有输入，这里等待上游所有节点完成计算
                        in_outputs[i] = in_futures->at(i).get();
                        assert(in_outputs[i].node_id > 0);
                    }
                }
                //所有输入都ready之后，开始本节点的计算逻辑
                return forward_compute_wrapper(in_outputs);
            });
        };
        assert(out_id_inc < out_cnt - 1);
        return out_complete_promises[out_id_inc++].get_future();
    }
    
    //节点的计算逻辑，由子类实现
    virtual void forward_compute(const std::vector<DAG_Output>&) = 0;
    
    //为向前计算做初始化，主要是设置子图中所有节点的状态，清理缓存
    void init_forward_Flow(bool keep_intermediate) {
        assert(in_nodes.size() == in_cnt);
        forward_flag = 0; // force reset first targeting
        
        if (!keep_intermediate) {
            node_output.deal_flag = false;
        }
        for(auto& in_node : in_nodes) {
            in_node->init_forward_Flow(keep_intermediate);
        }
    }
    
    //触发输出节点的promise
    inline void complete_out_promises() {
        for (auto& promise : out_complete_promises) {
            promise.set_value(node_output);
        }
    }
    
    //in_cnt表示本节点依赖的上游节点的数量
    //out_cnt表示依赖本节点的下游节点数量，注意这里不是输出参数的个数，
    //节点默认只输出一个float向量，封装在node_output中，out_cnt表示out_complete_promises
    size_t in_cnt, out_cnt;
    //表示是否正在执行本节点，由于DAG是多线程运行，这个flag用于无锁化
    uint32_t forward_flag = 0;
    
    DAG_Output node_output;
    std::vector<DAG_Output> compute_records;
    
    ThreadPool& dag_threadpool;
    
private:
    DAG_Output forward_compute_wrapper(const std::vector<DAG_Output>& out_deltas) {
        forward_compute(out_deltas);
        node_output.deal_flag = true;

        //printf("run done %ld\n", node_id);
        
        complete_out_promises();
        return node_output;
    }
    
    void forward_reset() {
        out_id_inc = 0;
        compute_records.clear(); // clear records when forward reset
        if (likely(out_cnt > 0)) { // terminus_node out_cnt == 0
            out_complete_promises.clear();
            out_complete_promises.reserve(out_cnt - 1);
            for (size_t i = 0; i < out_cnt - 1; i++) {
                out_complete_promises.emplace_back(std::promise<DAG_Output>());
            }
        }
    }
    
    //输出节点的primse
    std::vector<std::promise<DAG_Output> > out_complete_promises;
    
    //依赖的上游节点
    std::vector<std::shared_ptr<Node_Abst> > in_nodes;
    
    //下游节点请求本节点的数量
    size_t out_id_inc = 0;
    size_t node_id;
};


//Node_Abst包含了DAG向前计算，Autograd_Node_Abst在此基础上增加了反向梯度计算
class Autograd_Node_Abst : public Node_Abst {
public:
    Autograd_Node_Abst() = delete;
    Autograd_Node_Abst(size_t _in_cnt, size_t _out_cnt) : Node_Abst(_in_cnt, _out_cnt) {
        node_delta.node_id = getNodeId(); // set delta node_id
        
        out_nodes.reserve(out_cnt);
    };
    
    inline void regist_out_node(std::shared_ptr<Autograd_Node_Abst> ptr) {
        assert(ptr != nullptr && out_nodes.size() < out_cnt);
        out_nodes.emplace_back(ptr);
    }
    
    inline const DAG_Output& getDelta() const {
        return node_delta;
    }
    
protected:
    //反向计算
    std::future<DAG_Output> backward_run(size_t in_id = 0) {
        assert(out_nodes.size() == out_cnt);
        
        if (node_delta.deal_flag) { // return cached intermediate result
            return dag_threadpool.addTask([&]() -> DAG_Output {
                return node_delta;
            });
        }
        
        if(unlikely(CAS32(&backward_flag, 0, 1))) {
            backward_reset();
            first_target_id = in_id;
            auto out_futures = std::make_shared<std::vector<std::future<DAG_Output> > >(out_cnt);
            if (out_cnt > 0) {
                for(size_t i = 0; i < out_cnt; i++) {
                    //在反向图中，节点依赖于对应正向图的下游节点
                    out_futures->at(i) = out_nodes[i].lock()->backward_run(getNodeId());
                }
            }
            return dag_threadpool.addTask([&, out_futures]() -> DAG_Output {
                std::vector<DAG_Output> out_deltas(out_cnt);
                if (out_cnt > 0) { // terminus_node out_cnt == 0
                    for(size_t i = 0; i < out_cnt; i++) {
                        out_deltas[i] = out_futures->at(i).get();
                        assert(out_deltas[i].node_id > 0);
                    }
                }
                return backward_compute_wrapper(out_deltas);
            });
        };
        assert(in_id_inc < in_cnt - 1);
        in_promises_ids.emplace_back(in_id);
        return in_complete_promises[in_id_inc++].get_future();
    }
    
    virtual void backward_compute(const std::vector<DAG_Output>&) = 0;
    
    //为反向计算做初始化
    void init_backward_Flow(bool keep_intermediate) {
        assert(out_nodes.size() == out_cnt);
        backward_flag = 0; // force reset first targeting
        
        if (!keep_intermediate) {
            node_delta.deal_flag = false;
        }
        for(auto& out_node : out_nodes) {
            out_node.lock()->init_backward_Flow(keep_intermediate);
        }
    }
    
    inline void complete_in_promises() {
        for (auto& promise : in_complete_promises) {
            promise.set_value(node_delta);
        }
    }
    
    inline const std::vector<size_t>& get_in_promises_ids() const {
        return in_promises_ids;
    }
    
    inline std::vector<std::promise<DAG_Output> >& get_in_complete_promises() {
        return in_complete_promises;
    }
    
    inline size_t get_first_target_id() const {
        assert(first_target_id > 0);
        return first_target_id;
    }
    
    DAG_Output node_delta;
    
private:
    DAG_Output backward_compute_wrapper(const std::vector<DAG_Output>& out_deltas) {
        backward_compute(out_deltas);
        node_delta.deal_flag = true;
        
        complete_in_promises();
        return node_delta;
    }
    
    void backward_reset() {
        in_id_inc = 0;
        if (likely(in_cnt > 0)) { // source_node in_cnt == 0
            in_promises_ids.clear();
            in_promises_ids.reserve(in_cnt - 1);
            
            in_complete_promises.clear();
            in_complete_promises.reserve(in_cnt - 1);
            for (size_t i = 0; i < in_cnt - 1; i++) {
                in_complete_promises.emplace_back(std::promise<DAG_Output>());
            }
        }
    }
    
    std::vector<std::promise<DAG_Output> > in_complete_promises;
    std::vector<size_t> in_promises_ids;
    size_t first_target_id; // record the first targeting node_id
    
    std::vector<std::weak_ptr<Autograd_Node_Abst> > out_nodes;
    
    uint32_t backward_flag = 0;
    size_t in_id_inc = 0;
};

#endif /* node_abst_h */
