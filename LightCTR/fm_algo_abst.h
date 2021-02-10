//
//  fm_algo_abst.h
//  LightCTR
//
//  Created by SongKuangshi on 2017/9/23.
//  Copyright © 2017年 SongKuangshi. All rights reserved.
//

#ifndef fm_algo_abst_h
#define fm_algo_abst_h

#include <iostream>
#include <stdio.h>
#include <vector>
#include <set>
#include <thread>
#include <fstream>
#include <string>
#include <cmath>
#include "assert.h"
#include "util/random.h"
#include "util/gradientUpdater.h"
#include "util/momentumUpdater.h"

#define FM

using namespace std;

struct FMFeature {
    //特征ID
    size_t first; // feature id
    //特征值
    float second; // value
    //特征域
    size_t field;
    FMFeature(size_t _first, float _second, size_t _field):
    first(_first), second(_second), field(_field) {}
};

class FM_Algo_Abst {
public:
    FM_Algo_Abst(string _dataPath, size_t _factor_cnt,
                 size_t _field_cnt = 0, size_t _feature_cnt = 0):
    feature_cnt(_feature_cnt), field_cnt(_field_cnt), factor_cnt(_factor_cnt) {
        //多线程，火力全开
        proc_cnt = thread::hardware_concurrency();
        printf("proc_cnt=%ld", proc_cnt);
        loadDataRow(_dataPath);
        init();
    }
    virtual ~FM_Algo_Abst() {
        delete [] W;
#ifdef FM
        delete [] V;
        delete [] sumVX;
#endif
    }

    //初始化模型参数
    void init() {
        //一次项参数
        W = new float[this->feature_cnt];
        memset(W, 0, sizeof(float) * this->feature_cnt);
#ifdef FM
        //for FM
        size_t memsize = this->feature_cnt * this->factor_cnt;
        if (this->field_cnt > 0) {
            //for FFM
            memsize = this->feature_cnt * this->field_cnt * this->factor_cnt;
        }
        V = new float[memsize];
        const float scale = 1.0 / sqrt(this->factor_cnt);
        for (size_t i = 0; i < memsize; i++) {
            V[i] = GaussRand() * scale;
        }
        sumVX = NULL;
#endif
    }
    
    //加载训练数据
    //同时统计特征ID的维度、特征域的数量等信息
    void loadDataRow(string dataPath) {
        dataSet.clear();
        
        ifstream fin_;
        string line;
        int nchar, y;
        size_t fid, fieldid;
        float val;
        fin_.open(dataPath, ios::in);
        if(!fin_.is_open()){
            cout << "open file error!" << endl;
            exit(1);
        }
        vector<FMFeature> tmp;
        while(!fin_.eof()){
            getline(fin_, line);
            tmp.clear();
            const char *pline = line.c_str();
            // label field_id:feature_id:feature_value ...
            if(sscanf(pline, "%d%n", &y, &nchar) >= 1){
                pline += nchar + 1;
                label.emplace_back(y);
                while(pline < line.c_str() + (int)line.length() &&
                      sscanf(pline, "%zu:%zu:%f%n", &fieldid, &fid, &val, &nchar) >= 2){
                    pline += nchar + 1;
                    tmp.emplace_back(*new FMFeature(fid, val, fieldid));
                    //找到特征ID的最大值，作为特征的维度
                    this->feature_cnt = max(this->feature_cnt, fid + 1);
                    if (this->field_cnt > 0) {
                        this->field_cnt = max(this->field_cnt, fieldid + 1);
                    }
                }
            }
            if (tmp.empty()) {
                continue;
            }
            this->dataSet.emplace_back(move(tmp));
        }
        this->dataRow_cnt = this->dataSet.size();
    }
    
    //保存模型参数
    void saveModel(size_t epoch) {
        char buffer[1024];
        snprintf(buffer, 1024, "%d", (int)epoch);
        string filename = buffer;
        ofstream md("./output/model_epoch_" + filename + ".txt");
        if(!md.is_open()){
            cout<<"save model open file error" << endl;
            exit(1);
        }
        //保存非0参数
        for (size_t fid = 0; fid < this->feature_cnt; fid++) {
            if (W[fid] != 0) {
                md << fid << ":" << W[fid] << " ";
            }
        }
        md << endl;
#ifdef FM
        // print all factor V
        for (size_t fid = 0; fid < this->feature_cnt; fid++) {
            md << fid << ":";
            for (size_t fac_itr = 0; fac_itr < this->factor_cnt; fac_itr++) {
                md << *getV(fid, fac_itr) << " ";
            }
            md << endl;
        }
#endif
        md.close();
    }
    
    virtual void Train() = 0;
    
    float L2Reg_ratio;
    
    float *W;
    size_t feature_cnt, proc_cnt, field_cnt, factor_cnt;
    size_t dataRow_cnt;
    
    float *V, *sumVX;
    inline float* getV(size_t fid, size_t facid) const {
        return &V[fid * this->factor_cnt + facid];
    }
    inline float* getV_field(size_t fid, size_t fieldid, size_t facid) const {
        return &V[fid * this->field_cnt * this->factor_cnt + fieldid * this->factor_cnt + facid];
    }
    inline float* getSumVX(size_t rid, size_t facid) const {
        return &sumVX[rid * this->factor_cnt + facid];
    }
    
    //数据集
    vector<vector<FMFeature> > dataSet;
    
protected:
    inline float LogisticGradW(float pred, float label, float x) {
        //TODO 这里loss用的是MSE？为啥loss用的是cross-entropy
        return (pred - label) * x;
    }
    inline float LogisticGradV(float gradW, float sum, float v, float x) {
        return gradW * (sum - v * x);
    }
    
    AdagradUpdater_Num updater;
    float __loss;
    float __accuracy;
    
    vector<int> label;
    vector<set<int> > cross_field;
};

#endif /* fm_algo_abst_h */
