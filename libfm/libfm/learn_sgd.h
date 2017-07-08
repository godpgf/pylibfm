//
// Created by yanyu on 2017/7/5.
//

#ifndef PYLIBFM_LEARN_SGD_H
#define PYLIBFM_LEARN_SGD_H

#include "learn.h"
#include "../core/sgd.h"

class FMLearnSGD: public FMLearn {
    public:
        //训练次数
        int num_iter;
        //学习率
        double learn_rate;

        virtual void init(){
            FMLearn::init();
        }


        void SGD(SparseRow<float> &x, const double multiplier, DVector<double> &sum){
            fm_SGD(fm, learn_rate, x, multiplier, sum);
        }

        virtual void learn(LargeSparseMatrix<float> &x, DVector<double>& target) {
            FMLearn::learn(x, target);
        }

        virtual void predict(LargeSparseMatrix<float> &x, DVector<double>& out){
            assert(x.getNumRows() == out.dim);
            for (x.begin(); !x.end(); x.next()){
                double p = predict_case(x);
                if (task == TASK_REGRESSION ) {
                    p = std::min(fm->max_target, p);
                    p = std::max(fm->min_target, p);
                } else if (task == TASK_CLASSIFICATION) {
                    p = 1.0/(1.0 + exp(-p));
                } else {
                    throw "task not supported";
                }
                out(x.getRowIndex()) = p;
            }
        }
};

#endif //PYLIBFM_LEARN_SGD_H
