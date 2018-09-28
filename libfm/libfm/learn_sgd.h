//
// Created by yanyu on 2017/7/5.
//

#ifndef PYLIBFM_LEARN_SGD_H
#define PYLIBFM_LEARN_SGD_H

#include "learn.h"
#include "../core/sgd.h"

class FMLearnSGD : public FMLearn {
public:
    //学习率
    double learn_rate;

    virtual void init() {
        FMLearn::init();
    }

    virtual void learn(LargeSparseMatrix<float> &x, DVector<float> &target) {
        FMLearn::learn(x, target);
    }


    void SGD(SparseRow<float> &x, const double multiplier, DVector<float> &sum) {
        fm_SGD(fm, learn_rate, x, multiplier, sum);
    }

    virtual void predict(LargeSparseMatrix<float> &x, DVector<float> &out) {
        assert(x.getNumRows() == out.dim);
        for (x.begin(); !x.end(); x.next()) {
            double p = predict_case(x);
            if (task == TASK_REGRESSION) {
                p = std::fminf(fm->max_target, p);
                p = std::fmaxf(fm->min_target, p);
            } else if (task == TASK_CLASSIFICATION) {
                p = 1.0 / (1.0 + exp(-p));
            } else {
                throw "task not supported";
            }
            out(x.getRowIndex()) = p;
        }
    }
};

#endif //PYLIBFM_LEARN_SGD_H
