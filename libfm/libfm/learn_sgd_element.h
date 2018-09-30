//
// Created by yanyu on 2017/7/5.
//

#ifndef PYLIBFM_LEARN_SGD_ELEMENT_H
#define PYLIBFM_LEARN_SGD_ELEMENT_H

#include "learn_sgd.h"

class FMLearnSGDElement: public FMLearnSGD{
    public:
        virtual void init(){
            FMLearnSGD::init();
        }

        virtual void learn(LargeSparseMatrix<float> &x, DVector<float>& target){
            FMLearnSGD::learn(x, target);
            for (x.begin(); !x.end(); x.next()){
                float p = fm->predict(x.getRow(), sum, sum_sqr);
//                std::cout<<p<<std::endl;
                float mult = 0;
                if (task == TASK_REGRESSION) {
                    p = std::fminf(fm->max_target, p);
                    p = std::fmaxf(fm->min_target, p);
                    mult = -(target(x.getRowIndex())-p);
                } else if (task == TASK_CLASSIFICATION) {
                    mult = -target(x.getRowIndex())*(1.0-1.0/(1.0+exp(-target(x.getRowIndex())*p)));
                }
                SGD(x.getRow(), mult, sum);
            }
        }
};

#endif //PYLIBFM_LEARN_SGD_ELEMENT_H
