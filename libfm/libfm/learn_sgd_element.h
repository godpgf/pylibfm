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

        virtual void learn(LargeSparseMatrix<float> &x, DVector<double>& target){
            FMLearnSGD::learn(x, target);

            std::cout << "SGD: DON'T FORGET TO SHUFFLE THE ROWS IN TRAINING DATA TO GET THE BEST RESULTS." << std::endl;
            for (int i = 0; i < num_iter; i++){
                for (x.begin(); !x.end(); x.next()){
                    double p = fm->predict(x.getRow(), sum, sum_sqr);
                    double mult = 0;
                    if (task == TASK_REGRESSION) {
                        p = std::min(fm->max_target, p);
                        p = std::max(fm->min_target, p);
                        mult = -(target(x.getRowIndex())-p);
                    } else if (task == TASK_CLASSIFICATION) {
                        mult = -target(x.getRowIndex())*(1.0-1.0/(1.0+exp(-target(x.getRowIndex())*p)));
                    }
                    SGD(x.getRow(), mult, sum);
                }
                double rmse = evaluate(x, target);
                std::cout << "#Iter= " << i << "\tTrain=" << rmse << std::endl;
            }
        }
};

#endif //PYLIBFM_LEARN_SGD_ELEMENT_H
