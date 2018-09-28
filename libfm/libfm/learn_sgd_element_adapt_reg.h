//
// Created by godpgf on 18-9-28.
// 自动调整正则项的sgd
//

#ifndef LIBFM_LEARN_SGD_ELEMENT_ADAPT_REG_H
#define LIBFM_LEARN_SGD_ELEMENT_ADAPT_REG_H

#include "learn_sgd.h"

class FMLearnSGDAElement: public FMLearnSGD{
public:
    virtual void init(){
        FMLearnSGD::init();
    }

    virtual void learn(LargeSparseMatrix<float> &x, DVector<float>& target){
        FMLearnSGD::learn(x, target);
        fm->w.init(0);
        fm->reg0 = 0;
        fm->regw = 0;
        fm->regv = 0;
    }
};

#endif //LIBFM_LEARN_SGD_ELEMENT_ADAPT_REG_H
