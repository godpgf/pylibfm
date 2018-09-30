//
// Created by godpgf on 18-9-30.
//

#ifndef LIBFM_LEARN_MCMC_SIMULTANEOUS_H
#define LIBFM_LEARN_MCMC_SIMULTANEOUS_H

#include "learn_mcmc.h"

class FMLearnMCMCSimultaneous : FMLearnMCMC{
protected:
    virtual void _learn(LargeSparseMatrix<float> &x, DVector<float>& target){

    }
};

#endif //LIBFM_LEARN_MCMC_SIMULTANEOUS_H
