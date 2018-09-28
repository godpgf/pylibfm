//
// Created by yanyu on 2017/7/6.
//
#include "util/fmatrix.h"
#include "util/matrix.h"
#include <float.h>
#include "libfm/learn_sgd_element.h"

#ifndef WIN32 // or something like that...
#define DLLEXPORT
#else
#define DLLEXPORT __declspec(dllexport)
#endif

extern "C"
{
//稀疏矩阵相关----------------------------------------------------------------

//创建一个dim行的稀疏矩阵
LargeSparseMatrixMemory<float> *DLLEXPORT createSparseMatrix(int num_cols) {
    LargeSparseMatrixMemory<float> *sparseMatrix = new LargeSparseMatrixMemory<float>(num_cols);
    return sparseMatrix;
}

//在稀疏矩阵填写数据
void DLLEXPORT
fillSparseMatrix(LargeSparseMatrixMemory<float> *sparseMatrix, float *data, int *indices, int *indptr, int num_rows,
                 int num_values) {
    sparseMatrix->fill(data, indices, indptr, num_rows, num_values);
}

//清除
void DLLEXPORT releaseSparseMatrix(LargeSparseMatrixMemory<float> *sparseMatrix) {
    delete sparseMatrix;
}

//动态向量相关------------------------------------------------------------------

//创建一个动态向量
DVector<float> *DLLEXPORT createDVector() {
    DVector<float> *vector = new DVector<float>();
//        vector->setSize(len);
//        memcpy(vector->value, values, len * sizeof(float));
    return vector;
}

void DLLEXPORT fillDVector(DVector<float> *vector, float *values, int len) {
    vector->setSize(len);
    memcpy(vector->value, values, len * sizeof(float));
}

void DLLEXPORT transformVector2Array(DVector<float> *vector, float *out) {
    memcpy(out, vector->value, vector->dim * sizeof(float));
}

//清除
void DLLEXPORT releaseDVector(DVector<double> *vector) {
    delete vector;
}

//学习器相关---------------------------------------------------------------------

FMModel *DLLEXPORT createFMModel(int num_attribute, int num_factor, bool is_use_w0, bool is_use_w, double init_stdev,
                                 double reg0, double regw, double regv) {
    FMModel *fm = new FMModel();
    fm->num_attribute = num_attribute;
    fm->num_factor = num_factor;
    fm->is_use_w0 = is_use_w0;
    fm->is_use_w = is_use_w;
    fm->init_stdev = init_stdev;
    fm->reg0 = reg0;
    fm->regw = regw;
    fm->regv = regv;
    fm->max_target = -FLT_MAX;
    fm->min_target = FLT_MAX;
    fm->init();
    return fm;
}

void DLLEXPORT releaseFMModel(FMModel *fm) {
    delete fm;
}

//创建学习器
FMLearn *DLLEXPORT createFM(char *task, char *algorithm, FMModel *fm, double learning_rate) {
    FMLearn *fml = NULL;

//        std::cout<<task<<std::endl;
//        std::cout<<algorithm<<std::endl;
//        std::cout<<fm<<std::endl;
//        std::cout<<fm->num_factor<<std::endl;
//        return NULL;

    if (strcmp("sgd", algorithm) == 0) {
        fml = new FMLearnSGDElement();
        ((FMLearnSGDElement *) fml)->learn_rate = learning_rate;
    } else if (strcmp("sgda", algorithm) == 0) {
        //todo
    } else if (strcmp("mcmc", algorithm) == 0) {
        //todo
    } else {
        throw "algorithm has not support !";
    }

//        std::cout<<strcmp("regression", task)<<std::endl;

    if (strcmp("regression", task) == 0) {
        fml->task = TASK_REGRESSION;
    } else if (strcmp("classification", task) == 0) {
        fml->task = TASK_CLASSIFICATION;
    } else {
        throw "task error";
    }


    fml->fm = fm;
    fml->init();
    return fml;
}

void DLLEXPORT learn(FMLearn *fml, LargeSparseMatrix<float> *x, DVector<float> *target) {
    fml->learn(*x, *target);
}

void DLLEXPORT predict(FMLearn *fml, LargeSparseMatrix<float> *x, DVector<float> *out) {
    fml->predict(*x, *out);
}

float DLLEXPORT evaluate(FMLearn *fml, LargeSparseMatrix<float> *x, DVector<float> *target) {
    return fml->evaluate(*x, *target);
}

void DLLEXPORT releaseFM(FMLearn *fml) {
    delete fml;
}

}



