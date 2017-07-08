//
// Created by yanyu on 2017/7/6.
//
#include "util/fmatrix.h"
#include "util/matrix.h"
#include "libfm/learn_sgd_element.h"

extern "C"
{
    //稀疏矩阵相关----------------------------------------------------------------

    //创建一个dim行的稀疏矩阵
    LargeSparseMatrixMemory<float>* createSparseMatrix(int dim);
    //在稀疏矩阵中填写一行
    void fillSparseMatrixLine(LargeSparseMatrixMemory<float>* sparseMatrix, int line, int* ids, float* values, int len);
    //清除
    void releaseSparseMatrix(LargeSparseMatrixMemory<float>* sparseMatrix);

    //动态向量相关------------------------------------------------------------------

    //创建一个动态向量
    DVector<double>* createDVector(float* values, int len);
    void transformVector2Array(DVector<double>* vector, float* out);
    //清除
    void releaseDVector(DVector<double>* vector);

    //学习器相关---------------------------------------------------------------------

    FMModel* createFMModel(int num_attribute, int num_factor, bool is_use_w0, bool is_use_w, double init_stdev, double reg0, double regw, double regv);
    void releaseFMModel(FMModel* fm);

    //创建学习器
    FMLearn* createFM(int num_iter, char* task, char* algorithm, FMModel* fm, double learning_rate);
    void learn(FMLearn* fml, LargeSparseMatrix<float> *x, DVector<double> *target);
    void predict(FMLearn* fml, LargeSparseMatrix<float> *x, DVector<double> *out);
    void releaseFM(FMLearn* fml);


    //测试
    int test_connect(int a, int b, char* name, bool c, double d){
        std::cout << "hello " << name << "!" << std::endl;
        if(c){std::cout<<"true";}else{std::cout<<"false";}
        std::cout<<std::endl;
        std::cout << d << std::endl;
        return a + b;
    }
}

LargeSparseMatrixMemory<float>* createSparseMatrix(int dim){
    LargeSparseMatrixMemory<float>* sparseMatrix = new LargeSparseMatrixMemory<float>();
    sparseMatrix->data.setSize(dim);
    return sparseMatrix;
}

void fillSparseMatrixLine(LargeSparseMatrixMemory<float>* sparseMatrix, int line, int* ids, float* values, int len)
{
    sparseMatrix->data(line).size = len;
    sparseMatrix->data(line).data = new SparseEntry<float>[len];
    for(int i = 0; i < len; i++){
        sparseMatrix->data(line).data[i].id = ids[i];
        sparseMatrix->data(line).data[i].value = values[i];
        //std::cout<<ids[i]<<" "<<values[i]<<std::endl;
    }
}

void releaseSparseMatrix(LargeSparseMatrixMemory<float>* sparseMatrix)
{
    for(int i = 0; i < sparseMatrix->data.dim; i++){
        delete[] sparseMatrix->data(i).data;
    }
    delete sparseMatrix;
}

DVector<double>* createDVector(float* values, int len){
    DVector<double>* vector = new DVector<double>();
    vector->setSize(len);
    for(int i = 0; i < len; i++){
        vector->value[i] = values[i];
    }
    return vector;
}

void transformVector2Array(DVector<double>* vector, float* out) {
    for(int i = 0; i < vector->dim; i++){
        out[i] = (float)(*vector)(i);
    }
}

void releaseDVector(DVector<double>* vector) {
    delete vector;
}

FMModel* createFMModel(int num_attribute, int num_factor,bool is_use_w0, bool is_use_w, double init_stdev, double reg0, double regw, double regv){
    FMModel* fm = new FMModel();
    fm->num_attribute = num_attribute;
    fm->num_factor = num_factor;
    fm->is_use_w0 = is_use_w0;
    fm->is_use_w = is_use_w;
    fm->init_stdev = init_stdev;
    fm->reg0 = reg0;
    fm->regw = regw;
    fm->regv = regv;
    fm->init();
    return fm;
}

void releaseFMModel(FMModel* fm){
    delete fm;
}

FMLearn* createFM(int num_iter, char* task, char* algorithm, FMModel* fm, double learning_rate){
    FMLearn* fml = NULL;

    //std::cout<<task<<std::endl;
    //std::cout<<algorithm<<std::endl;
    //std::cout<<fm<<std::endl;
    //std::cout<<fm->num_factor<<std::endl;
    //return NULL;

    if(strcmp("sgd", algorithm) == 0){
        fml = new FMLearnSGDElement();
        ((FMLearnSGDElement*)fml)->num_iter = num_iter;
        ((FMLearnSGDElement*)fml)->learn_rate = learning_rate;
    } else if (strcmp("sgda", algorithm) == 0){
        //todo
    } else if (strcmp("mcmc", algorithm) == 0){
        //todo
    } else{
        throw "algorithm has not support !";
    }

    std::cout<<strcmp("regression", task)<<std::endl;

    if(strcmp("regression", task) == 0){
        fml->task = TASK_REGRESSION;
    } else if(strcmp("classification", task) == 0){
        fml->task = TASK_CLASSIFICATION;
    } else {
        throw "task error";
    }


    fml->fm = fm;
    fml->init();
    return fml;
}

void learn(FMLearn* fml, LargeSparseMatrix<float> *x, DVector<double> *target) {
    fml->learn(*x, *target);
}

void predict(FMLearn* fml, LargeSparseMatrix<float> *x, DVector<double> *out){
    fml->predict(*x, *out);
}

void releaseFM(FMLearn* fml) {
    delete fml;
}

//测试算法
void test(){
    FMLearn* fml = new FMLearnSGDElement();
    fml->task = TASK_REGRESSION;
    ((FMLearnSGDElement*)fml)->num_iter = 100;
    ((FMLearnSGDElement*)fml)->learn_rate = 0.01;

    FMModel fm;
    {
        fm.num_attribute = 9;
        fm.num_factor = 2;
        //初始化内存数据
        fm.init();
    }
    fml->fm = &fm;

    LargeSparseMatrixMemory<float> x;

    x.data.setSize(4);
    int value[4] = {19, 33, 55, 20};
    int ids2[4] = {4,3,2,1};
    int ids3[4] = {5,6,7,8};

    DVector<double> y;
    y.setSize(4);

    DVector<double> t;
    t.setSize(4);
    for(int i = 0; i < 4; i++){
        x.data(i).size = 3;
        x.data(i).data = new SparseEntry<float>[3];
        x.data(i).data[0].id = 0;
        x.data(i).data[0].value = value[i];
        x.data(i).data[1].id = ids2[i];
        x.data(i).data[1].value = 1;
        x.data(i).data[2].id = ids3[i];
        x.data(i).data[2].value = 1;

        y.value[i] = i % 2;
        t.value[i] = 0;
    }


    fml->init();
    fml->learn(x, y);
    fml->predict(x, t);

    for(int i = 0; i < t.dim; i++)
        std::cout << t.value[i] << std::endl;
}