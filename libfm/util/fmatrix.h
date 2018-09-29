//
// Created by yanyu on 2017/7/4.
// 特征矩阵,即稀疏矩阵 Large-Scale Sparse Matrix
//

#ifndef PYLIBFM_FMATRIX_H
#define PYLIBFM_FMATRIX_H

#include <limits>
#include <vector>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include "../util/random.h"
#include "util.h"
#include "matrix.h"

//版本号
const int FMATRIX_EXPECTED_FILE_ID = 2;

//矩阵的行
template <typename T> struct SparseRow {
public:
    T* values;
    int* ids;
    int* col2group;
    int size;
};

//文件头
struct FileHeader {
    int id;
    int float_size;
    uint64 num_values;
    int num_rows;
    int num_cols;
};

template <typename T> class LargeSparseMatrix {
public:
    virtual ~LargeSparseMatrix() {};
    virtual void begin() = 0; // go to the beginning
    virtual bool end() = 0;   // are we at the end?
    virtual void next() = 0; // go to the next line
    virtual SparseRow<T>& getRow() = 0; // pointer to the current row
    virtual int getRowIndex() = 0; // index of current row (starting with 0)
    virtual int getNumRows() = 0; // get the number of Rows
    virtual int getNumCols() = 0; // get the number of Cols
    virtual uint64 getNumValues() = 0; // get the number of Values


    void saveToBinaryFile(std::string filename) {
        std::cout << "printing to " << filename << std::endl; std::cout.flush();
        std::ofstream out(filename.c_str(), std::ios_base::out | std::ios_base::binary);
        if (out.is_open()) {
            FileHeader fh;
            fh.id = FMATRIX_EXPECTED_FILE_ID;
            fh.num_values = getNumValues();
            fh.num_rows = getNumRows();
            fh.num_cols = getNumCols();
            fh.float_size = sizeof(T);
            out.write(reinterpret_cast<char*>(&fh), sizeof(fh));
            for (begin(); !end(); next()) {
                out.write(reinterpret_cast<char*>(&(getRow().size)), sizeof(int));
                out.write(reinterpret_cast<char*>(getRow().values), sizeof(T)*getRow().size);
                out.write(reinterpret_cast<char*>(getRow().ids), sizeof(int)*getRow().size);
            }
            out.close();
        } else {
            throw "could not open " + filename;
        }
    }
};

/*
 * 设数组格式如下：
 * [[1, 0, 2],
 * [0, 0, 3],
 * [4, 5, 6]]
 * data 表示 元数据 显然为1， 2， 3， 4， 5， 6
 * indices 表示 各个数据在各行的下标， 从该数据我们可以知道：数据1在某行的0位置处， 数据2在某行的2位置处，6在某行的2位置处。
 * indptr 表示每行数据的个数：[0 2 3 6]表示从第0行开始数据的个数，0表示默认起始点，0之后有几个数字就表示有几行，第一个数字2表示第一行有2 - 0 = 2个数字，因而数字1，2都第0行，第二行有3 - 2 = 1个数字，因而数字3在第1行，以此类推，我们能够知道所有数字的行号
 * col2group 表示各行下标到组的映射
 * */
template <typename T> class LargeSparseMatrixMemory : public LargeSparseMatrix<T> {
protected:
    int index;
public:
    LargeSparseMatrixMemory(int num_cols){
        this->num_cols = num_cols;
        col2group = new int[num_cols];
    }
    ~LargeSparseMatrixMemory(){
        if(data != nullptr)
            delete []data;
        if(indices != nullptr)
            delete []indices;
        if(indptr != nullptr)
            delete []indptr;
        delete []col2group;
    }

    void fill(T* data, int* indices, int* indptr, int* col2group, int num_rows, uint64 num_values){
        if(col2group == nullptr){
            for(int i = 0; i < num_cols; ++i){
                this->col2group[i] = i;
            }
        } else {
            memcpy(this->col2group, col2group, sizeof(int) * this->num_cols);
        }

        if(num_rows > max_num_rows){
            if(this->indptr != nullptr)
                delete []this->indptr;
            this->indptr = new int[num_rows + 1];
            max_num_rows = num_rows;
        }
        this->num_rows = num_rows;
        memcpy(this->indptr, indptr, (num_rows + 1) * sizeof(int));

        if(num_values > max_num_values){
            if(this->data != nullptr)
                delete []this->data;
            this->data = new T[num_values];
            if(this->indices != nullptr)
                delete []this->indices;
            this->indices = new int[num_values];
            max_num_values = num_values;
        }
        this->num_values = num_values;
        memcpy(this->data, data, num_values * sizeof(T));
        memcpy(this->indices, indices, num_values * sizeof(int));
//        for(int i = 0; i < num_values; ++i)
//            std::cout<<this->data[i]<<" ";
//        std::cout<<"\n";
    }

    T* data = {nullptr};
    int* indices = {nullptr};
    int* indptr = {nullptr};
    int* col2group = {nullptr};

    int num_cols;
    int num_rows = {0};
    int max_num_rows = {0};
    uint64 num_values = {0};
    uint64 max_num_values = {0};
    SparseRow<T> curRow;
    virtual void begin() {
        index = 0;
        curRow.ids = this->indices + this->indptr[index];
        curRow.values = this->data + this->indptr[index];
        curRow.size = this->indptr[index + 1] - this->indptr[index];
        curRow.col2group = this->col2group;
    };
    virtual bool end() { return index >= num_rows; }
    virtual void next() {
        index++;
        curRow.ids = this->indices + this->indptr[index];
        curRow.values = this->data + this->indptr[index];
        curRow.size = this->indptr[index + 1] - this->indptr[index];
    }
    virtual SparseRow<T>& getRow() { return curRow; };
    virtual int getRowIndex() { return index; };
    virtual int getNumRows() { return num_rows; };
    virtual int getNumCols() { return num_cols; };
    virtual uint64 getNumValues() { return num_values; };
};


#endif //PYLIBFM_FMATRIX_H
