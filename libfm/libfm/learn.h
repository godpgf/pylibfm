//
// Created by yanyu on 2017/7/5.
//

#ifndef PYLIBFM_LEARN_H
#define PYLIBFM_LEARN_H

#include <cmath>
#include "../core/model.h"

class FMLearn {
    protected:
        //缓存中间计算结果,避免反复申请释放内存
        DVector<double> sum, sum_sqr;
        //预测某个数据的结果,在MCMC算法中会被重载
        virtual double predict_case(LargeSparseMatrix<float> &x){
            //预测矩阵当前行的结果
            return fm->predict(x.getRow());
        }

    public:
        FMModel* fm;
        TaskType task;

        FMLearn(){task = TASK_REGRESSION;}
        virtual ~FMLearn(){}

        virtual void init(){
            sum.setSize(fm->num_factor);
            sum_sqr.setSize(fm->num_factor);
        }

        //验证某个输入的准确率
        virtual double evaluate(LargeSparseMatrix<float> &x, DVector<double> &target){
            if (task == TASK_REGRESSION){
                return evaluateRegression(x, target);
            } else if (task == TASK_CLASSIFICATION){
                return evaluateClassification(x, target);
            } else {
                throw "unknown task";
            }
        }

        virtual void learn(LargeSparseMatrix<float> &x, DVector<double>& target){
            //计算结果的最大最小值
            if (task == TASK_REGRESSION){
                fm->min_target = +std::numeric_limits<float>::max();
                fm->max_target = -std::numeric_limits<float>::max();
                for (uint i = 0; i < target.dim; i++) {
                    fm->min_target = std::min(target(i), fm->min_target);
                    fm->max_target = std::max(target(i), fm->max_target);
                }
                //std::cout<<fm->min_target<<" "<<fm->max_target<<std::endl;
            }
        }

        virtual void predict(LargeSparseMatrix<float> &x, DVector<double>& out) = 0;
    protected:
        virtual double evaluateClassification(LargeSparseMatrix<float> &x, DVector<double> &target){
            int num_correct = 0;
            for (x.begin(); !x.end(); x.next()) {
                double p = predict_case(x);
                if (((p >= 0) && (target(x.getRowIndex()) >= 0)) || ((p < 0) && (target(x.getRowIndex()) < 0))) {
                    num_correct++;
                }
            }
            return (double) num_correct / (double) x.getNumRows();
        }

        virtual double evaluateRegression(LargeSparseMatrix<float> &x, DVector<double> &target){
            double rmse_sum_sqr = 0;
            double mae_sum_abs = 0;
            for (x.begin(); !x.end(); x.next()) {
                double p = predict_case(x);
                p = std::min(fm->max_target, p);
                p = std::max(fm->min_target, p);
                double err = p - target(x.getRowIndex());
                rmse_sum_sqr += err*err;
                mae_sum_abs += std::abs((double)err);
            }
            return std::sqrt(rmse_sum_sqr/x.getNumRows());
        }
};

#endif //PYLIBFM_LEARN_H
