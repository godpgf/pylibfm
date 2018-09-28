//
// Created by yanyu on 2017/7/4.
// 特征机器的模型
//

#ifndef PYLIBFM_MODEL_H
#define PYLIBFM_MODEL_H

#include "../util/matrix.h"
#include "../util/fmatrix.h"

enum TaskType {
    TASK_REGRESSION = 0,
    TASK_CLASSIFICATION = 1,
};

class FMModel {
private:
    //存储中间数据
    DVector<float> m_sum, m_sum_sqr;
public:
    double w0;
    DVectorDouble w;
    DMatrixDouble v;
public:
    //特征数
    uint num_attribute;
    bool is_use_w0, is_use_w;
    //压缩后的因子数
    int num_factor;

    //在做回归的时候,需要限制结果的取值范围,这是一个提高准确率的优化.
    float min_target;
    float max_target;

    //正则项,用来防止过拟合(在损失函数中加入,防止w的斜率太大)
    double reg0;
    double regw, regv;

    //初始化v时用的均值和标准差
    double init_stdev;
    double init_mean;
public:
    FMModel();

    void debug();

    void init();

    //得到当前行的预测值
    double predict(SparseRow<float> &x);

    double predict(SparseRow<float> &x, DVector<float> &sum, DVector<float> &sum_sqr);

    void saveModel(std::string model_file_path);

    int loadModel(std::string model_file_path);

private:
    void splitString(const std::string &s, char c, std::vector<std::string> &v);
};

FMModel::FMModel() {
    num_factor = 0;
    reg0 = 0.0;
    regw = 0.0;
    regv = 0.0;
    is_use_w0 = true;
    is_use_w = true;
    init_mean = 0.0;
    init_stdev = 0.1;
}

void FMModel::debug() {
    std::cout << "num_attributes=" << num_attribute << std::endl;
    std::cout << "range (" << min_target << "," << max_target << ")" << std::endl;
    std::cout << "use w0=" << is_use_w0 << std::endl;
    std::cout << "use w1=" << is_use_w << std::endl;
    std::cout << "dim v =" << num_factor << std::endl;
    std::cout << "reg_w0=" << reg0 << std::endl;
    std::cout << "reg_w=" << regw << std::endl;
    std::cout << "reg_v=" << regv << std::endl;
    std::cout << "init ~ N(" << init_mean << "," << init_stdev << ")" << std::endl;
}

void FMModel::init() {
    w0 = 0;
    w.setSize(num_attribute);
    v.setSize(num_factor, num_attribute);
    w.init(0);
    v.init(init_mean, init_stdev);
    m_sum.setSize(num_factor);
    m_sum_sqr.setSize(num_factor);
    min_target = max_target = 0.0;
}

double FMModel::predict(SparseRow<float> &x) {
    return predict(x, m_sum, m_sum_sqr);
}

double FMModel::predict(SparseRow<float> &x, DVector<float> &sum, DVector<float> &sum_sqr) {
    // 参加doc下面的论文
    double result = 0;
    if (is_use_w0) {
        result += w0;
//        std::cout<<"w0 "<<w0<<std::endl;
    }
    if (is_use_w) {
        for (uint i = 0; i < x.size; i++) {
            assert(x.ids[i] < num_attribute);
            result += w(x.ids[i]) * x.values[i];
//            std::cout<<"id "<<x.ids[i]<<" w1 "<<w(x.ids[i])<<" x"<<x.values[i]<<std::endl;
        }
    }
    for (int f = 0; f < num_factor; f++) {
        sum(f) = 0;
        sum_sqr(f) = 0;
        for (uint i = 0; i < x.size; i++) {
            double d = v(f, x.ids[i]) * x.values[i];
            sum(f) += d;
            sum_sqr(f) += d * d;
//            std::cout<<"d:"<<v(f, x.ids[i])<<std::endl;
        }
        result += 0.5 * (sum(f) * sum(f) - sum_sqr(f));
    }
    return result;
}

/*
 * Write the FM model (all the parameters) in a file.
 */
void FMModel::saveModel(std::string model_file_path) {
    std::ofstream out_model;
    out_model.open(model_file_path.c_str());

    //modify by yanyu
    out_model << "#global target range" << std::endl;
    out_model << min_target << ' ' << max_target << std::endl;

    if (is_use_w0) {
        out_model << "#global bias W0" << std::endl;
        out_model << w0 << std::endl;
    }
    if (is_use_w) {
        out_model << "#unary interactions Wj" << std::endl;
        for (uint i = 0; i < num_attribute; i++) {
            out_model << w(i) << std::endl;
        }
    }
    out_model << "#pairwise interactions Vj,f" << std::endl;
    for (uint i = 0; i < num_attribute; i++) {
        for (int f = 0; f < num_factor; f++) {
            out_model << v(f, i);
            if (f != num_factor - 1) { out_model << ' '; }
        }
        out_model << std::endl;
    }
    out_model.close();
}

/*
 * Read the FM model (all the parameters) from a file.
 * If no valid conversion could be performed, the function std::atof returns zero (0.0).
 */
int FMModel::loadModel(std::string model_file_path) {
    std::string line;
    std::ifstream model_file(model_file_path.c_str());
    if (model_file.is_open()) {
        if (!std::getline(model_file, line)) { return 0; } // "#global target range"
        if (!std::getline(model_file, line)) { return 0; }
        std::vector<std::string> range_str;
        splitString(line, ' ', range_str);
        if ((int) range_str.size() != 2) { return 0; }
        min_target = std::atof(range_str[0].c_str());
        max_target = std::atof(range_str[1].c_str());

        if (is_use_w0) {
            if (!std::getline(model_file, line)) { return 0; } // "#global bias W0"
            if (!std::getline(model_file, line)) { return 0; }
            w0 = std::atof(line.c_str());
        }
        if (is_use_w) {
            if (!std::getline(model_file, line)) { return 0; } //"#unary interactions Wj"
            for (uint i = 0; i < num_attribute; i++) {
                if (!std::getline(model_file, line)) { return 0; }
                w(i) = std::atof(line.c_str());
            }
        }
        if (!std::getline(model_file, line)) { return 0; }; // "#pairwise interactions Vj,f"
        for (uint i = 0; i < num_attribute; i++) {
            if (!std::getline(model_file, line)) { return 0; }
            std::vector<std::string> v_str;
            splitString(line, ' ', v_str);
            if ((int) v_str.size() != num_factor) { return 0; }
            for (int f = 0; f < num_factor; f++) {
                v(f, i) = std::atof(v_str[f].c_str());
            }
        }
        model_file.close();
    } else { return 0; }
    return 1;
}

/*
 * Splits the string s around matches of the given character c, and stores the substrings in the vector v
 */
void FMModel::splitString(const std::string &s, char c, std::vector<std::string> &v) {
    std::string::size_type i = 0;
    std::string::size_type j = s.find(c);
    while (j != std::string::npos) {
        v.push_back(s.substr(i, j - i));
        i = ++j;
        j = s.find(c, j);
        if (j == std::string::npos)
            v.push_back(s.substr(i, s.length()));
    }
}

#endif //PYLIBFM_MODEL_H
