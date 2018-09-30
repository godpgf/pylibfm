//
// Created by godpgf on 18-9-28.
// 自动调整正则项的sgd,具体思想是固定正则项优化其他参数，再固定其它参数优化正则项,原文如下：
// starting with an initial guess of (Θ, λ) and alter-
// nate between improving Θ while λ is fixed and improving
// λ while Θ is fixed.
// 实现上也很简单，直接固定 Θ 对 λ 求偏导数，让它等于0，然后梯度下降就可以。具体推倒方式参见原文。
// 原文已经下载下来放到doc目录下面。
//

#ifndef LIBFM_LEARN_SGD_ELEMENT_ADAPT_REG_H
#define LIBFM_LEARN_SGD_ELEMENT_ADAPT_REG_H

#include "learn_sgd.h"

class FMLearnSGDAElement: public FMLearnSGD{
public:
    virtual void init(){
        FMLearnSGD::init();

        reg_0 = 0;
        reg_w.setSize(fm->num_group);
        reg_v.setSize(fm->num_group, fm->num_factor);

        mean_v.setSize(fm->num_factor);
        var_v.setSize(fm->num_factor);

        grad_w.setSize(fm->num_attribute);
        grad_v.setSize(fm->num_factor, fm->num_attribute);

        grad_w.init(0);
        grad_v.init(0);

        lambda_w_grad.setSize(fm->num_group);
        sum_f.setSize(fm->num_group);
        sum_f_dash_f.setSize(fm->num_group);

        // make sure that fm-parameters are initialized correctly (no other side effects)
        fm->w.init(0);
        fm->reg0 = 0;
        fm->regw = 0;
        fm->regv = 0;

        // start with no regularization
        reg_w.init(0.0);
        reg_v.init(0.0);
    }

    virtual void learn(LargeSparseMatrix<float> &x, DVector<float>& target){
        FMLearnSGD::learn(x, target);
        update_means();
        x.begin();
        while (!x.end()){
            sgd_theta_step(x.getRow(), target.get(x.getRowIndex()));
            x.next();
            if(rand() % 2 == 0 && !x.end()){
                // 交叉验证
                sgd_lambda_step(x.getRow(), target.get(x.getRowIndex()));
                x.next();
            }

        }
    }

protected:
    void update_means() {
        mean_w = 0;
        mean_v.init(0);
        var_w = 0;
        var_v.init(0);
        for (uint j = 0; j < fm->num_attribute; j++) {
            mean_w += fm->w(j);
            var_w += fm->w(j)*fm->w(j);
            for (int f = 0; f < fm->num_factor; f++) {
                mean_v(f) += fm->v(f,j);
                var_v(f) += fm->v(f,j)*fm->v(f,j);
            }
        }
        mean_w /= (float) fm->num_attribute;
        var_w = var_w/fm->num_attribute - mean_w*mean_w;
        for (int f = 0; f < fm->num_factor; f++) {
            mean_v(f) /= fm->num_attribute;
            var_v(f) = var_v(f)/fm->num_attribute - mean_v(f)*mean_v(f);
        }

        mean_w = 0;
        for (int f = 0; f < fm->num_factor; f++) {
            mean_v(f) = 0;
        }
    }

    void sgd_theta_step(SparseRow<float>& x, const float target) {
        float p = fm->predict(x, sum, sum_sqr);
        float mult = 0;
        if (task == TASK_REGRESSION) {
            p = std::fminf(fm->max_target, p);
            p = std::fmaxf(fm->min_target, p);
            mult = 2 * (p - target);
        } else if (task == TASK_CLASSIFICATION) {
            mult = target * (  (1.0/(1.0+exp(-target*p))) - 1.0 );
        }

        // make the update with my regularization constants:
        if (fm->is_use_w0) {
            float& w0 = fm->w0;
            float grad_0 = mult;
            w0 -= learn_rate * (grad_0 + 2 * reg_0 * w0);
        }
        if (fm->is_use_w) {
            for (uint i = 0; i < x.size; i++) {
                uint g = x.col2group[x.ids[i]];
                float& w = fm->w(x.ids[i]);
                grad_w(x.ids[i]) = mult * x.values[i];
                w -= learn_rate * (grad_w(x.ids[i]) + 2 * reg_w(g) * w);
            }
        }
        for (int f = 0; f < fm->num_factor; f++) {
            for (uint i = 0; i < x.size; i++) {
                uint g = x.col2group[x.ids[i]];
                float& v = fm->v(f,x.ids[i]);
                grad_v(f,x.ids[i]) = mult * (x.values[i] * (sum(f) - v * x.values[i])); // grad_v_if = (y(x)-y) * [ x_i*(\sum_j x_j v_jf) - v_if*x^2 ]
                v -= learn_rate * (grad_v(f,x.ids[i]) + 2 * reg_v(g,f) * v);
            }
        }
    }

    float predict_scaled(SparseRow<float>& x) {
        float p = 0.0;
        if (fm->is_use_w0) {
            p += fm->w0;
        }
        if (fm->is_use_w) {
            for (uint i = 0; i < x.size; i++) {
                assert(x.ids[i] < fm->num_attribute);
                uint g = x.col2group[x.ids[i]];
                float& w = fm->w(x.ids[i]);
                float w_dash = w - learn_rate * (grad_w(x.ids[i]) + 2 * reg_w(g) * w);
                p += w_dash * x.values[i];
            }
        }
        for (int f = 0; f < fm->num_factor; f++) {
            sum(f) = 0.0;
            sum_sqr(f) = 0.0;
            for (uint i = 0; i < x.size; i++) {
                uint g = x.col2group[x.ids[i]];
                float& v = fm->v(f,x.ids[i]);
                float v_dash = v - learn_rate * (grad_v(f,x.ids[i]) + 2 * reg_v(g,f) * v);
                float d = v_dash * x.values[i];
                sum(f) += d;
                sum_sqr(f) += d*d;
            }
            p += 0.5 * (sum(f)*sum(f) - sum_sqr(f));
        }
        return p;
    }

    void sgd_lambda_step(SparseRow<float>& x, const float target) {
        float p = predict_scaled(x);
        float grad_loss = 0;
        if (task == 0) {
            p = std::fminf(fm->max_target, p);
            p = std::fmaxf(fm->min_target, p);
            grad_loss = 2 * (p - target);
        } else if (task == 1) {
            grad_loss = target * ( (1.0/(1.0+exp(-target*p))) -  1.0);
        }

        if (fm->is_use_w0) {
            lambda_w_grad.init(0.0);
            for (uint i = 0; i < x.size; i++) {
                uint g = x.col2group[x.ids[i]];
                lambda_w_grad(g) += x.values[i] * fm->w(x.ids[i]);
            }
            for (uint g = 0; g < fm->num_group; g++) {
                lambda_w_grad(g) = -2 * learn_rate * lambda_w_grad(g);
                reg_w(g) -= learn_rate * grad_loss * lambda_w_grad(g);
                reg_w(g) = std::fmaxf(0.0, reg_w(g));
            }
        }
        for (int f = 0; f < fm->num_factor; f++) {
            // grad_lambdafg = (grad l(y(x),y)) * (-2 * alpha * (\sum_{l} x_l * v'_lf) * (\sum_{l \in group(g)} x_l * v_lf) - \sum_{l \in group(g)} x^2_l * v_lf * v'_lf)
            // sum_f_dash      := \sum_{l} x_l * v'_lf, this is independent of the groups
            // sum_f(g)        := \sum_{l \in group(g)} x_l * v_lf
            // sum_f_dash_f(g) := \sum_{l \in group(g)} x^2_l * v_lf * v'_lf
            float sum_f_dash = 0.0;
            sum_f.init(0.0);
            sum_f_dash_f.init(0.0);
            for (uint i = 0; i < x.size; i++) {
                // v_if' =  [ v_if * (1-alpha*lambda_v_f) - alpha * grad_v_if]
                uint g = x.col2group[x.ids[i]];
                float& v = fm->v(f,x.ids[i]);
                float v_dash = v - learn_rate * (grad_v(f,x.ids[i]) + 2 * reg_v(g,f) * v);

                sum_f_dash += v_dash * x.values[i];
                sum_f(g) += v * x.values[i];
                sum_f_dash_f(g) += v_dash * x.values[i] * v * x.values[i];
            }
            for (uint g = 0; g < fm->num_group; g++) {
                float lambda_v_grad = -2 * learn_rate *  (sum_f_dash * sum_f(g) - sum_f_dash_f(g));
                reg_v(g,f) -= learn_rate * grad_loss * lambda_v_grad;
                reg_v(g,f) = std::fmaxf(0.0, reg_v(g,f));
            }
        }
    }
protected:
    // regularization parameter
    float reg_0; // shrinking the bias towards the mean of the bias (which is the bias) is the same as no regularization.

    DVector<float> reg_w;
    DMatrix<float> reg_v;

    float mean_w, var_w;
    DVector<float> mean_v, var_v;

    // for each parameter there is one gradient to store
    DVector<float> grad_w;
    DMatrix<float> grad_v;

    // local parameters in the lambda_update step
    DVector<float> lambda_w_grad;
    DVector<float> sum_f, sum_f_dash_f;
};

#endif //LIBFM_LEARN_SGD_ELEMENT_ADAPT_REG_H
