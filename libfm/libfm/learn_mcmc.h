//
// Created by godpgf on 18-9-30.
//

#ifndef LIBFM_LEARN_MCMC_H
#define LIBFM_LEARN_MCMC_H

#include "learn.h"

struct e_q_term {
    float e;
    float q;
};

struct relation_cache {
    float /*uint*/ wnum;   // #
    float q;               // q_if^B
    float wc;              // c_if^B
    float wc_sqr;          // c_if^B,S
    float y;               // y_i^B
    float we;              // e_i
    float weq;             // e_if^B,q
};

class FMLearnMCMC : FMLearn {
public:
    void init() {
        FMLearn::init();

        cache_for_group_values.setSize(fm->num_group);

        alpha_0 = 1.0;
        gamma_0 = 1.0;
        beta_0 = 1.0;
        mu_0 = 0.0;

        alpha = 1;

        w0_mean_0 = 0.0;

        w_mu.setSize(fm->num_group);
        w_lambda.setSize(fm->num_group);
        w_mu.init(0.0);
        w_lambda.init(0.0);

        v_mu.setSize(fm->num_group, fm->num_factor);
        v_lambda.setSize(fm->num_group, fm->num_factor);
        v_mu.init(0.0);
        v_lambda.init(0.0);
    }

    virtual void learn(LargeSparseMatrix<float> &x, DVector<float>& target){
        FMLearn::learn(x, target);
        _learn(x, target);
    }

    virtual void predict(LargeSparseMatrix<float> &x, DVector<float> &out) {
        pred_sum_all.setSize(x.getNumRows());
        pred_sum_all_but5.setSize(x.getNumRows());
        pred_this.setSize(x.getNumRows());
        pred_sum_all.init(0.0);
        pred_sum_all_but5.init(0.0);
        pred_this.init(0.0);

        cache.setSize(x.getNumRows());
    }

protected:
    virtual void _learn(LargeSparseMatrix<float> &x, DVector<float>& target) = 0;

    void predict_data_and_write_to_eterms(LargeSparseMatrix<float> &x, DVector<e_q_term> &cache) {

        for(uint i = 0; i < x.getNumRows(); ++i){
            e_q_term& c = cache.get(i);
            c.e = 0;
            c.q = 0;
        }


        // (1) do the 1/2 sum_f (sum_i v_if x_i)^2 and store it in the e/y-term
        // (1.1) e_j = 1/2 sum_f (q_jf+ sum_R q^R_jf)^2
        // (1.2) y^R_j = 1/2 sum_f q^R_jf^2
        // Complexity: O(N_z(X^M) + \sum_{B} N_z(X^B) + n*|B| + \sum_B n^B) = O(\mathcal{C})
        for (int f = 0; f < fm->num_factor; f++) {
            // calculate cache[i].q = sum_i v_if x_i (== q_f-term)
            // Complexity: O(N_z(X^M))
            float* v = fm->v.value + (f * fm->v.dim2);
            x.beginT();
            uint row_index;
            SparseRow<float>* feature_data;
            for(uint i = 0; i < x.getNumCols(); ++i){
                {
                    row_index = x.getRowIndexT();
                    feature_data = &x.getRowT();
                    x.nextT();
                }
                float& v_if = v[row_index];

                for (uint i_fd = 0; i_fd < feature_data->size; i_fd++) {
                    int& train_case_index = feature_data->ids[i_fd];
                    float& x_li = feature_data->values[i_fd];
                    e_q_term& c = cache.get(train_case_index);
                    c.q += v_if * x_li;
                }
            }
        }

        // (2) do -1/2 sum_f (sum_i v_if^2 x_i^2) and store it in the q-term
        for (int f = 0; f < fm->num_factor; f++) {
            float* v = fm->v.value + (f * fm->v.dim2);

            // sum up the q^S_f terms in the main-q-cache: 0.5*sum_i (v_if x_i)^2 (== q^S_f-term)
            // Complexity: O(N_z(X^M))
            x.beginT();
            uint row_index;
            SparseRow<float>* feature_data;
            for (uint i = 0; i < x.getNumCols(); i++) {
                {
                    row_index = x.getRowIndexT();
                    feature_data = &x.getRowT();
                    x.nextT();
                }
                float& v_if = v[row_index];

                for (uint i_fd = 0; i_fd < feature_data->size; i_fd++) {
                    int& train_case_index = feature_data->ids[i_fd];
                    float& x_li = feature_data->values[i_fd];
                    e_q_term& c = cache.get(train_case_index);
                    c.q -= 0.5 * v_if * v_if * x_li * x_li;
                }
            }

        }

        // (3) add the w's to the q-term
        if (fm->is_use_w) {
            x.beginT();
            uint row_index;
            SparseRow<float>* feature_data;
            for (uint i = 0; i < x.getNumCols(); i++) {
                {
                    row_index = x.getRowIndexT();
                    feature_data = &x.getRowT();
                    x.nextT();
                }
                float& w_i = fm->w(row_index);

                for (uint i_fd = 0; i_fd < feature_data->size; i_fd++) {
                    int& train_case_index = feature_data->ids[i_fd];
                    float& x_li = feature_data->values[i_fd];
                    e_q_term& c = cache.get(train_case_index);
                    c.q += w_i * x_li;
                }
            }

        }
        // (3) merge both for getting the prediction: w0+e(c)+q(c)
        for (uint i = 0; i < x.getNumRows(); i++) {
            e_q_term& c = cache.get(i);
            if (fm->is_use_w0) {

                c.e += fm->w0;
            }
            c.q = 0.0;
        }
    }
protected:

    // Hyperpriors
    float alpha_0, gamma_0, beta_0, mu_0;
    float w0_mean_0;

    // Priors
    float alpha;
    DVector<float> w_mu, w_lambda;
    DMatrix<float> v_mu, v_lambda;

    // switch between choosing expected values and drawing from distribution
    bool do_sample;
    // use the two-level (hierarchical) model (TRUE) or the one-level (FALSE)
    bool do_multilevel;

    uint nan_cntr_v, nan_cntr_w, nan_cntr_w0, nan_cntr_alpha, nan_cntr_w_mu, nan_cntr_w_lambda, nan_cntr_v_mu, nan_cntr_v_lambda;
    uint inf_cntr_v, inf_cntr_w, inf_cntr_w0, inf_cntr_alpha, inf_cntr_w_mu, inf_cntr_w_lambda, inf_cntr_v_mu, inf_cntr_v_lambda;

    DVector<float> cache_for_group_values;

    // A dummy row for attributes that exist only in the test data.
    LargeSparseMatrix<float> empty_data_row;

    DVector<float> pred_sum_all;
    DVector<float> pred_sum_all_but5;
    DVector<float> pred_this;

    DVector<e_q_term> cache;
};

#endif //LIBFM_LEARN_MCMC_H
