#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from functools import reduce
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import warnings
import pandas as pd
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import sys
sys.path.insert(0,'..')
import interventions.backlink_relevance_combined as intervention
import seaborn as sns
import matplotlib.pyplot as plt

log = True

def compute_diff(res_df, pre_col, post_col):
    res_df[post_col] = res_df[post_col].clip(lower=0)
    try:
        if log:
            res_df['diff'] = np.exp(res_df[post_col]) / np.exp(res_df[pre_col])
        else:
            res_df['diff'] = (res_df[post_col]) / (res_df[pre_col])
    except:
        res_df['diff'] = 1
    return res_df

def pre_post_intervention_diff(weighted_df, reg_model, reg_attrs):
    df = weighted_df.copy()
    df['pre_backlinks'] = np.log(df['pre_backlinks']+1)
    df['post_backlinks'] = np.log(df['post_backlinks']+1)
    df['label'].replace({1:3, 6:5}, inplace=True)
    df['reg_before'] = reg_model.predict(df[['pre_backlinks'] + reg_attrs])
    df['reg_after'] = reg_model.predict(df[['post_backlinks'] + reg_attrs])
    return compute_diff(df, 'reg_before', 'reg_after')

def run_regression(traffic_df, attribute_df, reg_var = 'traffic', experiments = None, log = True):
    urls_to_remove = ['youtube.com', 'facebook.com']
    for url in urls_to_remove:
        traffic_df = traffic_df[~traffic_df['url'].str.contains(url)]
        attribute_df = attribute_df[~attribute_df['url'].str.contains(url)]

    attribute_df.dropna(inplace=True)
    url_df = attribute_df.copy() 

    run_exprs = experiments is None
    if run_exprs:
        print(f"Running {reg_var} interventions")
        experiments = {
            "\\textbf{L}ink Scheme": intervention.run_combined_intervention(link_scheme_removal=True, multiplicity_weighting=False, attribute_df=attribute_df)[0],
            "\\textbf{M}ultiplicity": intervention.run_combined_intervention(link_scheme_removal=False, multiplicity_weighting=True, attribute_df=attribute_df)[0],
            "L+M Combined": intervention.run_combined_intervention(link_scheme_removal=True, multiplicity_weighting=True, attribute_df=attribute_df)[0],
        }
        print(f"Finished {reg_var} interventions")

    features_to_keep = ['backlinks', 'html_pages', 'links_external']
    X_train = url_df[features_to_keep].clip(lower=0)
    if log:
        X_train = np.log(1+X_train)

    y_train = np.log(1+traffic_df[reg_var]) if reg_var == 'traffic' else traffic_df[reg_var]
    labels = url_df['label']

    reg_model = sm.OLS(y_train, X_train).fit()
    if run_exprs:
        print(reg_model.summary())

    def pre_post_intervention_diff_(weighted_df, train_df, reg_model, reg_attrs):
        weighted_df = weighted_df[['url', 'pre_backlinks', 'pre_refpages', 'post_backlinks', 'post_refpages']]
        train_df['url'] = url_df['url']
        weighted_df = pd.merge(weighted_df, train_df, on='url', how='inner')
        return pre_post_intervention_diff(weighted_df, reg_model, reg_attrs)

    res_df = pd.DataFrame(columns=['name',3,4,5])
    X_train['label'] = labels
    for name, df in experiments.items():
        print('Running experiment:', name)
        dist_df = pre_post_intervention_diff_(df, X_train, reg_model, features_to_keep[1:])
        res = dist_df.drop(columns=['url']).groupby('label').mean()['diff']
        res_df.loc[len(res_df)] = [name]+ res.values.tolist()

        if log:
            dist_df['reg_before'] = np.exp(dist_df['reg_before'])
            dist_df['reg_after'] = np.exp(dist_df['reg_after'])
        melted_df = pd.melt(dist_df, id_vars=['label'], value_vars=['reg_before', 'reg_after'], var_name='group', value_name='value')
        # dist_plot = sns.boxplot(x='label', y='value', hue='group', data=melted_df, palette=['blue', 'orange'])

        # plt.legend(title='Group')
        # plt.xlabel('Label')
        # plt.ylabel(reg_var)
        # plt.title(f"Intervention: {name}, DV: {reg_var}")

        # fig = dist_plot.get_figure()
        # fig.savefig(f"../results/dists/{name}_{reg_var}_dist.png") 
        # plt.close(fig)

    if run_exprs:
        control_df = X_train.copy()
        control_df = control_df[features_to_keep]
        for control in [0, 0.5, 1]:
            control_exp_name = 'Control ' + str(int(control*100)) + "\\%"
            print('Running experiment:', control_exp_name)
            test_intervention = control_df.copy()
            test_res = test_intervention.copy()
            test_intervention['backlinks'] = np.log(np.exp(test_intervention['backlinks']) * control)

            test_res['reg_before'] = reg_model.predict(control_df)
            test_res['reg_after'] = reg_model.predict(test_intervention)
            test_res['label'] = labels

            test_res['label'].replace({1:3, 6:5}, inplace=True)

            res = compute_diff(test_res, 'reg_before', 'reg_after').groupby('label').mean()['diff']
            res_df.loc[len(res_df)] = [control_exp_name]+ res.values.tolist()

    res_df['metric'] = res_df[5] - ((res_df[4] + res_df[3])/2)
    return res_df, dist_df


if __name__ == '__main__':
    # read in data
    traffic_df = pd.read_csv('../data/traffic.csv')
    attribute_df = pd.read_csv('../data/filtered_attrs.csv')
    # run regression
    rank_res_df, _ = run_regression(traffic_df, attribute_df, reg_var='rank')
    traffic_res_df, _ = run_regression(traffic_df, attribute_df, reg_var='traffic')
    # save results
    traffic_res_df.columns = ['traffic_' + str(x) for x in traffic_res_df.columns]
    rank_res_df.columns = ['rank_' + str(x) for x in rank_res_df.columns]
    result = pd.concat([traffic_res_df,rank_res_df], axis=1)
    result.to_csv('../results/regression_results.csv', index=False)
    print(result)