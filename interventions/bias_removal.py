#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from aif360.sklearn import metrics
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover, OptimPreproc
from aif360.algorithms.inprocessing import AdversarialDebiasing, ExponentiatedGradientReduction
import tensorflow as tf
import os
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def get_bias_removed_attributes(attributes_df, bias_labels_df):

    favorable = 0
    unfavorable = 1

    # add bias label
    df = attributes_df.merge(bias_labels_df, on='url', how='inner')

    privileged_groups = [{'bias':-2}]
    unprivileged_groups = [{'bias':2}]

    label='label'
    df.groupby([label,'bias']).describe()

    df_output_append = df[['url', 'label']]
    df.drop(columns=['url'], inplace=True)
    # df.drop(columns=['url', 'source'], inplace=True)
    df.dropna(inplace=True)

    df = df.sample(frac=1, random_state=42)
    df['label'] = df['label'].apply(lambda x: 1 if x <= 4 else 0)
    df.groupby([label,'bias']).describe()

    chosen_attribute = 'bias'
    split = 0.7
    binary_label_dataset = BinaryLabelDataset(
        favorable_label=favorable,
        unfavorable_label=unfavorable,
        df=df,
        label_names=[label],
        protected_attribute_names=[chosen_attribute])
    orig_train, orig_test = binary_label_dataset.split([split], shuffle=False)

    clf = RandomForestRegressor(n_estimators=50, random_state=0)
    clf.fit(orig_train.features, orig_train.labels.ravel())
    test_orig_pred = orig_test.copy()
    test_orig_pred.labels = clf.predict(orig_test.features) > 0.75

    original_dataset_metric = BinaryLabelDatasetMetric(test_orig_pred, 
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)

    index = orig_train.feature_names.index(chosen_attribute)

    aucs = []
    acc = []
    f1 = []
    DIs = []
    spds = []
    for level in tqdm(np.linspace(0., 1., 11)):
        di = DisparateImpactRemover(repair_level=level, sensitive_attribute=chosen_attribute)
        train_repd = di.fit_transform(orig_train)
        test_repd = di.fit_transform(orig_test)

        X_tr = np.delete(train_repd.features, index, axis=1)
        X_te = np.delete(test_repd.features, index, axis=1)
        y_tr = train_repd.labels.ravel()
        
        clf = RandomForestRegressor(n_estimators=50, random_state=0)
        clf.fit(X_tr, y_tr)

        # lmod = LogisticRegression(class_weight='balanced', solver='liblinear')
        # lmod.fit(X_tr, y_tr)
        
        test_repd_pred = test_repd.copy()
        test_repd_pred.labels = clf.predict(X_te) > 0.75
        aucs.append(roc_auc_score(test_repd.labels.ravel(), test_repd_pred.labels))
        acc.append(accuracy_score(test_repd.labels.ravel(), test_repd_pred.labels))
        f1.append(f1_score(test_repd.labels.ravel(), test_repd_pred.labels, average='macro'))

        cm = BinaryLabelDatasetMetric(test_repd_pred, privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
        DIs.append(cm.disparate_impact())
        spds.append(cm.statistical_parity_difference())

    def delta(arr):
        return arr
        # mean = arr[0]#np.mean(arr)
        # return [x - mean for x in arr]

    plt.plot(np.linspace(0, 1, 11), delta(DIs), marker='.')
    # plt.plot(np.linspace(0, 1, 11), mean_shift(spds), marker='')
    plt.plot(np.linspace(0, 1, 11), delta(f1), marker='.')

    # set legend labels
    plt.legend(['DI(r) - DI(0)', 'F1(r) - F1(0)'])
    plt.title('Disparate Impact Improves as Performance Falls')
    plt.ylabel('Change in measure vs R = 0')#DI(r) - DI(0)
    plt.xlabel('Repair level R') #F1(r) - F1(0)
    plt.show()
    plt.savefig('../results/transformed/repair_level.png')

    di = DisparateImpactRemover(repair_level=1, sensitive_attribute=chosen_attribute)
    transformed_dataset = di.fit_transform(binary_label_dataset)

    print("{:<30} {:<15} {:<15}".format('Experiment', 'DI', 'SPD'))
    print('-'*80)
    print("{:<30} {:<15} {:<15}".format('Original data', f"{original_dataset_metric.disparate_impact():.4f}", f"{original_dataset_metric.statistical_parity_difference():.4f}"))

    transformed_dataset_metric = BinaryLabelDatasetMetric(transformed_dataset, privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
    print("{:<30} {:<15} {:<15}".format('Transformed data', f"{transformed_dataset_metric.disparate_impact():.4f}",  f"{transformed_dataset_metric.statistical_parity_difference():.4f}"))
    # create df out of numpy array
    transformed_attributes_df = pd.DataFrame(transformed_dataset.features, columns=binary_label_dataset.feature_names)
    transformed_attributes_df['url'] = df_output_append['url']
    transformed_attributes_df['label'] = df_output_append['label']
    return transformed_attributes_df

# if main
if __name__ == '__main__':
    df = pd.read_csv('../data/filtered_attrs.csv')
    df_categorical = df[['url', 'label']]
    df.drop(columns=['url', 'label', 'source'], inplace=True)
    # log all attrbiutes
    df = df.apply(lambda x: np.log(x + 1))
    # normalize all attributes
    df = (df - df.mean()) / df.std()
    df = df.fillna(0)
    df[['url', 'label']] = df_categorical

    bias_labels_df = pd.read_csv('../data/bias_labels.csv')
    df_output = get_bias_removed_attributes(df, bias_labels_df)
    df_output.to_csv('results/transformed_features.csv', index=False)
