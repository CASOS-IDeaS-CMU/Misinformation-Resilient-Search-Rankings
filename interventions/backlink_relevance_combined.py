#!/usr/bin/env python
# coding: utf-8

import networkx as nx
import pandas as pd
from functools import reduce
import numpy as np
from enum import Enum
import seaborn as sns
import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
import sys
sys.path.insert(0,'..')
from interventions.link_schemes import get_link_schemes

def percent_above_threshold(x, threshold):
    return np.sum(x > threshold) / len(x)

def get_source_domain_network_mapping(source_domain_relevancy_scores, reduced_link_network_path):
    # check if domain_name is a substring of any domain in domain_list
    def substr(domain_list, domain_name):
        for domain in domain_list:
            if domain_name in domain:
                return domain
        return None

    reduced_source_names = pd.read_csv(reduced_link_network_path)['domain_from'].unique()
    scraped_domains = source_domain_relevancy_scores.source_domain.values
    substr_dict = {substr(scraped_domains, dom): dom for dom in reduced_source_names}
    return substr_dict

def run_combined_intervention(
    link_scheme_removal = True,
    multiplicity_weighting = False,
    sample = 0.25, 
    attributes_path = '../data/filtered_attrs.csv',
    link_network_path = '../data/filtered_backlinks.csv',
    multiplicity_file = '../results/multiplicity_scores.csv',
    attribute_df = None,
):
    edge_df = pd.read_csv(link_network_path)
    edge_df.reset_index(inplace=True)
    edge_df['domain_from'] = edge_df['domain_from'].str.lower()
    edge_df['domain_to'] = edge_df['domain_to'].str.lower()
    # print('Source domains in network:',edge_df.domain_from.nunique())
    # print('Target domains in network:',edge_df.domain_to.nunique())

    attrs = ['links','unique_pages']#,'tb_ratio','so_ratio', 'e_tb_ratio','e_so_ratio','tp_ratio', 'sp_ratio']
    node_mapping = {k: v for v, k in enumerate(set(list(edge_df.domain_from.unique()) + list(edge_df.domain_to.unique())))}

    edge_df['domain_from_idx'] = edge_df.domain_from.map(node_mapping)
    edge_df['domain_to_idx'] = edge_df.domain_to.map(node_mapping)

    G_backlink = nx.from_pandas_edgelist(edge_df, source='domain_from_idx', target='domain_to_idx', edge_attr=attrs, create_using=nx.DiGraph())
    inv_node_mapping = {v: k for k, v in node_mapping.items()}

    len(G_backlink.nodes)

    if attribute_df is None:
        attributes_df = pd.read_csv(attributes_path)
    else:
        attributes_df = attribute_df.copy()
    attributes_df['url'] = attributes_df['url'].str.lower()
    url_df = attributes_df.copy()
    url_df = url_df[['url','label']]

    unreliable_labels = url_df[url_df['label']<=4]['url'].to_list()
    random.seed(11)
    unreliable_targets = list(set([node_mapping[t] for t in list(edge_df.domain_to.unique()) if t in unreliable_labels]))
    if sample:
        unreliable_targets = random.sample(unreliable_targets, int(len(unreliable_targets)*sample))

    if link_scheme_removal:
        df_schemes = get_link_schemes(G_backlink, unreliable_targets)
    else:
        df_schemes = []
    # print('Num link schemes: ', len(df_schemes))

    # Iterate over df rows and set the source and target nodes' attributes for each row as as G_backlink attribute:
    news_url_list = []
    for index, row in attributes_df.iterrows():
        try:
            G_backlink.nodes[node_mapping[row['url']]]['backlinks'] = row['backlinks']
            G_backlink.nodes[node_mapping[row['url']]]['refpages'] = row['refpages']
            G_backlink.nodes[node_mapping[row['url']]]['pre_backlinks'] = 0
            G_backlink.nodes[node_mapping[row['url']]]['post_backlinks'] = 0
            G_backlink.nodes[node_mapping[row['url']]]['pre_refpages'] = 0
            G_backlink.nodes[node_mapping[row['url']]]['post_refpages'] = 0
            
            news_url_list.append(node_mapping[row['url']])
        except KeyError:
            continue

    attributes_df.drop(['backlinks', 'refpages'], axis=1, inplace=True)

    # only use edges in reduced_source_names
    # reduced_edge_df = pd.read_csv(reduced_link_network_path)
    # reduced_edge_df.reset_index(inplace=True)
    # reduced_edge_df['domain_from'] = reduced_edge_df['domain_from'].str.lower()
    # reduced_edge_df['domain_to'] = reduced_edge_df['domain_to'].str.lower()
    # reduced_edge_df['domain_from_idx'] = reduced_edge_df.domain_from.map(node_mapping)
    # reduced_edge_df['domain_to_idx'] = reduced_edge_df.domain_to.map(node_mapping)

    multiplicity_links = pd.read_csv(multiplicity_file)[['source_domain', 'label_domain', 'inv_count_scaled']]

    # G_backlink_reduced = nx.from_pandas_edgelist(edge_df, source='domain_from_idx', target='domain_to_idx', create_using=nx.DiGraph())
    # backlinks_t -= backlinks_e * (1 - relevance_score)
    def downrank_low_relevancy(G, urls):

        relevancy_weighted_traffic_data = []
        errors = 0
        for news_url in urls:
            try:
                preds = G.predecessors(news_url)
                for neighbor in list(preds):

                    neighbor_url = inv_node_mapping[neighbor]
                    edge_data = G_backlink.get_edge_data(neighbor, news_url)

                    relevancy_score = 1 - (neighbor in df_schemes)
                    if multiplicity_weighting:
                        if ((multiplicity_links['source_domain'] == neighbor_url) & (multiplicity_links['label_domain'] == inv_node_mapping[news_url])).sum() > 0:
                            relevancy_score = relevancy_score * multiplicity_links[(multiplicity_links['source_domain'] == neighbor_url) & (multiplicity_links['label_domain'] == inv_node_mapping[news_url])]['inv_count_scaled'].values[0]

                    # update post scores
                    G_backlink.nodes[news_url]['post_backlinks'] = G_backlink.nodes[news_url]['post_backlinks'] + edge_data['links'] * relevancy_score
                    G_backlink.nodes[news_url]['post_refpages'] = G_backlink.nodes[news_url]['post_refpages'] + edge_data['unique_pages'] * relevancy_score
                    # update pre scores (for comparison)
                    G_backlink.nodes[news_url]['pre_backlinks'] = G_backlink.nodes[news_url]['pre_backlinks'] + edge_data['links']
                    G_backlink.nodes[news_url]['pre_refpages'] = G_backlink.nodes[news_url]['pre_refpages'] + edge_data['unique_pages']

                relevancy_weighted_traffic_data.append({
                    'url': news_url,
                    'pre_backlinks': G_backlink.nodes[news_url]['pre_backlinks'],
                    'pre_refpages': G_backlink.nodes[news_url]['pre_refpages'],
                    'post_backlinks': G_backlink.nodes[news_url]['post_backlinks'],
                    'post_refpages': G_backlink.nodes[news_url]['post_refpages'],
                })

            except Exception as e:
                # errors here are due to nodes not being in the graph
                # this especially happens when news_urls are a sample of the full graph
                # i.e. only using news_urls with bias scores for bias_transformed
                errors += 1
                continue
        return pd.DataFrame(relevancy_weighted_traffic_data)

    negated_df = downrank_low_relevancy(G_backlink, news_url_list)

    # map negated_df to attributes_df
    negated_df['url'] = negated_df['url'].map(inv_node_mapping)
    negated_attributes_df = negated_df.merge(attributes_df, on='url', how='left')

    # if backlinks contains np array, take first val
    def get_first_val(x):

        if isinstance(x, np.ndarray):
            return x[0]
        else:
            return x

    negated_attributes_df['post_backlinks'] = negated_attributes_df['post_backlinks'].apply(lambda x: get_first_val(x))
    negated_attributes_df['post_refpages'] = negated_attributes_df['post_refpages'].apply(lambda x: get_first_val(x))

    return (negated_attributes_df, G_backlink, inv_node_mapping)
