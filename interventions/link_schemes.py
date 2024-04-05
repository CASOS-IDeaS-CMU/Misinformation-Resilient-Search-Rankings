#!/usr/bin/env python
# coding: utf-8

import networkx as nx
import pandas as pd
from functools import reduce
import numpy as np
import random

def get_link_schemes(G_backlink, unreliable_targets, breath_criteria=3, depth_criteria=100000):
    candidates = set()
    for target in unreliable_targets:
        # current_node = node_mapping[target]
        preds = list(G_backlink.predecessors(target))
        for pred in preds:
            candidates.add(pred)

    link_scheme_stats = {}
    for candidate in candidates:
        unreliable_link_count = 0
        unreliable_link_sum = 0
        for succ in G_backlink.successors(candidate):
            if succ in unreliable_targets:
                unreliable_link_count += 1
                unreliable_link_sum += G_backlink.get_edge_data(candidate, succ)['links']
        link_scheme_stats[candidate] = {}
        link_scheme_stats[candidate]['count'] = unreliable_link_count
        link_scheme_stats[candidate]['sum'] = unreliable_link_sum

    def is_link_scheme(url):
        stats = link_scheme_stats[url]
        return stats['count'] >= breath_criteria and stats['sum'] >= depth_criteria

    schemes = []
    for url in link_scheme_stats.keys():
        if is_link_scheme(url):
            schemes.append(url)

    return schemes

if __name__ == '__main__':
    attributes_path = '../data/filtered_attrs.csv'
    link_network_path = '../data/filtered_backlinks.csv'
    sample = 0.20
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
    attributes_df = pd.read_csv(attributes_path)

    attributes_df['url'] = attributes_df['url'].str.lower()
    url_df = attributes_df.copy()
    url_df = url_df[['url','label']]

    unreliable_labels = url_df[url_df['label']<=4]['url'].to_list()
    random.seed(11)
    unreliable_targets = list(set([node_mapping[t] for t in list(edge_df.domain_to.unique()) if t in unreliable_labels]))
    if sample:
        unreliable_targets = random.sample(unreliable_targets, int(len(unreliable_targets)*sample))

    link_schemes = get_link_schemes(G_backlink, unreliable_targets, 2, 25000)
    for link_scheme in link_schemes:
        print(link_scheme)