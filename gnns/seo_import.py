import pandas as pd
import networkx as nx
import numpy as np
from torch_geometric.loader import NeighborLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import torch
import random
random.seed(123)
np.random.seed(123)
NO_LABEL = -1
output_prefix = 'gnn_'

label_schemes = {
    'traffic': './data/traffic.csv', # reliability labels
    'rank': './data/traffic.csv', # absolute bias: extreme vs centrist
} 

weight_schemes = ['log_links']

link_networks = {
    'backlinks':('./data/filtered_combined_attrs.csv', './data/filtered_backlinks.csv'),
}


def drop_corrupted_rows(links, labelled):
    """
    Drop rows with corrupted data- there were like 3 cryllic websites that the api couldn't ingest.
    They return nonsense- we'll delete them here.
    """
    a = links.domain_from.unique().tolist()
    b = links.domain_to.unique().tolist()
    uh = list(set(a +b))
    todrop_inlabeled = labelled[~labelled.url.isin(uh)].url
    labelled = labelled[~labelled.url.isin(todrop_inlabeled)]
 
    links = links[links.domain_from.isin(labelled.url.tolist())]
    links = links[links.domain_to.isin(labelled.url.tolist())]

    return links, labelled 

def import_seo_links(data_input_path, links_input_path, label_column, labels_input_path = ""):
    """
    import data
    """
    if label_column == 'cc-orig-pr':
        attribute_df = pd.read_csv(data_input_path).drop_duplicates(subset='url')
        prdf = pd.read_csv('../data/cc_link_scheme_removal_results.csv').dropna()
        prdf = prdf[prdf.url.isin(attribute_df.url.tolist())]
        attribute_df = attribute_df[attribute_df.url.isin(prdf.url.tolist())]
        prdf = prdf.sort_values('url')
        attribute_df = attribute_df.sort_values('url')
        #np.array_equal(prdf.url.values, attribute_df.url.values)
        labelled = pd.merge(attribute_df, prdf[['url', label_column]], how='left', on='url')

    else:
        attributes = pd.read_csv(data_input_path).drop_duplicates(subset='url')
        if labels_input_path != "":
            labels = pd.read_csv(labels_input_path)[['url', label_column]]
            if label_column in attributes.columns:
                attributes.drop(columns=label_column, inplace=True)
            labelled = pd.merge(attributes, labels, how='left', on='url').fillna(NO_LABEL)
        else:
            assert(label_column in attributes.columns)
            labelled = attributes

    links = pd.read_csv(links_input_path).dropna()
    link_attrs = pd.read_csv('../data/3kupdate/filtered_backlink_attrs.csv')
    links = links[links.domain_from.isin(link_attrs.url.unique())]
    links = links[links.domain_to.isin(link_attrs.url.unique())]

    label_abr = labelled[['url', 'label', label_column]]
    link_attrs = link_attrs.drop_duplicates(subset = 'url')
    label_abr = link_attrs.merge(label_abr, how = 'left', left_on = 'url', right_on = 'url')
    label_abr.label = label_abr.label.fillna(-1)


    links, labelled = drop_corrupted_rows(links, label_abr)

    urls = labelled.url.unique().tolist()
    url_mapper = {url: i for i, url in enumerate(urls)}
    labelled['id'] = labelled['url'].map(url_mapper)
    links['source'] = links['domain_from'].map(url_mapper)
    links['target'] = links['domain_to'].map(url_mapper)

    links.drop(columns = ['domain_from', 'domain_to'], inplace=True)
    
    #lets try dropping 3 and 4's: 
    #labelled.dropna(inplace=True)

    # labelled['label_column'] = labelled.label.astype(int).replace(label_scheme)
    # if 'traffic' in label_column:
    #     labelled[label_column] = np.log1p(labelled[label_column])

    return labelled, links, url_mapper

def train_val_test_split(labelled, links, weight_scheme, label_column, batch_size=64):
    """
    Generate train, val, and test masks and y tensor
    """
    targets = labelled[labelled[label_column] > NO_LABEL]
    ids = targets[['id', label_column]]
    X_train, X_v = train_test_split(ids, test_size=0.20, random_state=42, shuffle=True, stratify = None)#ids.label)
    X_val, X_test = train_test_split(X_v, test_size=0.5, random_state=42, shuffle=True, stratify = None)#X_v.label)

    train_mask = torch.tensor(X_train.id.values).long()
    val_mask = torch.tensor(X_val.id.values).long()
    test_mask = torch.tensor(X_test.id.values).long()

    # lets log y
    y = labelled[label_column].to_numpy()
    if label_column == 'cc-orig-pr':
        y = np.log(y)
        y = torch.FloatTensor(y)
    else:
        y = np.log(y + 1)
        y = torch.FloatTensor(y)

    # normalize feature matrix
    features = labelled.copy().drop(columns = ['url', 'id', 'label', label_column]).values
    #min_max_scaler = preprocessing.MinMaxScaler()
    #x_scaled = min_max_scaler.fit_transform(np.log1p(features + 1))
    x_scaled = np.log1p(features)
    x_scaled = torch.FloatTensor(x_scaled)
    
    # NOTE: comment for topN experiments
    # links[['so_ratio']] = links[['so_ratio']].replace(np.inf, 1)
    link_attrs = links.loc[:, ['links','unique_pages']]#,'tb_ratio','so_ratio', 'e_tb_ratio','e_so_ratio','tp_ratio', 'sp_ratio']]
    # NOTE: uncomment for topN experiments
    # link_attrs = links.loc[:, ['links']]
    
    edge_list = torch.tensor([links['source'].values, links['target'].values]).long()
    if 'log' in weight_scheme or 'none' in weight_scheme:
        edge_weight = torch.FloatTensor(np.log1p(link_attrs['links'].values))
    else:
        edge_weight = torch.FloatTensor(link_attrs[weight_scheme].values)

    # edge_weight = torch.FloatTensor(np.log([x+1 for x in links['tb_ratio'].values])) # tb_ratio, so_ratio, tp_ratio, unique_pages

    data = Data(x=x_scaled, edge_index = edge_list, edge_weight=edge_weight, y=y, train_mask=train_mask, val_mask = val_mask, test_mask = test_mask)

    # sample neighbors - we use this for minibatching
    train_loader = NeighborLoader(data, input_nodes=(data.train_mask),
                                num_neighbors=[25, 15], batch_size=batch_size, shuffle=True,
                                num_workers=0)
    valid_loader = NeighborLoader(data, input_nodes = (data.val_mask),
                            num_neighbors=[25, 15], batch_size=batch_size, shuffle=False,
                            num_workers=0)
    test_loader = NeighborLoader(data, input_nodes = (data.test_mask),
                            num_neighbors=[25, 15], batch_size=batch_size, shuffle=False,
                            num_workers=0)

    return train_loader, valid_loader, test_loader


# def data_only(labelled, links, label_column):
#     """
#     Generate train, val, and test masks and y tensor
#     """
#     targets = labelled[labelled['label'] > -1]
#     ids = targets[['id', 'label']]
#     X_train, X_v = train_test_split(ids, test_size=0.20, random_state=42, shuffle=True, stratify = ids.label)
#     X_val, X_test = train_test_split(X_v, test_size=0.5, random_state=42, shuffle=True, stratify = X_v.label)

#     train_mask = torch.tensor(X_train.id.values).long()
#     val_mask = torch.tensor(X_val.id.values).long()
#     test_mask = torch.tensor(X_test.id.values).long()

#     y = torch.tensor(labelled.label.to_numpy()).long()

#     # normalize feature matrix
#     features = labelled.copy().drop(columns = ['url', 'id', label_column]).values
#     min_max_scaler = preprocessing.MinMaxScaler()
#     x_scaled = min_max_scaler.fit_transform(features)
#     x_scaled = torch.FloatTensor(x_scaled)

#     edge_list = torch.tensor([links['source'].values, links['target'].values]).long()

#     edge_weight = torch.FloatTensor(np.log([links['links'].values]))

#     data = Data(x=x_scaled, edge_index = edge_list, edge_weight=edge_weight.reshape(-1), y=y, train_mask=train_mask, val_mask = val_mask, test_mask = test_mask)
#     return data

def create_inference_loader(labelled, links, weight_scheme, label_column, batch_size=64):
    """
    Generate a DataLoader for inference for every URL.
    """
    # Normalize feature matrix
    targets = labelled[labelled[label_column] > NO_LABEL]
    label_mask = torch.tensor(targets.id.values).long()
    features = labelled.copy().drop(columns=['url', 'id', 'label', label_column]).values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(np.log1p(features))
    x_scaled = torch.FloatTensor(x_scaled)
    
    # Prepare edge attributes and edge list
    edge_list = torch.tensor([links['source'].values, links['target'].values]).long()
    
    if 'log' in weight_scheme or 'none' in weight_scheme:
        edge_weight = torch.FloatTensor(np.log1p(links['links'].values))
    else:
        edge_weight = torch.FloatTensor(links[weight_scheme].values)

    # Create Data object
    data = Data(x=x_scaled, edge_index=edge_list, edge_weight=edge_weight, label_mask=label_mask)

    # Create a DataLoader for all nodes
    inference_loader = NeighborLoader(data, 
                                      input_nodes=(data.label_mask),
                                      num_neighbors=[25, 15], 
                                      batch_size=batch_size, 
                                      shuffle=False, 
                                      num_workers=0)

    return inference_loader
