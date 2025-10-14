import torch
from torch import Tensor
from torch_geometric.data import download_url, extract_zip
import pandas as pd
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
import tqdm

from sklearn.metrics import roc_auc_score,auc,accuracy_score,roc_curve, average_precision_score
from sklearn.metrics import roc_auc_score,auc,accuracy_score,roc_curve,f1_score

import numpy as np

from sklearn.metrics import accuracy_score, roc_curve
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, util

#####################
import  networkx as nx

from cnlp.similarity_methods.local_similarity import common_neighbors, adamic_adar, jaccard, node_clustering
from cnlp.similarity_methods.global_similarity import katz_index, link_prediction_rwr, sim_rank
from cnlp.similarity_methods.quasi_local_similarity import path_of_length_three, local_path_index
from cnlp.probabilistic_methods import stochastic_block_model
from cnlp.other_methods.information_theory import path_entropy


from cnlp.utils import nodes_to_indexes
from cnlp.utils import get_top_predicted_link

import random
import logging
import time
import itertools
from sklearn import preprocessing
import gc

import os
import psutil
#___________________________________________
#-----------------------------------------------------------------#
import csv
import pandas as pd # pandas to create small dataframes 
import datetime # Convert to unix time
import time # Convert to unix time
import random
# if numpy is not installed already : pip3 install numpy
import numpy as np # do arithmetic operations on arrays
# matplotlib: used to plot graphs
import matplotlib.pyplot as plt
import seaborn as sns # Visualizations
from matplotlib import rcParams # Size of plots  
import math
import pickle
import os
# to install xgboost: pip3 install xgboost
import xgboost as xgb # ML ensemble model
#-----------------------------------------------------------------#
from pandas import HDFStore, DataFrame
from pandas import read_hdf
from scipy.sparse.linalg import svds, eigs
import gc
from tqdm import tqdm
#-----------------------------------------------------------------#
import networkx as nx
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
#___________________________________________

from xgboost.sklearn import XGBClassifier


logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
logging.warning(msg={'start time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
logging.warning(msg={'file_name':os.path.basename(__file__)})

if __name__ == '__main__':

    comm=pd.read_csv('../datasets/single_graph_merged_data.csv',engine="c")
    com=comm.head(200)
    logging.warning(msg={'com_count':com.count()})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish read data time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    com=com.drop_duplicates(subset=['pmcid','ID'],keep='first')
    logging.warning(msg={'com_count':com.count()})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish drop duplicates time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    # com=com.sample(90000)
    # logging.warning(msg={'com_count':com.count()})

    paper=com[['pmcid','title','abstract']].drop_duplicates(subset=['pmcid'],keep='first')
    logging.warning(msg={'paper_count':paper.count()})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish paper time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    software=com[['ID','aggtext']].drop_duplicates(subset=['ID'],keep='first')

    logging.warning(msg={'software_count':software.count()})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish software time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    paper['paper_id']=[i for i in np.arange(0,paper.shape[0])]

    logging.warning(msg={'paper_count':paper.count()})
    new_paper=paper
    logging.warning(msg={'copied_new_paper':new_paper.count()})

    paper=paper.drop(['title'], axis= 1)
    logging.warning(msg={'paper_drop_title_count':paper.count()})
    paper=paper.drop(['abstract'], axis= 1)
    logging.warning(msg={'paper_drop_abstract_count':paper.count()})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish paper_id time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    software['software_id']=[i for i in np.arange(paper.shape[0],paper.shape[0]+software.shape[0])]
    logging.warning(msg={'software_count':software.count()})
    new_software=software
    logging.warning(msg={'copied_new_software':new_software.count()})

    software=software.drop(['aggtext'], axis= 1)
    logging.warning(msg={'software_count':software.count()})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish software_id time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    # na_data_df=pd.merge(na_data_df,paper,on='pmcid')
    # na_data_df=pd.merge(na_data_df,software,on='ID')

    com=com[['pmcid','ID']]
    logging.warning(msg={'com_count':com.count()})
    com=pd.merge(com,paper,on='pmcid')
    logging.warning(msg={'com_count':com.count()})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish merge paper time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    com=pd.merge(com,software,on='ID')
    logging.warning(msg={'com_count':com.count()})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish merge software time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    #______________________________________________________#
    g=nx.DiGraph()
    
    
    edges=[i for i in com[['paper_id','software_id']].values.tolist()]
    g.add_edges_from(edges)
    logging.warning(msg={'g':g})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish graph time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    r=com[['paper_id','software_id']].values.tolist()
    edges = dict()
    for edge in r:
        edges[(edge[0], edge[1])] = 1
    # logging.warning(msg={'edges':edges})
    logging.warning(msg={'edges_len':len(edges)})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish edges time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    missing_edges = set([])
    while (len(missing_edges) < com.shape[0]):
        ui = random.randint(1, paper.shape[0])
        uj = random.randint(paper.shape[0],paper.shape[0]+ software.shape[0])
        temp = edges.get((ui, uj), -1)

        if temp == -1 and ui != uj:
            try:
                if nx.shortest_path_length(g, source = ui, target = uj) >= 2:
                    missing_edges.add((ui, uj))
                else:
                    continue
            except:
                missing_edges.add((ui, uj))
        else:
            continue
    # logging.warning(msg={'missing_edges':missing_edges})
    logging.warning(msg={'missing_edges':len(missing_edges)})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish miss_edges time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    logging.warning(msg="df_pos:____________________________________________________")
    df_pos = com[['paper_id','software_id']]# train data as it is
    logging.warning(msg={'df_pos':df_pos})
    logging.warning(msg={'df_pos_count':df_pos.count()})

    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish df_pos time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    logging.warning(msg="df_pos:____________________________________________________")
        
    logging.warning(msg="df_neg:____________________________________________________")
    df_neg = pd.DataFrame(list(missing_edges), columns = ['paper_id', 'software_id'])
    logging.warning(msg={'df_neg':df_neg})
    logging.warning(msg={'df_neg_count':df_neg.count()})

    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish df_neg time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    logging.warning(msg="df_neg:____________________________________________________")
    
    from sklearn.model_selection import train_test_split
    logging.warning(msg={'split':'split'})
    # Train-Test split (80 - 20)
    # split positive links and negative links seperatly because we need positive training data only for feature engineering
    X_train_pos, X_test_pos, y_train_pos, y_test_pos  = train_test_split(df_pos, np.ones(len(df_pos)), test_size = 0.2, random_state = 101)
    X_train_neg, X_test_neg, y_train_neg, y_test_neg  = train_test_split(df_neg, np.zeros(len(df_neg)), test_size = 0.2, random_state = 101)
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish split time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})



    train_graph=nx.DiGraph()
    edges=[i for i in X_train_pos.values.tolist()]
    train_graph.add_edges_from(edges)
    logging.warning(msg={'train_graph':train_graph})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish train_graph time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    test_graph=nx.DiGraph()
    edges=[i for i in X_test_pos.values.tolist()]
    test_graph.add_edges_from(edges)
    logging.warning(msg={'test_graph':test_graph})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish test_graph time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    # finding the unique nodes in the both train and test graphs
    train_nodes_pos = set(train_graph.nodes())
    logging.warning(msg={'train_nodes_pos_len':len(train_nodes_pos)})

    test_nodes_pos = set(test_graph.nodes())
    logging.warning(msg={'test_nodes_pos_len':len(test_nodes_pos)})

    trY_teY = len(train_nodes_pos.intersection(test_nodes_pos))
    trY_teN = len(train_nodes_pos - test_nodes_pos)
    teY_trN = len(test_nodes_pos - train_nodes_pos)
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish trY_te_Y time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    X_train = X_train_pos.append(X_train_neg, ignore_index = True) 
    logging.warning(msg={'X_train':X_train})
    logging.warning(msg={'X_train':X_train.count()})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish X_train time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    y_train = np.concatenate((y_train_pos, y_train_neg))
    logging.warning(msg={'y_train':y_train})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish y_train time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    X_test = X_test_pos.append(X_test_neg, ignore_index = True)
    logging.warning(msg={'X_test':X_test})
    logging.warning(msg={'X_test':X_test.count()})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish X_test time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})    

    y_test = np.concatenate((y_test_pos, y_test_neg)) 
    logging.warning(msg={'y_test':y_test})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish y_test time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    # pr = nx.pagerank(train_graph, alpha = 0.85)
    # logging.warning(msg={'page_rank':'pr'})
    # logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    # logging.warning(msg={'finish pr time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    # # #______________________________________#
    # wcc = list(nx.weakly_connected_components(train_graph))
    # logging.warning(msg={'wcc':'wcc'})
    # logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    # logging.warning(msg={'finish wcc time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    # # #______________________________________#
    # katz = nx.katz.katz_centrality(train_graph, alpha = 0.005, beta = 1)
    # logging.warning(msg={'katz':'katz'})
    # logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    # logging.warning(msg={'finish katz time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    # # #______________________________________#
    # hits = nx.hits(train_graph, max_iter = 100, tol = 1e-08, nstart = None, normalized = True)
    # logging.warning(msg={'hits':'hits'})
    # logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    # logging.warning(msg={'finish hits time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    df_final_train=X_train
    logging.warning(msg={'df_final_train':df_final_train.count()})
    logging.warning(msg={'df_final_train':df_final_train.head()})

    df_final_train['indicator_link'] = y_train
    logging.warning(msg={'df_final_train':df_final_train.count()})
    logging.warning(msg={'df_final_train':df_final_train.head()})

    df_final_test=X_test
    # df_final_test=X_test
    logging.warning(msg={'df_final_test':df_final_test.count()})
    logging.warning(msg={'df_final_test':df_final_test.head()})

    df_final_test['indicator_link'] = y_test
    logging.warning(msg={'df_final_test':df_final_test.count()})
    logging.warning(msg={'df_final_test':df_final_test.head()})

    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish df_final data time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    #_______________graph feature_______________________#
    logging.warning(msg={'df_final_test':"#_______________graph feature_______________________#"})

    ################################################
    df_final_train_G=nx.DiGraph()
    df_final_test_G=nx.DiGraph()
    logging.warning(msg='start building graph')
    logging.warning(msg={'start building graph':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    df_final_train_edges=[i for i in df_final_train[['paper_id','software_id']].values.tolist()]
    df_final_test_edges=[i for i in df_final_test[['paper_id','software_id']].values.tolist()]

    df_final_train_G.add_edges_from(df_final_train_edges)
    df_final_test_G.add_edges_from(df_final_test_edges)

    logging.warning(msg="Number of nodes : " + str(df_final_train_G.number_of_nodes()))
    logging.warning(msg="Number of edges : " + str(df_final_train_G.number_of_edges()))
    logging.warning(msg="Number of nodes : " + str(df_final_test_G.number_of_nodes()))
    logging.warning(msg="Number of edges : " + str(df_final_test_G.number_of_edges()))

    build_graph_time=time.time()
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg='start computing graph feature')

    df_final_train_G_out_degree_centrality = nx.out_degree_centrality(df_final_train_G)
    df_final_train_G_in_degree_centrality = nx.in_degree_centrality(df_final_train_G)

    df_final_test_G_out_degree_centrality = nx.out_degree_centrality(df_final_test_G)
    df_final_test_G_in_degree_centrality = nx.in_degree_centrality(df_final_test_G)

    df_final_train['source_out_centrality'] = df_final_train.apply(lambda row: df_final_train_G_out_degree_centrality[row.paper_id],axis=1)
    df_final_train['target_in_centrality'] = df_final_train.apply(lambda row: df_final_train_G_in_degree_centrality[row.software_id],axis=1)
    df_final_test['source_out_centrality'] = df_final_test.apply(lambda row: df_final_test_G_out_degree_centrality[row.paper_id],axis=1)
    df_final_test['target_in_centrality'] = df_final_test.apply(lambda row: df_final_test_G_in_degree_centrality[row.software_id],axis=1)

    del df_final_train_G_out_degree_centrality
    del df_final_train_G_in_degree_centrality
    del df_final_test_G_out_degree_centrality
    del df_final_test_G_in_degree_centrality

    logging.warning(msg='finish degree')
    degree_time=time.time()
    logging.warning(msg=(degree_time-build_graph_time))
    logging.warning(msg={'df_final_train':df_final_train.head()})
    logging.warning(msg={'df_final_train':df_final_train.count()})
    logging.warning(msg={'df_final_train':df_final_test.head()})
    logging.warning(msg={'df_final_train':df_final_test.count()})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish degree time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    df_final_train_G_page_rank = nx.pagerank_scipy(df_final_train_G)
    df_final_test_G_page_rank = nx.pagerank_scipy(df_final_test_G)

    df_final_train['target_pagerank'] = df_final_train.apply(lambda row: df_final_train_G_page_rank[row.software_id],axis=1)
    df_final_train['source_pagerank'] = df_final_train.apply(lambda row: df_final_train_G_page_rank[row.paper_id],axis=1)
    df_final_test['target_pagerank'] = df_final_test.apply(lambda row: df_final_test_G_page_rank[row.software_id],axis=1)
    df_final_test['source_pagerank'] = df_final_test.apply(lambda row: df_final_test_G_page_rank[row.paper_id],axis=1)

    del df_final_train_G_page_rank
    del df_final_test_G_page_rank

    page_rank_time=time.time()
    logging.warning(msg=(page_rank_time-degree_time))
    logging.warning(msg={'df_final_train':df_final_train.head()})
    logging.warning(msg={'df_final_train':df_final_train.count()})
    logging.warning(msg={'df_final_train':df_final_test.head()})
    logging.warning(msg={'df_final_train':df_final_test.count()})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg='finish pagerank')
    logging.warning(msg={'finish pagerank time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    '''
    katz_centrality=nx.katz_centrality(G)

    data_df['target_katz_centrality'] = data_df.apply(lambda row: katz_centrality[row.ID],axis=1)
    data_df['source_katz_centrality'] = data_df.apply(lambda row: katz_centrality[row.pmcid],axis=1)
    '''

    df_final_train_G_katz_centrality=nx.katz_centrality(df_final_train_G)
    df_final_test_G_katz_centrality=nx.katz_centrality(df_final_test_G)

    df_final_train['target_katz_centrality'] = df_final_train.apply(lambda row: df_final_train_G_katz_centrality[row.software_id],axis=1)
    df_final_train['source_katz_centrality'] = df_final_train.apply(lambda row: df_final_train_G_katz_centrality[row.paper_id],axis=1)
    df_final_test['target_katz_centrality'] = df_final_test.apply(lambda row: df_final_test_G_katz_centrality[row.software_id],axis=1)
    df_final_test['source_katz_centrality'] = df_final_test.apply(lambda row: df_final_test_G_katz_centrality[row.paper_id],axis=1)

    del df_final_train_G_katz_centrality
    del df_final_test_G_katz_centrality

    logging.warning(msg='finish katz_centrality')
    katz_time=time.time()
    logging.warning(msg=(katz_time-page_rank_time))
    logging.warning(msg={'df_final_train':df_final_train.head()})
    logging.warning(msg={'df_final_train':df_final_train.count()})
    logging.warning(msg={'df_final_train':df_final_test.head()})
    logging.warning(msg={'df_final_train':df_final_test.count()})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))

    df_final_train['preferencial_attachment'] = df_final_train.apply(lambda row: row.source_out_centrality * row.target_in_centrality,axis=1)
    df_final_test['preferencial_attachment'] = df_final_test.apply(lambda row: row.source_out_centrality * row.target_in_centrality,axis=1)

    logging.warning(msg='finish preferencial_attachment')
    pa_time=time.time()
    logging.warning(msg={'df_final_train':df_final_train.head()})
    logging.warning(msg={'df_final_train':df_final_train.count()})
    logging.warning(msg={'df_final_train':df_final_test.head()})
    logging.warning(msg={'df_final_train':df_final_test.count()})
    logging.warning(msg=(pa_time-katz_time))
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish pa time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})


    df_final_train_hub_score, df_final_train_authority_score = nx.hits(df_final_train_G)
    df_final_test_hub_score, df_final_test_authority_score = nx.hits(df_final_test_G)

    df_final_train['source_hub_score'] = df_final_train.apply(lambda row: df_final_train_hub_score[row.paper_id],axis=1)
    df_final_train['target_authority_score'] = df_final_train.apply(lambda row: df_final_train_authority_score[row.software_id],axis=1)

    df_final_test['source_hub_score'] = df_final_test.apply(lambda row: df_final_test_hub_score[row.paper_id],axis=1)
    df_final_test['target_authority_score'] = df_final_test.apply(lambda row: df_final_test_authority_score[row.software_id],axis=1)

    logging.warning(msg='finish  HITS algorithm')
    hits_time=time.time()
    logging.warning(msg={'df_final_train':df_final_train.head()})
    logging.warning(msg={'df_final_train':df_final_train.count()})
    logging.warning(msg={'df_final_train':df_final_test.head()})
    logging.warning(msg={'df_final_train':df_final_test.count()})
    logging.warning(msg=(hits_time-pa_time))
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish HITS time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    del df_final_train_hub_score
    del df_final_train_authority_score
    del df_final_test_hub_score
    del df_final_test_authority_score

    e1 = 1e-20 # for HITS
    e2 = 1e-7 # for page rank
    df_final_train['source_out_centrality'] = df_final_train['source_out_centrality'].apply(lambda x: -1/math.log10(e2 + x))
    df_final_train['target_in_centrality'] = df_final_train['target_in_centrality'].apply(lambda x: -1/math.log10(e2 + x))
    df_final_test['source_out_centrality'] = df_final_test['source_out_centrality'].apply(lambda x: -1/math.log10(e2 + x))
    df_final_test['target_in_centrality'] = df_final_test['target_in_centrality'].apply(lambda x: -1/math.log10(e2 + x))

    df_final_train['target_pagerank'] = df_final_train['target_pagerank'].apply(lambda x: -1/math.log10(e2 + x))
    df_final_train['source_pagerank'] = df_final_train['source_pagerank'].apply(lambda x: -1/math.log10(e2 + x))
    df_final_test['target_pagerank'] = df_final_test['target_pagerank'].apply(lambda x: -1/math.log10(e2 + x))
    df_final_test['source_pagerank'] = df_final_test['source_pagerank'].apply(lambda x: -1/math.log10(e2 + x))

    df_final_train['target_katz_centrality'] = df_final_train['target_katz_centrality'].apply(lambda x: -1/math.log10(e2 + x))
    df_final_train['source_katz_centrality'] = df_final_train['source_katz_centrality'].apply(lambda x: -1/math.log10(e2 + x))
    df_final_test['target_katz_centrality'] = df_final_test['target_katz_centrality'].apply(lambda x: -1/math.log10(e2 + x))
    df_final_test['source_katz_centrality'] = df_final_test['source_katz_centrality'].apply(lambda x: -1/math.log10(e2 + x))

    df_final_train['preferencial_attachment'] = df_final_train['preferencial_attachment'].apply(lambda x: -1/math.log10(e1 + x))
    df_final_test['preferencial_attachment'] = df_final_test['preferencial_attachment'].apply(lambda x: -1/math.log10(e1 + x))

    df_final_train['source_hub_score'] = df_final_train['source_hub_score'].apply(lambda x: -1/math.log10(max(e1 + x,e1)))
    df_final_train['target_authority_score'] = df_final_train['target_authority_score'].apply(lambda x: -1/math.log10(max(e1 + x,e1)))
    df_final_test['source_hub_score'] = df_final_test['source_hub_score'].apply(lambda x: -1/math.log10(max(e1 + x,e1)))
    df_final_test['target_authority_score'] = df_final_test['target_authority_score'].apply(lambda x: -1/math.log10(max(e1 + x,e1)))

    # #================================================================================

    # df_pos=pd.merge(df_pos,paper,on='paper_id')
    # logging.warning(msg={'df_pos':df_pos})
    # logging.warning(msg={'df_pos_count':df_pos.count()})

    # df_pos=pd.merge(df_pos,software,on='software_id')

    # logging.warning(msg={'df_pos':df_pos})
    # logging.warning(msg={'df_pos_count':df_pos.count()})

    # df_neg = pd.merge(df_neg,paper,on='paper_id')
    # logging.warning(msg={'df_neg':df_neg})
    # logging.warning(msg={'df_neg_count':df_neg.count()})

    # df_neg = pd.merge(df_neg,software,on='software_id')
    # logging.warning(msg={'df_neg':df_neg})
    # logging.warning(msg={'df_neg_count':df_neg.count()})
    
    #___________________ebedding_________________________
    # paper=paper.drop(['abstract'], axis= 1)
    new_paper=new_paper.drop(['pmcid'],axis=1)
    new_software=new_software.drop(['ID'],axis=1)

    # com=pd.merge(com,paper,on='pmcid')
    logging.warning(msg={'df_final_train':df_final_train.count()})

    df_final_train=pd.merge(df_final_train,new_paper,on='paper_id')
    logging.warning(msg={'df_final_train':df_final_train.count()})


    df_final_train=pd.merge(df_final_train,new_software,on='software_id')
    logging.warning(msg={'df_final_train':df_final_train.count()})

    logging.warning(msg={'df_final_test':df_final_test.count()})

    df_final_test=pd.merge(df_final_test,new_paper,on='paper_id')
    logging.warning(msg={'df_final_test':df_final_test.count()})

    df_final_test=pd.merge(df_final_test,new_software,on='software_id')
    logging.warning(msg={'df_final_test':df_final_test.count()})

    #———————————start_multi_process_pool—————————————————
    #———————————start_multi_process_pool—————————————————

    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.max_seq_length = 200

    logging.warning(msg='start pool')
    logging.warning(msg={'start ebedding time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    pool = model.start_multi_process_pool()
    logging.warning(msg='using start_multi_process_pool')
    logging.warning(msg=pool)

    logging.warning(msg='start embedding')
    logging.warning(msg={'start ebedding time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    aggtext_embeddings = model.encode_multi_process(df_final_train.aggtext.to_list(),pool,batch_size=128)

    logging.warning(msg={'aggtext ebedding consumption time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))

    abstract_embeddings = model.encode_multi_process(df_final_train.abstract.to_list(),pool,batch_size=128)

    logging.warning(msg={'abstract ebedding consumption time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))

    title_embeddings = model.encode_multi_process(df_final_train.title.to_list(),pool,batch_size=128)

    logging.warning(msg={'title ebedding consumption time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))

    model.stop_multi_process_pool(pool)
    # title_embeddings
    # cos_sim_list=[util.cos_sim(model.encode(row.title, convert_to_tensor=True), model.encode(row.abstract, convert_to_tensor=True)).cpu().detach().numpy()[0][0] for row in ()]

    df_final_train['abs_agg_cos_sim']=util.pairwise_cos_sim(abstract_embeddings,aggtext_embeddings)

    logging.warning(msg={'abs_agg_cos_sim consumption time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))

    df_final_train['title_agg_cos_sim']=util.pairwise_cos_sim(title_embeddings,aggtext_embeddings)

    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'abs_agg_cos_sim consumption time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    logging.warning(msg={'cos_sim df_final_train':df_final_train.count()})
    logging.warning(msg={'cos_sim df_final_train':df_final_train[['abs_agg_cos_sim','title_agg_cos_sim']].head(2)})


    del model
    del aggtext_embeddings
    del abstract_embeddings
    del title_embeddings
    logging.warning(msg=u'after delete memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))


    #———————————- end_multi_process_pool -—————————————————
    #———————————- end_multi_process_pool -—————————————————

    #___________________ebedding_________________________
    #———————————start_multi_process_pool—————————————————
    #———————————start_multi_process_pool—————————————————
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.max_seq_length = 200

    logging.warning(msg='start pool')
    logging.warning(msg={'start ebedding time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    pool = model.start_multi_process_pool()
    logging.warning(msg='using start_multi_process_pool')
    logging.warning(msg=pool)

    logging.warning(msg='start embedding')
    logging.warning(msg={'start ebedding time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    aggtext_embeddings = model.encode_multi_process(df_final_test.aggtext.to_list(),pool,batch_size=128)

    logging.warning(msg={'aggtext ebedding consumption time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))

    abstract_embeddings = model.encode_multi_process(df_final_test.abstract.to_list(),pool,batch_size=128)

    logging.warning(msg={'abstract ebedding consumption time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))

    title_embeddings = model.encode_multi_process(df_final_test.title.to_list(),pool,batch_size=128)

    logging.warning(msg={'title ebedding consumption time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))

    model.stop_multi_process_pool(pool)
    # title_embeddings
    # cos_sim_list=[util.cos_sim(model.encode(row.title, convert_to_tensor=True), model.encode(row.abstract, convert_to_tensor=True)).cpu().detach().numpy()[0][0] for row in ()]

    df_final_test['abs_agg_cos_sim']=util.pairwise_cos_sim(abstract_embeddings,aggtext_embeddings)

    logging.warning(msg={'abs_agg_cos_sim consumption time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))

    df_final_test['title_agg_cos_sim']=util.pairwise_cos_sim(title_embeddings,aggtext_embeddings)

    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'abs_agg_cos_sim consumption time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    logging.warning(msg={'cos_sim df_final_test':df_final_test.count()})
    logging.warning(msg={'cos_sim df_final_test':df_final_test[['abs_agg_cos_sim','title_agg_cos_sim']].head(2)})


    del model
    del aggtext_embeddings
    del abstract_embeddings
    del title_embeddings
    logging.warning(msg=u'after delete memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))


    #———————————- end_multi_process_pool -—————————————————
    #———————————- end_multi_process_pool -—————————————————

    # train and test data 
    logging.warning(msg='train and test data')
    y_train = df_final_train.indicator_link
    X_train = df_final_train.drop(['paper_id', 'software_id','indicator_link','title','aggtext','abstract'], axis = 1)
    logging.warning(msg={'X_train':X_train})

    y_test = df_final_test.indicator_link
    X_test = df_final_test.drop(['paper_id', 'software_id','indicator_link','title','aggtext','abstract'], axis = 1)
    logging.warning(msg={'X_test':X_test})

    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish X train test Y train test time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    logging.warning(msg="================================================================================")

    logging.warning(msg='start training')

    bst = XGBClassifier(n_estimators=1000, learning_rate=0.1,
                        eta=0.1, nrounds=1000, max_depth=8, colsample_bytree=0.7,
                        scale_pos_weight=1.0, booster='gbtree',subsample=0.9,
                        objective='binary:logistic',tree_method="hist",enable_categorical=True)

    bst.fit(X_train, y_train)
    pred = bst.predict(X_test)
    
    ground_truth=y_test

    acc=accuracy_score(ground_truth,pred)
    f1=f1_score(ground_truth,pred)
    r_a_s = roc_auc_score(ground_truth, pred)

    finish_xg_time=time.time()
    logging.warning(msg={'acc':acc,'f1':f1,'r_a_s':r_a_s})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish xgboost time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    bst.save_model("./model/categorical-model"+time.strftime('%y_%m_%d_%H_%M_%S',time.localtime(time.time()))+".json")
    logging.warning(msg='saved model!')

    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish xgboost time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    '''
    logging.warning(msg='train and test data')
    y_train = df_final_train.indicator_link
    X_train = df_final_train.drop(['paper_id', 'software_id','indicator_link','title','aggtext','abstract'], axis = 1)
    logging.warning(msg={'X_train':X_train})

    y_test = df_final_test.indicator_link
    X_test = df_final_test.drop(['paper_id', 'software_id','indicator_link','title','aggtext','abstract'], axis = 1)
    logging.warning(msg={'X_test':X_test})
    '''
    logging.warning(msg="data saving")
    path='./graph_and_embedding_data/'
    if os.path.exists(path):
        df_final_train.to_csv(os.path.join(path,'df_final_train.csv'),columns=df_final_train.columns,index=False)
        df_final_test.to_csv(os.path.join(path,'df_final_test.csv'),columns=df_final_test.columns,index=False)
    else:
        os.makedirs(path)
        df_final_train.to_csv(os.path.join(path,'df_final_train.csv'),columns=df_final_train.columns,index=False)
        df_final_test.to_csv(os.path.join(path,'df_final_test.csv'),columns=df_final_test.columns,index=False)
    logging.warning(msg="data saved")
    logging.warning(msg="_______________________________________________")
    
    #=======================================================
    '''

    from xgboost.sklearn import XGBClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import f1_score

    xgb = XGBClassifier(n_jobs = -1, random_state = 21)
    params = {'n_estimators': [150, 250, 500, 750], 
            'max_depth': [3, 5, 7, 15, 35]}
    clf = GridSearchCV(xgb, params, scoring = 'f1', cv = 5, return_train_score = True, n_jobs = -1)
    clf.fit(X_train, y_train)

    # plot the performance of model both on train data and cross validation data for each hyper parameter using a Seaborn Heatmap
    # reference -> https://kavisekhon.com/Grid%20Search.html

    sns.set()
    max_scores = pd.DataFrame(clf.cv_results_).groupby(
                ['param_n_estimators', 'param_max_depth']).max().unstack()[
                ['mean_train_score', 'mean_test_score']]

    fig, ax = plt.subplots(1,2, figsize = (18, 4))
    sns.heatmap(max_scores.mean_train_score, annot = True, ax = ax[0], fmt = '.5g', cmap = "YlGnBu")
    sns.heatmap(max_scores.mean_test_score, annot = True, ax = ax[1], fmt ='.5g', cmap = "YlGnBu")

    ax[0].set_title('Train Set', fontsize = 15)
    ax[1].set_title('CV Set', fontsize = 15)
    plt.show()

    # train xgboost model with the optimal hyperparameters found above 

    xgb = XGBClassifier(n_estimators = 500, max_depth = 15, n_jobs = -1, random_state = 21)
    xgb.fit(X_train, y_train)
    y_pred_train =  xgb.predict(X_train)
    y_pred_test = xgb.predict(X_test)
    train_f1 = f1_score(y_train, y_pred_train)
    test_f1 = f1_score(y_test, y_pred_test)

    
    

    # plotting confusion matrix, precision matrix and recall matrix

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    
    # plot_confusion_matrix(y_train, y_pred_train)
    
    

    
    # plot_confusion_matrix(y_test, y_pred_test)
    
    


    # relative feature importance

    feat_df = pd.DataFrame({'Features' : X_train.columns, 'Relative Importance' :xgb.feature_importances_})
    imp_feat_df = feat_df.sort_values('Relative Importance', ascending = False)[:25]
    sns.set(style = 'ticks')
    plt.figure(figsize = (8, 10))
    sns.barplot(y = 'Features', x = 'Relative Importance', data = imp_feat_df)
    plt.title('Feature Importance', fontsize = 15)
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.yticks(fontsize = 12)
    plt.grid()
    plt.show()

    '''