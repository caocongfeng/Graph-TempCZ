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

# import spacy
# nlp = spacy.load('en_core_web_lg')

logging.warning(msg={'file_name':os.path.basename(__file__)})

start_time=time.time()
logging.warning(msg='starting')
logging.warning(msg={'start time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

if __name__ == '__main__':

    com=pd.read_csv('../datasets/single_graph_merged_data.csv',engine="c")

    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'row_data_info':com.info()})
    logging.warning(msg={'row_data_count':com.count()})
    logging.warning(msg='finish read data')

    com=com.drop_duplicates(subset=['pmcid','ID'],keep='first')

    logging.warning(msg={'row_data_after_drop_info':com.info()})
    logging.warning(msg={'row_data_after_drop_count':com.count()})

    lbl = preprocessing.LabelEncoder()
    com['pubdate'] = lbl.fit_transform(com['pubdate'].astype(str))
    com['pubdate']=com['pubdate'].astype('category')

    logging.warning(msg=com.head(2))

    finish_read_data_time=time.time()
    logging.warning(msg=(finish_read_data_time-start_time))

    data_df=com
    # data_df=com.sample(int(com.shape[0]/2))
    logging.warning(msg={'original_data':data_df.count()})
    logging.warning(msg='finish sample')
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))

    data_df['y']=1

    del com
    gc.collect()

    logging.warning(msg='build paper and software nodes')
    logging.warning(msg='start negative sample')
    logging.warning(msg={'start negative sample time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    negative_sample_start_time=time.time()

    paper=data_df[['pmcid','title','abstract','pubdate']].drop_duplicates('pmcid',keep='first')
    software = data_df[['ID','curation_label','aggtext','mapped_to_software','text','weight']].drop_duplicates('ID',keep='first')

    paper_and_software_time=time.time()

    logging.warning(msg={'paper':paper.count()})
    logging.warning(msg={'software':software.count()})
    logging.warning(msg={'paper_and_software_time':(paper_and_software_time-negative_sample_start_time)})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))

    new_paper=data_df[['pmcid']].drop_duplicates('pmcid',keep='first')
    new_paper=new_paper.sample(30000)
    new_software = data_df[['ID']].drop_duplicates('ID',keep='first')
    new_software=new_software.sample(1200)

    new_paper_and_software_time=time.time()
    logging.warning(msg={'new_paper_count':new_paper.count()})
    logging.warning(msg={'new_software_count':new_software.count()})
    logging.warning(msg={'new_paper_and_software_time':(new_paper_and_software_time-paper_and_software_time)})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))

    all_data_df=new_paper.join(new_software,how='cross')
    all_data_df['y']=0

    logging.warning(msg={'before_append_all_data_df':all_data_df.count()})
    logging.warning(msg={'before_append_all_data_df_y':all_data_df.y.value_counts()})

    all_data_df=all_data_df.append(data_df[['pmcid','ID','y']])

    logging.warning(msg={'after_append_all_data_df':all_data_df.count()})
    logging.warning(msg={'after_append_all_data_df_y':all_data_df.y.value_counts()})

    all_data_df=all_data_df.drop_duplicates(subset=['pmcid','ID'],keep=False)

    logging.warning(msg={'after_drop_duplicates_all_data_df':all_data_df.count()})
    logging.warning(msg={'after_drop_duplicates_all_data_df_y':all_data_df.y.value_counts()})

    all_data_df_time=time.time()
    logging.warning(msg={'all_data_df_time':(all_data_df_time-new_paper_and_software_time)})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'before_condiction':all_data_df.count()})
    logging.warning(msg={'before_condiction_y':all_data_df.y.value_counts()})

    # control condition and data balance
    condition=all_data_df['y']==0
    all_data_df=all_data_df[condition]

    logging.warning(msg={'after_condiction':all_data_df.count()})
    logging.warning(msg={'after_condiction_y':all_data_df.y.value_counts()})

    na_data_df=all_data_df.sample(data_df.shape[0])

    logging.warning(msg={'negative sample count':na_data_df.shape})

    na_data_df=pd.merge(na_data_df,paper,on='pmcid')

    logging.warning(msg={'merge paper':na_data_df.shape})

    na_data_df=pd.merge(na_data_df,software,on='ID')

    logging.warning(msg={'merge software':na_data_df.shape})

    logging.warning(msg={'na_data_dfcount':na_data_df.count()})
    logging.warning(msg={'na_data_dfcount_y':na_data_df.y.value_counts()})
    # na_data_df['y']=0

    na_data_df_time=time.time()
    logging.warning(msg={'na_data_df_time':(na_data_df_time-all_data_df_time)})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish na_data_df time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    del paper
    del software
    del new_paper
    del new_software
    gc.collect()

    #===new negative sample=====

    finish_negative_sample_time=time.time()

    data_df=pd.concat([data_df,na_data_df],axis=0,ignore_index=True)

    del na_data_df
    gc.collect()

    logging.warning(msg={'concat_data_and_na_data':data_df.count()})
    logging.warning(msg={'concat_data_y':data_df.y.value_counts()})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish negative time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

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

    aggtext_embeddings = model.encode_multi_process(data_df.aggtext.to_list(),pool,batch_size=128)

    logging.warning(msg={'aggtext ebedding consumption time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))

    abstract_embeddings = model.encode_multi_process(data_df.abstract.to_list(),pool,batch_size=128)

    logging.warning(msg={'abstract ebedding consumption time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))

    title_embeddings = model.encode_multi_process(data_df.title.to_list(),pool,batch_size=128)

    logging.warning(msg={'title ebedding consumption time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))

    model.stop_multi_process_pool(pool)
    # title_embeddings
    # cos_sim_list=[util.cos_sim(model.encode(row.title, convert_to_tensor=True), model.encode(row.abstract, convert_to_tensor=True)).cpu().detach().numpy()[0][0] for row in ()]

    data_df['abs_agg_cos_sim']=util.pairwise_cos_sim(abstract_embeddings,aggtext_embeddings)

    logging.warning(msg={'abs_agg_cos_sim consumption time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))

    data_df['title_agg_cos_sim']=util.pairwise_cos_sim(title_embeddings,aggtext_embeddings)

    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'abs_agg_cos_sim consumption time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    logging.warning(msg={'cos_sim data_df':data_df.count()})
    logging.warning(msg={'cos_sim data_df':data_df[['abs_agg_cos_sim','title_agg_cos_sim']].head(2)})


    del model
    del aggtext_embeddings
    del abstract_embeddings
    del title_embeddings
    logging.warning(msg=u'after delete memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))

    #### training model
    logging.warning(msg={'training data Y':data_df.y.value_counts()})

    logging.warning(msg={'training data':data_df.count()})
    logging.warning(msg={'training data':data_df.info()})

    # save data
    logging.warning(msg='saving data')
    # data_df.to_csv('./data_df'+time.strftime('%y_%m_%d_%H_%M_%S',time.localtime(time.time()))+'index_false.csv',columns=data_df.columns,index=False)
    # data_df.to_csv('./data_df'+time.strftime('%y_%m_%d_%H_%M_%S',time.localtime(time.time()))+'index_True.csv',columns=data_df.columns,index=True)
    logging.warning(msg='savd data')


    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(data_df.drop(['pmcid','ID','text','weight','aggtext','title','abstract','curation_label','mapped_to_software','y'], axis= 1), data_df.y, test_size=0.2, shuffle=True)


    ##xgboost
    

    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV

    logging.warning(msg="start xgboost")
    logging.warning(msg={'start xhboost time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})

    start_xgboost_time=time.time()

    bst = XGBClassifier(n_estimators=1000, learning_rate=0.1,
                        eta=0.1, nrounds=1000, max_depth=8, colsample_bytree=0.7,
                        scale_pos_weight=1.0, booster='gbtree',subsample=0.9,
                        objective='binary:logistic',tree_method="hist",enable_categorical=True)

    bst.fit(X_train, y_train)
    pred = bst.predict(X_test)
    bst.save_model("./model/categorical-model"+time.strftime('%y_%m_%d_%H_%M_%S',time.localtime(time.time()))+".json")

    logging.warning(msg='saved model!')

    ground_truth=y_test

    acc=accuracy_score(ground_truth,pred)
    f1=f1_score(ground_truth,pred)
    r_a_s = roc_auc_score(ground_truth, pred)

    finish_xg_time=time.time()
    logging.warning(msg={'acc':acc,'f1':f1,'r_a_s':r_a_s})
    logging.warning(msg=(finish_xg_time-start_xgboost_time))
    logging.warning(msg=u'memory usage:%.2f GB'%(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 /1024))
    logging.warning(msg={'finish xgboost time':time.strftime('%y-%m-%d %H:%M:%S',time.localtime(time.time()))})
    logging.warning(msg="data saving")
    path='./embedding_data/'
    if os.path.exists(path):
        data_df.drop(['pmcid','ID','text','weight','aggtext','title','abstract','curation_label','mapped_to_software'], axis= 1).to_csv(os.path.join(path,'data_df.csv'),columns=X_train.columns,index=False)
    else:
        os.makedirs(path)
        data_df.drop(['pmcid','ID','text','weight','aggtext','title','abstract','curation_label','mapped_to_software'], axis= 1).to_csv(os.path.join(path,'data_df.csv'),columns=X_train.columns,index=False)
    logging.warning(msg="data saved")
    logging.warning(msg="_______________________________________________")
    logging.warning(msg="_______________________________________________")