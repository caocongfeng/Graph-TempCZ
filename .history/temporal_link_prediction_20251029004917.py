# -*- coding: utf-8 -*-
"""
Single-Year Train (8:1:1 In-Year Split) + All-Years Test (2010..2020, test split)
Graph cutoff: graph = K (use only edges with pubyear == K for message passing)
"""

import os
import json
import random
import gc
import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, HeteroConv

from sentence_transformers import SentenceTransformer
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             average_precision_score, recall_score)
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import tqdm
import wandb
import logging

# =========================
# 基础与全局参数（更省内存）
# =========================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

CACHE_DIR = Path("./single_year_split_cache_refine_2020_2021")
EMB_DIR = CACHE_DIR / "embeddings"
SPLIT_DIR = CACHE_DIR / "splits"
GRAPHS_DIR = CACHE_DIR / "graphs"
for d in [EMB_DIR, SPLIT_DIR, GRAPHS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 可选：更强确定性（PyG里某些算子可能不支持）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(False)

# =========================
# 数据加载 & 年份
# =========================
def load_and_prepare_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logging.info("从CSV加载数据...")
    data_df = pd.read_csv(data_path, engine="c")

    # pubyear
    if 'pubyear' not in data_df.columns:
        data_df['pubyear'] = data_df['pubdate'].astype(int)

    # 唯一论文与软件文本
    paper = data_df[['pmcid', 'title', 'abstract', 'pubdate']].drop_duplicates(subset=['pmcid']).reset_index(drop=True)
    software_mentions = data_df[['ID', 'aggtext']].drop_duplicates()
    software_text = software_mentions.groupby('ID')['aggtext'].apply(lambda texts: ', '.join(texts.astype(str))).reset_index()
    software_text = software_text.rename(columns={'aggtext': 'text'})

    logging.info(f"加载了 {len(paper)} 篇唯一论文和 {len(software_text)} 个唯一软件实体。")
    yr_min, yr_max = int(data_df['pubyear'].min()), int(data_df['pubyear'].max())
    print(f"[pubyear] year range: {yr_min} -> {yr_max}")
    return data_df, paper, software_text

# =========================
# Embedding 缓存
# =========================
def _emb_paths(model_name: str) -> Dict[str, Path]:
    safe = model_name.replace("/", "_")
    return {
        "paper_ids": EMB_DIR / f"paper_ids_{safe}.csv",
        "paper_emb": EMB_DIR / f"paper_emb_{safe}.npy",
        "sw_ids": EMB_DIR / f"software_ids_{safe}.csv",
        "sw_emb": EMB_DIR / f"software_emb_{safe}.npy",
        "meta": EMB_DIR / f"meta_{safe}.json",
    }

def _save_embeddings(model_name: str, paper: pd.DataFrame, software_text: pd.DataFrame,
                     paper_emb: np.ndarray, sw_emb: np.ndarray):
    paths = _emb_paths(model_name)
    paper[['pmcid']].to_csv(paths["paper_ids"], index=False)
    software_text[['ID']].to_csv(paths["sw_ids"], index=False)
    np.save(paths["paper_emb"], paper_emb)
    np.save(paths["sw_emb"], sw_emb)
    meta = {
        "model": model_name,
        "paper_count": int(paper_emb.shape[0]),
        "software_count": int(sw_emb.shape[0]),
        "dim": int(sw_emb.shape[1]),
        "saved_at": datetime.datetime.now().isoformat()
    }
    with open(paths["meta"], "w") as f:
        json.dump(meta, f)

def _try_load_embeddings(model_name: str, paper: pd.DataFrame, software_text: pd.DataFrame
                        ) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    paths = _emb_paths(model_name)
    try:
        for p in paths.values():
            if not p.exists():
                return None
        saved_paper_ids = pd.read_csv(paths["paper_ids"])['pmcid'].tolist()
        saved_sw_ids = pd.read_csv(paths["sw_ids"])['ID'].tolist()
        if saved_paper_ids != paper['pmcid'].tolist() or saved_sw_ids != software_text['ID'].tolist():
            logging.warning("缓存的 embedding ID 顺序与当前数据不一致，将放弃缓存并重新编码。")
            return None
        paper_emb = np.load(paths["paper_emb"], mmap_mode='r')
        sw_emb = np.load(paths["sw_emb"], mmap_mode='r')
        with open(paths["meta"], "r") as f:
            meta = json.load(f)
        emb_dim = int(meta.get("dim", sw_emb.shape[1]))
        logging.info(f"已从本地缓存加载 embeddings（维度={emb_dim}）：paper={paper_emb.shape}, software={sw_emb.shape}")
        return np.array(paper_emb), np.array(sw_emb), emb_dim
    except Exception as e:
        logging.warning(f"加载本地 embedding 失败：{e}")
        return None

def compute_or_load_embeddings(model_name: str, paper: pd.DataFrame, software_text: pd.DataFrame,
                               device: torch.device) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    loaded = _try_load_embeddings(model_name, paper, software_text)
    if loaded is not None:
        paper_emb, sw_emb, emb_dim = loaded
        paper = paper.copy(); software_text = software_text.copy()
        paper['embedding'] = list(paper_emb); software_text['embedding'] = list(sw_emb)
        return paper, software_text, emb_dim

    logging.info(f"初始化SentenceTransformer模型: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    logging.info("正在为软件计算嵌入...")
    sw_emb = model.encode(
        software_text['text'].tolist(),
        batch_size=128, show_progress_bar=True, device=device,
        convert_to_tensor=True, truncate=True
    ).cpu().numpy()

    logging.info("正在为论文计算嵌入...")
    p_texts = (paper['title'].fillna('') + ' ' + paper['abstract'].fillna('')).tolist()
    paper_emb = model.encode(
        p_texts, batch_size=128, show_progress_bar=True, device=device,
        convert_to_tensor=True, truncate=True
    ).cpu().numpy()

    emb_dim = sw_emb.shape[1]
    logging.info(f"嵌入计算完成，维度为: {emb_dim}")

    _save_embeddings(model_name, paper, software_text, paper_emb, sw_emb)

    paper = paper.copy(); software_text = software_text.copy()
    paper['embedding'] = list(paper_emb); software_text['embedding'] = list(sw_emb)
    del paper_emb, sw_emb; gc.collect()
    return paper, software_text, emb_dim

# =========================
# 基础节点特征 / ID 映射
# =========================
def build_id_maps(paper: pd.DataFrame, software_text: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    paper_id_map = paper[['pmcid']].drop_duplicates().reset_index().rename(columns={'index': 'mappedID'})
    software_id_map = software_text[['ID']].drop_duplicates().reset_index().rename(columns={'index': 'mappedID'})
    return paper_id_map, software_id_map

def build_heterodata_base(paper: pd.DataFrame, software_text: pd.DataFrame) -> HeteroData:
    paper_x = torch.as_tensor(np.vstack(paper['embedding'].values), dtype=torch.float)
    software_x = torch.as_tensor(np.vstack(software_text['embedding'].values), dtype=torch.float)

    data = HeteroData()
    data['paper'].x = paper_x
    data['paper'].node_id = torch.arange(paper_x.shape[0], dtype=torch.long)
    data['software'].x = software_x
    data['software'].node_id = torch.arange(software_x.shape[0], dtype=torch.long)
    return data

# =========================
# 图结构缓存（== year 的边）
# =========================
def graph_edges_cache_path(max_year: int) -> Path:
    return GRAPHS_DIR / f"graph_edges_le_{max_year}.parquet"

def build_graph_edges_up_to_year_cached(data_df: pd.DataFrame, max_year: int) -> pd.DataFrame:
    path = graph_edges_cache_path(max_year)
    if path.exists():
        try:
            df = pd.read_parquet(path)
            if {'pmcid','ID'}.issubset(df.columns):
                logging.info(f"已加载缓存的图结构边（== {max_year}）：{len(df)} 条")
                return df
        except Exception as e:
            logging.warning(f"读取缓存图结构失败，将重新构建：{e}")
    sub = data_df.loc[data_df['pubyear'] == max_year, ['pmcid','ID']].drop_duplicates()
    sub.to_parquet(path, index=False)
    logging.info(f"已构建并缓存图结构边（== {max_year}）：{len(sub)} 条 -> {path}")
    return sub

# =========================
# 年内 8:1:1 划分（按每年）
# =========================
def build_year_splits_cached(
    data_df: pd.DataFrame, start_year: int, end_year: int,
    train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
    seed: int = 42
) -> pd.DataFrame:
    cache_path = SPLIT_DIR / f"year_splits_{start_year}_{end_year}.parquet"
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            if {'pmcid','ID','pubyear','split'}.issubset(df.columns):
                logging.info(f"[splits] 加载缓存的年内划分：{cache_path} ({len(df)} rows)")
                return df
        except Exception as e:
            logging.warning(f"[splits] 读取失败，重建：{e}")

    rng = np.random.default_rng(seed)
    parts = []
    for y in range(start_year, end_year + 1):
        sub = data_df.loc[data_df['pubyear'] == y, ['pmcid','ID']].drop_duplicates()
        n = len(sub)
        if n == 0:
            continue
        idx = np.arange(n); rng.shuffle(idx)
        n_train = int(round(train_ratio * n))
        n_val   = int(round(val_ratio * n))
        n_test  = n - n_train - n_val
        # 边界修正
        if n_test < 0:
            n_test = 0
        if n_train == 0 and n > 0:
            n_train = min(1, n)

        tr = sub.iloc[idx[:n_train]].copy(); tr['pubyear'] = y; tr['split'] = 'train'
        va = sub.iloc[idx[n_train:n_train+n_val]].copy(); va['pubyear'] = y; va['split'] = 'val'
        te = sub.iloc[idx[n_train+n_val:]].copy(); te['pubyear'] = y; te['split'] = 'test'
        parts += [tr, va, te]

    splits = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['pmcid','ID','pubyear','split'])
    splits.to_parquet(cache_path, index=False)
    logging.info(f"[splits] 年内划分完成并缓存：{cache_path} ({len(splits)} rows)")
    return splits

# =========================
# 标签（基于 split + 年份）缓存
# =========================
def make_edge_labels_for_years_with_split_cached(
    data_df: pd.DataFrame,
    year_splits: pd.DataFrame,
    years: List[int],
    split_tag: str,               # 'train' | 'val' | 'test'
    restrict_nodes_to_history: bool,
    history_max_year: int,
    seed: int,
    cache_dir: Path,
    neg_ratio: float,
    max_neg_per_paper: Optional[int] = None
) -> pd.DataFrame:
    years_key = f"{years[0]}_{years[-1]}" if len(years) > 1 else f"{years[0]}"
    labels_path = cache_dir / f"labels_{split_tag}_years_{years_key}_histLE_{history_max_year}.parquet"
    if labels_path.exists():
        try:
            df = pd.read_parquet(labels_path)
            if {'pmcid','ID','label'}.issubset(df.columns):
                logging.info(f"[labels/{split_tag}] 读取缓存：{labels_path} ({len(df)} rows)")
                return df
        except Exception as e:
            logging.warning(f"[labels/{split_tag}] 读取失败，将重建：{e}")

    split_pos = year_splits.loc[
        (year_splits['pubyear'].isin(years)) & (year_splits['split'] == split_tag),
        ['pmcid','ID','pubyear']
    ].drop_duplicates()

    if len(split_pos) == 0:
        empty = pd.DataFrame(columns=['pmcid','ID','label'])
        empty.to_parquet(labels_path, index=False)
        return empty

    if restrict_nodes_to_history:
        hist_soft = data_df.loc[data_df['pubyear'] <= history_max_year, 'ID'].unique()
    else:
        hist_soft = data_df['ID'].unique()

    grp = split_pos.groupby('pmcid')['ID'].apply(lambda s: set(s.values))
    pos_records = split_pos[['pmcid','ID']].drop_duplicates().assign(label=1)

    rng = np.random.default_rng(seed)
    neg_rows = []
    for pmcid, pos_set in grp.items():
        pos_k = len(pos_set)
        if pos_k == 0:
            continue
        target_neg = max(1 if neg_ratio > 0 else 0, int(np.ceil(neg_ratio * pos_k)))
        if max_neg_per_paper is not None:
            target_neg = min(target_neg, max_neg_per_paper)
        draw_n = max(target_neg, min(len(hist_soft), (target_neg * 3) // 2))
        cand = rng.choice(hist_soft, size=draw_n, replace=draw_n > len(hist_soft))
        seen = set()
        for sid in cand:
            if sid in pos_set or sid in seen:
                continue
            seen.add(sid)
            neg_rows.append((pmcid, sid, 0))
            if len(seen) >= target_neg:
                break

    neg_df = pd.DataFrame(neg_rows, columns=['pmcid','ID','label'])
    edge_lab = pd.concat([pos_records[['pmcid','ID','label']], neg_df], axis=0, ignore_index=True)
    edge_lab.to_parquet(labels_path, index=False)
    logging.info(f"[labels/{split_tag}] 生成并缓存：{labels_path} (pos={len(pos_records)}, neg={len(neg_df)})")
    return edge_lab

# =========================
# 组装 HeteroData（graph <= K + 指定 years 的指定 split）
# =========================
# def assemble_dataset_from_cache_with_split(
#     base_data: HeteroData, data_df: pd.DataFrame,
#     paper_id_map: pd.DataFrame, software_id_map: pd.DataFrame,
#     year_splits: pd.DataFrame,
#     graph_max_year: int, label_years: List[int],
#     neg_ratio: float, seed: int, split_dir: Path,
#     split_tag: str
# ) -> HeteroData:
#     # 图边：始终用 <= graph_max_year
#     edges_df = build_graph_edges_up_to_year_cached(data_df, graph_max_year)

#     # 标签：只取指定 years 的 split = split_tag
#     labels_df = make_edge_labels_for_years_with_split_cached(
#         data_df=data_df, year_splits=year_splits, years=label_years, split_tag=split_tag,
#         restrict_nodes_to_history=True, history_max_year=graph_max_year,
#         seed=seed, cache_dir=split_dir, neg_ratio=neg_ratio
#     )

#     edges_map = edges_df.merge(paper_id_map, on='pmcid').merge(software_id_map, on='ID')[['mappedID_x','mappedID_y']]
#     edge_index = torch.as_tensor(np.vstack([edges_map['mappedID_x'].values, edges_map['mappedID_y'].values]),
#                                  dtype=torch.long)

#     data = HeteroData()
#     data['paper'].x = base_data['paper'].x
#     data['paper'].node_id = base_data['paper'].node_id
#     data['software'].x = base_data['software'].x
#     data['software'].node_id = base_data['software'].node_id
#     data['paper','mention','software'].edge_index = edge_index

#     if len(labels_df) == 0:
#         data['paper','mention','software'].edge_label_index = torch.empty((2,0), dtype=torch.long)
#         data['paper','mention','software'].edge_label = torch.empty((0,), dtype=torch.float)
#         return T.ToUndirected()(data)

#     labels_map = labels_df.merge(paper_id_map, on='pmcid').merge(software_id_map, on='ID')[['mappedID_x','mappedID_y','label']]
#     edge_label_index = torch.as_tensor(np.vstack([labels_map['mappedID_x'].values, labels_map['mappedID_y'].values]),
#                                        dtype=torch.long)
#     edge_label = torch.as_tensor(labels_map['label'].values, dtype=torch.float)
#     data['paper','mention','software'].edge_label_index = edge_label_index
#     data['paper','mention','software'].edge_label = edge_label
#     data = T.ToUndirected()(data)
#     return data

# =========================
# 组装 HeteroData（graph <= K + 指定 years 的指定 split） 预防数据泄露
# =========================
def assemble_dataset_from_cache_with_split(
    base_data: HeteroData, data_df: pd.DataFrame,
    paper_id_map: pd.DataFrame, software_id_map: pd.DataFrame,
    year_splits: pd.DataFrame,
    graph_max_year: int, label_years: List[int],
    neg_ratio: float, seed: int, split_dir: Path,
    split_tag: str
) -> HeteroData:
    # 1) 历史观测边（<= graph_max_year）
    edges_df = build_graph_edges_up_to_year_cached(data_df, graph_max_year)

    # 2) 本 split 的标签（含正负，正例需要从 edge_index 中移除）
    labels_df = make_edge_labels_for_years_with_split_cached(
        data_df=data_df, year_splits=year_splits, years=label_years, split_tag=split_tag,
        restrict_nodes_to_history=True, history_max_year=graph_max_year,
        seed=seed, cache_dir=split_dir, neg_ratio=neg_ratio
    )

    # --- 构造 ID 映射
    edges_map = edges_df.merge(paper_id_map, on='pmcid').merge(software_id_map, on='ID')[['mappedID_x','mappedID_y']]
    # label 正例（只取 label=1）
    pos_labels = labels_df.loc[labels_df['label'] == 1, ['pmcid','ID']]
    if len(pos_labels) > 0:
        pos_map = pos_labels.merge(paper_id_map, on='pmcid').merge(software_id_map, on='ID')[['mappedID_x','mappedID_y']]
        # 从 edges_map 中移除这些正边（避免边泄露）
        pos_tuples = set(map(tuple, pos_map.to_numpy()))
        # 过滤
        edges_map = edges_map[~edges_map.apply(lambda r: (r['mappedID_x'], r['mappedID_y']) in pos_tuples, axis=1)]
        # 注意：ToUndirected() 会自动添加反向边，因此移除正向即可（也可同时移除反向，效果等价）

    edge_index = torch.as_tensor(
        np.vstack([edges_map['mappedID_x'].values, edges_map['mappedID_y'].values]),
        dtype=torch.long
    )

    # --- 组装数据对象
    data = HeteroData()
    data['paper'].x = base_data['paper'].x
    data['paper'].node_id = base_data['paper'].node_id
    data['software'].x = base_data['software'].x
    data['software'].node_id = base_data['software'].node_id
    data['paper','mention','software'].edge_index = edge_index

    if len(labels_df) == 0:
        data['paper','mention','software'].edge_label_index = torch.empty((2,0), dtype=torch.long)
        data['paper','mention','software'].edge_label = torch.empty((0,), dtype=torch.float)
        return T.ToUndirected()(data)

    labels_map = labels_df.merge(paper_id_map, on='pmcid').merge(software_id_map, on='ID')[['mappedID_x','mappedID_y','label']]
    edge_label_index = torch.as_tensor(
        np.vstack([labels_map['mappedID_x'].values, labels_map['mappedID_y'].values]),
        dtype=torch.long
    )
    edge_label = torch.as_tensor(labels_map['label'].values, dtype=torch.float)

    data['paper','mention','software'].edge_label_index = edge_label_index
    data['paper','mention','software'].edge_label = edge_label

    # 统一转无向（会自动加反向边。我们已经把待预测正边的“正向”从 edge_index 里删掉了；
    # 反向边也不会出现，因为是 ToUndirected 在删边之后才添加的）
    data = T.ToUndirected()(data)
    return data

# =========================
# 模型（HeteroSAGE，避免 to_hetero）
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()

class Classifier(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x_p, x_s, edge_label_index):
        p = x_p[edge_label_index[0]]
        s = x_s[edge_label_index[1]]
        return self.mlp(torch.cat([p, s], dim=-1)).squeeze(-1)

class HeteroSAGE(nn.Module):
    """
    显式异构图：每层用 HeteroConv 包多个边类型的 SAGEConv，sum 聚合。
    （metadata 由训练图生成，包含反向边类型）
    """
    def __init__(self, hidden_dim: int, metadata):
        super().__init__()
        node_types, edge_types = metadata
        convs1 = {et: SAGEConv((-1, -1), hidden_dim) for et in edge_types}
        convs2 = {et: SAGEConv((-1, -1), hidden_dim) for et in edge_types}
        self.layer1 = HeteroConv(convs1, aggr='sum')
        self.layer2 = HeteroConv(convs2, aggr='sum')
        self.dropout = nn.Dropout(0.5)

    def forward(self, x_dict, edge_index_dict):
        h0 = x_dict
        h1 = self.layer1(h0, edge_index_dict)
        for ntype in h0.keys():
            if ntype not in h1:
                h1[ntype] = h0[ntype]
        h1 = {k: F.relu(v) for k, v in h1.items()}
        h1 = {k: self.dropout(v) for k, v in h1.items()}

        h2 = self.layer2(h1, edge_index_dict)
        for ntype in h1.keys():
            if ntype not in h2:
                h2[ntype] = h1[ntype]
        return h2

class Model(nn.Module):
    def __init__(self, emb_dim, hidden_dim, metadata, num_papers, num_software):
        super().__init__()
        self.paper_emb = nn.Embedding(num_papers, hidden_dim)
        self.software_emb = nn.Embedding(num_software, hidden_dim)
        nn.init.xavier_uniform_(self.paper_emb.weight)
        nn.init.xavier_uniform_(self.software_emb.weight)
        self.paper_lin = nn.Linear(emb_dim, hidden_dim)
        self.software_lin = nn.Linear(emb_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.gnn = HeteroSAGE(hidden_dim, metadata)
        self.classifier = Classifier(hidden_dim)

    def forward(self, data):
        x_p = self.dropout(self.paper_lin(data['paper'].x)) + self.paper_emb(data['paper'].node_id)
        x_s = self.dropout(self.software_lin(data['software'].x)) + self.software_emb(data['software'].node_id)
        x_dict = {'paper': x_p, 'software': x_s}
        x_out = self.gnn(x_dict, data.edge_index_dict)
        edge_idx = data['paper','mention','software'].edge_label_index
        return self.classifier(x_out['paper'], x_out['software'], edge_idx)

# =========================
# Loader / 评估 / 工具函数
# =========================
def get_loader(subset: HeteroData, batch_size: int, shuffle=True, num_workers: int = 0, seed: int = 42):
    g = torch.Generator()
    g.manual_seed(seed)
    return LinkNeighborLoader(
        data=subset,
        num_neighbors=[15, 8],
        edge_label_index=(('paper','mention','software'), subset['paper','mention','software'].edge_label_index),
        edge_label=subset['paper','mention','software'].edge_label,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=g,
    )

def print_label_distribution(loader, name=""):
    labels = []
    for batch in loader:
        l = batch["paper", "mention", "software"].edge_label
        labels.extend(l.cpu().detach().numpy())
    labels = np.array(labels)
    print(f"\n==== {name} 加载器标签分布 ====")
    counts = pd.Series(labels).value_counts().sort_index()
    print(counts)
    if len(labels) > 0:
        pos_ratio = counts.get(1, 0) / len(labels)
        print(f"总样本数: {len(labels)}, 正样本比例: {pos_ratio:.4f}")
    print("=" * 35)

def _safe_auc(y_true_np, y_prob_np):
    if len(np.unique(y_true_np)) < 2:
        return float('nan')
    return roc_auc_score(y_true_np, y_prob_np)

def _safe_ap(y_true_np, y_prob_np):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        return float(average_precision_score(y_true_np, y_prob_np))

def evaluate(model, loader, device):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.cpu())
            labels = batch['paper','mention','software'].edge_label
            labels = torch.where(labels>1, torch.ones_like(labels), labels)
            truths.append(labels.cpu())

    if len(preds) == 0:
        return {'acc': 0.0, 'f1': 0.0, 'recall': 0.0, 'auc': float('nan'), 'ap': float('nan'), 'loss': 0.0}

    pred = torch.cat(preds)
    truth = torch.cat(truths).float()
    prob = torch.sigmoid(pred)
    pred_label = (prob>0.5).long()

    y_true_np = truth.numpy()
    y_prob_np = prob.numpy()
    y_pred_np = pred_label.numpy()

    try:
        acc = accuracy_score(y_true_np, y_pred_np)
    except Exception:
        acc = 0.0
    try:
        f1 = f1_score(y_true_np, y_pred_np, zero_division=0)
        rec = recall_score(y_true_np, y_pred_np, zero_division=0)
    except Exception:
        f1, rec = 0.0, 0.0

    auc  = _safe_auc(y_true_np, y_prob_np)
    ap   = _safe_ap(y_true_np, y_prob_np)
    loss = F.binary_cross_entropy_with_logits(pred, truth).item()

    return {'acc': acc, 'f1': f1, 'recall': rec, 'auc': auc, 'ap': ap, 'loss': loss}

def predict_and_save_results(model, loader, device, data_df, paper, software_text, output_filename="test_predictions"):
    logging.info(f"开始在测试集上进行预测以保存结果...")
    model.eval()
    all_edge_indices, all_preds, all_truths = [], [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="在测试集上预测"):
            batch = batch.to(device)
            out = model(batch)
            all_preds.append(out.cpu())
            all_truths.append(batch["paper", "mention", "software"].edge_label.cpu())
            all_edge_indices.append(batch["paper", "mention", "software"].edge_label_index.cpu())

    if len(all_preds) == 0:
        logging.warning("测试集为空，未生成预测文件。")
        return

    preds_tensor = torch.cat(all_preds)
    truths_tensor = torch.cat(all_truths)
    edge_indices_tensor = torch.cat(all_edge_indices, dim=1)

    probs = torch.sigmoid(preds_tensor)
    pred_labels = (probs > 0.5).long()

    paper_id_map = paper[['pmcid']].drop_duplicates().reset_index().rename(columns={'index': 'mappedID'})
    software_id_map = software_text[['ID']].drop_duplicates().reset_index().rename(columns={'index': 'mappedID'})
    paper_id_rev = paper_id_map.set_index('mappedID')['pmcid'].to_dict()
    software_id_rev = software_id_map.set_index('mappedID')['ID'].to_dict()

    pmcids = [paper_id_rev.get(i) for i in edge_indices_tensor[0].numpy()]
    software_ids = [software_id_rev.get(i) for i in edge_indices_tensor[1].numpy()]

    results_df = pd.DataFrame({
        'pmcid': pmcids,
        'ID': software_ids,
        'ground_truth': truths_tensor.numpy(),
        'predicted_label': pred_labels.numpy(),
        'predicted_prob': probs.numpy()
    })
    results_df.dropna(subset=['pmcid', 'ID'], inplace=True)

    # 更稳健的列选择（你的数据里是 aggtext / title / abstract / pubdate / pubyear）
    cols_to_keep = ['pmcid', 'ID']
    for col in ['aggtext', 'title', 'abstract', 'pubdate', 'pubyear']:
        if col in data_df.columns:
            cols_to_keep.append(col)
    original_unique_edges = data_df[cols_to_keep].drop_duplicates(subset=['pmcid', 'ID'])
    final_df = pd.merge(results_df, original_unique_edges, on=['pmcid', 'ID'], how='left')

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{output_filename}_{timestamp}.csv"
    final_df.to_csv(out_path, index=False)
    logging.info(f"测试预测结果已保存到 {out_path}")

# =========================
# 多年评测（test split）
# =========================
def evaluate_on_years(
    model, base_data, data_df, paper_id_map, software_id_map,
    year_splits: pd.DataFrame,
    graph_max_year: int, test_years: List[int], neg_ratio: float,
    device, split_dir: Path, batch_size: int, num_workers: int,
    run_name_prefix: str, paper=None, software_text=None
):
    results = {}
    for ty in test_years:
        test_data = assemble_dataset_from_cache_with_split(
            base_data, data_df, paper_id_map, software_id_map, year_splits,
            graph_max_year=graph_max_year, label_years=[ty],
            neg_ratio=neg_ratio, seed=42, split_dir=split_dir, split_tag='test'
        )
        test_loader = get_loader(test_data, batch_size*2, shuffle=False, num_workers=num_workers)
        metrics = evaluate(model, test_loader, device)
        results[ty] = metrics
        logging.info(f"[{run_name_prefix}] TEST@{ty} -> AUC:{metrics['auc']:.4f}  AP:{metrics['ap']:.4f}  "
                     f"F1:{metrics['f1']:.4f}  Recall:{metrics['recall']:.4f}")
        if wandb.run:
            wandb.log({f"{run_name_prefix}_test{ty}_{k}": v for k, v in metrics.items()})
        # 如需各年CSV，放开下面
        # if paper is not None and software_text is not None:
        #     predict_and_save_results(model, test_loader, device, data_df, paper, software_text,
        #                              output_filename=f"{run_name_prefix}_year{ty}")
        del test_data, test_loader
        torch.cuda.empty_cache(); gc.collect()
    return results

# =========================
# 训练一个窗口（K）：year=K 的 train/val；graph=K
# =========================
def train_one_window(
    base_data: HeteroData, _metadata_unused, data_df: pd.DataFrame,
    paper_id_map: pd.DataFrame, software_id_map: pd.DataFrame,
    year_splits: pd.DataFrame,           # 新增
    train_years: List[int], graph_max_year: int, test_year: int,
    batch_size: int, hidden_dim: int, emb_dim: int, neg_ratio: float,
    num_epochs: int, lr: float, device: torch.device,
    run_name_for_wandb: Optional[str] = None, num_workers: int = 0, seed: int = 42
):
    set_seed(seed)
    win_dir = split_dir_for_window(train_years, graph_max_year, test_year, neg_ratio, seed)

    # year=K 的训练/验证集（按 split）
    train_all = assemble_dataset_from_cache_with_split(
        base_data, data_df, paper_id_map, software_id_map, year_splits,
        graph_max_year=graph_max_year, label_years=train_years,
        neg_ratio=neg_ratio, seed=seed, split_dir=win_dir, split_tag='train'
    )
    val_all = assemble_dataset_from_cache_with_split(
        base_data, data_df, paper_id_map, software_id_map, year_splits,
        graph_max_year=graph_max_year, label_years=train_years,
        neg_ratio=neg_ratio, seed=seed, split_dir=win_dir, split_tag='val'
    )
    if train_all['paper','mention','software'].edge_label.numel() == 0:
        logging.warning(f"训练窗口 {train_years} 无训练样本，跳过。")
        return None, None

    train_loader = get_loader(train_all, batch_size, shuffle=True,  num_workers=num_workers, seed=seed)
    val_loader   = get_loader(val_all,   batch_size*2, shuffle=False, num_workers=num_workers, seed=seed)

    print_label_distribution(train_loader, f"Train-{train_years[0]}")
    print_label_distribution(val_loader,   f"Val-{train_years[0]}")

    metadata = train_all.metadata()  # 确保包含反向边类型
    logging.info(f"Hetero edge types used: {metadata[1]}")

    model = Model(
        emb_dim=emb_dim, hidden_dim=hidden_dim, metadata=metadata,
        num_papers=base_data['paper'].num_nodes, num_software=base_data['software'].num_nodes
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = FocalLoss(alpha=1.0, gamma=2.0)

    best_val_loss = float('inf')
    patience = 0
    BEST_MODEL_PATH = win_dir / f"best_model_{run_name_for_wandb or 'tmp'}.pt"

    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss, total_count = 0.0, 0
        pbar = tqdm.tqdm(train_loader, desc=f"[{run_name_for_wandb}] Epoch {epoch:02d}")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            y = batch['paper','mention','software'].edge_label.float()
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * out.numel()
            total_count += out.numel()
            pbar.set_postfix({'loss': total_loss / max(total_count,1)})

        avg_loss = total_loss / max(total_count,1)
        val_metrics = evaluate(model, val_loader, device)
        logging.info(
            f"[{run_name_for_wandb}] Epoch {epoch:03d} | 训练损失: {avg_loss:.4f} | "
            f"验证 AUC: {val_metrics['auc']:.4f}, 验证 AP: {val_metrics['ap']:.4f}, "
            f"验证 F1: {val_metrics['f1']:.4f}, 验证 Recall: {val_metrics['recall']:.4f}"
        )
        if wandb.run:
            wandb.log({
                f"{run_name_for_wandb}_epoch": epoch,
                f"{run_name_for_wandb}_train_loss": avg_loss,
                **{f"{run_name_for_wandb}_val_{k}": v for k, v in val_metrics.items()}
            })

        scheduler.step(val_metrics['loss'])
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            patience = 0
        else:
            patience += 1
            if patience > 8:  # 更快早停
                logging.info(f"[{run_name_for_wandb}] Early stopping triggered")
                break

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    del train_loader, val_loader
    torch.cuda.empty_cache(); gc.collect()
    return model, win_dir

# =========================
# 窗口目录命名 & 计划表
# =========================
def split_dir_for_window(train_years: List[int], graph_max_year: int, test_year: int,
                         neg_ratio: float, seed: int) -> Path:
    tname = f"{train_years[0]}to{train_years[-1]}_{len(train_years)}y"
    win_dir = SPLIT_DIR / f"win_train_{tname}_graphLE_{graph_max_year}_test_{test_year}_neg{neg_ratio}_seed{seed}"
    win_dir.mkdir(parents=True, exist_ok=True)
    return win_dir

def upgraded_single_year_plans(start_year: int = 2010, end_year: int = 2020) -> List[Tuple[List[int], int, List[int]]]:
    """对每个 K ∈ [start..end-1]：train=[K] | graph=K | test years=[2010..2020]"""
    all_years = list(range(start_year, end_year + 1))
    plans = []
    for K in range(start_year, end_year):  # 2010..2019
        plans.append(([K], K, all_years))
    return plans

# ------------------------------
# 训练主流程（升级版）
# ------------------------------
if __name__ == '__main__':
    # =========================
    # 配置
    # =========================
    DATA_PATH = '../datasets/single_graph_merged_data.csv'
    EMB_MODEL = 'all-MiniLM-L6-v2'
    BATCH_SIZE = 256
    HIDDEN_DIM = 128
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    NEG_SAMPLING_RATIO = 2.0
    SEED = 42
    NUM_WORKERS = max(0, min(4, (os.cpu_count() or 2) - 1))

    YEAR_START = 2010
    YEAR_END   = 2022

    set_seed(SEED)

    # =========================
    # wandb 初始化（可选）
    # =========================
    config = {
        'lr': LEARNING_RATE, 'hidden_dim': HIDDEN_DIM, 'batch_size': BATCH_SIZE,
        'model': EMB_MODEL, 'neg_sampling_ratio': NEG_SAMPLING_RATIO,
        'epochs': NUM_EPOCHS, 'year_start': YEAR_START, 'year_end': YEAR_END, 'seed': SEED,
        'protocol': 'SingleYear(8:1:1 in-year train/val; test=all years), graph=K'
    }
    try:
        wandb.login()
        wandb.init(project="HeteroGNN_Link_Prediction_single_year", entity="byfrfy", config=config)
    except Exception as e:
        logging.warning(f"无法初始化wandb: {e}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")

    # =========================
    # 数据与嵌入（带缓存）
    # =========================
    data_df, paper, software_text = load_and_prepare_data(DATA_PATH)
    paper, software_text, emb_dim = compute_or_load_embeddings(EMB_MODEL, paper, software_text, device)

    # =========================
    # 基础节点特征 + ID 映射
    # =========================
    base_data = build_heterodata_base(paper, software_text)
    paper_id_map, software_id_map = build_id_maps(paper, software_text)

    # =========================
    # 年内 8:1:1 划分（缓存）
    # =========================
    year_splits = build_year_splits_cached(data_df, YEAR_START, YEAR_END, seed=SEED)

    # =========================
    # 计划表：train=[K], graph=K, test years=2010..2020
    # =========================
    plans = upgraded_single_year_plans(YEAR_START, YEAR_END)

    # =========================
    # 逐计划执行：一次训练（当年K），在所有年份测试
    # =========================
    for train_years, graph_max_year, test_years in plans:
        run_name = f"train_{train_years[0]}__tests_{test_years[0]}_{test_years[-1]}"
        logging.info(f"\n===== 当前窗口 =====\ntrain_years={train_years}, graph= {graph_max_year}, test_years={test_years}\n")

        # 训练（test_year仅用于缓存目录命名）
        model, split_dir = train_one_window(
            base_data, None, data_df, paper_id_map, software_id_map,
            year_splits=year_splits,
            train_years=train_years, graph_max_year=graph_max_year, test_year=test_years[0],
            batch_size=BATCH_SIZE, hidden_dim=HIDDEN_DIM, emb_dim=emb_dim,
            neg_ratio=NEG_SAMPLING_RATIO, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
            device=device, run_name_for_wandb=run_name, num_workers=NUM_WORKERS, seed=SEED
        )

        if model is None:
            logging.warning(f"窗口 {train_years} 无训练样本，跳过。")
            continue

        # 多年评测（test split）
        results = evaluate_on_years(
            model=model, base_data=base_data, data_df=data_df,
            paper_id_map=paper_id_map, software_id_map=software_id_map,
            year_splits=year_splits,
            graph_max_year=graph_max_year, test_years=test_years,
            neg_ratio=NEG_SAMPLING_RATIO, device=device,
            split_dir=split_dir, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
            run_name_prefix=run_name, paper=paper, software_text=software_text
        )

        logging.info(f"窗口 {train_years} 多年评测结果: {results}")
        torch.cuda.empty_cache(); gc.collect()

    if wandb.run:
        wandb.finish()
