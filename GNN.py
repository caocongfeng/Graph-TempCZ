import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score, recall_score
import tqdm
import wandb
import logging
import datetime

# 设置环境变量以避免分词器并行化警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 配置日志记录
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)


def load_and_prepare_data(data_path):
    logging.info("从CSV加载数据...")
    data_df = pd.read_csv(data_path, engine="c")

    paper = data_df[['pmcid', 'title', 'abstract', 'pubdate']].drop_duplicates(subset=['pmcid']).reset_index(drop=True)
    software_mentions = data_df[['ID', 'aggtext']].drop_duplicates()
    software_text = software_mentions.groupby('ID')['aggtext'].apply(lambda texts: ', '.join(texts.astype(str))).reset_index()
    software_text = software_text.rename(columns={'aggtext': 'text'})

    logging.info(f"加载了 {len(paper)} 篇唯一论文和 {len(software_text)} 个唯一软件实体。")
    return data_df, paper, software_text


def compute_embeddings(model_name, paper, software_text, device):
    logging.info(f"初始化SentenceTransformer模型: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    model.tokenizer.model_max_length = 128

    logging.info("正在为软件计算嵌入...")
    sw_emb = model.encode(
        software_text['text'].tolist(),
        batch_size=128,
        show_progress_bar=True,
        device=device,
        convert_to_tensor=True
    )
    software_text['embedding'] = sw_emb.cpu().numpy().tolist()

    logging.info("正在为论文计算嵌入...")
    p_texts = (paper['title'].fillna('') + ' ' + paper['abstract'].fillna('')).tolist()
    p_emb = model.encode(
        p_texts,
        batch_size=128,
        show_progress_bar=True,
        device=device,
        convert_to_tensor=True
    )
    paper['embedding'] = p_emb.cpu().numpy().tolist()

    emb_dim = sw_emb.shape[1]
    logging.info(f"嵌入计算完成，维度为: {emb_dim}")
    return paper, software_text, emb_dim


def build_heterodata(data_df, paper, software_text):
    paper_id_map = paper[['pmcid']].drop_duplicates().reset_index().rename(columns={'index': 'mappedID'})
    software_id_map = software_text[['ID']].drop_duplicates().reset_index().rename(columns={'index': 'mappedID'})

    feature_paper = paper.merge(paper_id_map, on='pmcid')
    feature_software = software_text.merge(software_id_map, on='ID')

    paper_x = torch.tensor(np.vstack(feature_paper['embedding'].values), dtype=torch.float)
    software_x = torch.tensor(np.vstack(feature_software['embedding'].values), dtype=torch.float)

    edges = data_df[['pmcid', 'ID']].drop_duplicates()
    edges = edges.merge(paper_id_map, on='pmcid').merge(software_id_map, on='ID')
    edge_index = torch.tensor([
        edges['mappedID_x'].values,
        edges['mappedID_y'].values
    ], dtype=torch.long)

    data = HeteroData()
    data['paper'].x = paper_x
    data['paper'].node_id = torch.tensor(feature_paper['mappedID'].values, dtype=torch.long)
    data['software'].x = software_x
    data['software'].node_id = torch.tensor(feature_software['mappedID'].values, dtype=torch.long)
    data['paper','mention','software'].edge_index = edge_index

    data = T.ToUndirected()(data)
    return data


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


class GNN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_dim)
        self.conv2 = SAGEConv((-1, -1), hidden_dim)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


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
        self.gnn = to_hetero(GNN(hidden_dim), metadata, aggr='sum')
        self.classifier = Classifier(hidden_dim)

    def forward(self, data):
        x_p = self.dropout(self.paper_lin(data['paper'].x)) + self.paper_emb(data['paper'].node_id)
        x_s = self.dropout(self.software_lin(data['software'].x)) + self.software_emb(data['software'].node_id)
        x_dict = {'paper': x_p, 'software': x_s}
        x_out = self.gnn(x_dict, data.edge_index_dict)
        edge_idx = data['paper','mention','software'].edge_label_index
        return self.classifier(x_out['paper'], x_out['software'], edge_idx)


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
    pred = torch.cat(preds)
    truth = torch.cat(truths)
    prob = torch.sigmoid(pred)
    pred_label = (prob>0.5).long()
    return {
        'acc': accuracy_score(truth.numpy(), pred_label.numpy()),
        'f1': f1_score(truth.numpy(), pred_label.numpy()),
        'recall': recall_score(truth.numpy(), pred_label.numpy()),
        'auc': roc_auc_score(truth.numpy(), prob.numpy()),
        'ap': average_precision_score(truth.numpy(), prob.numpy()),
        'loss': F.binary_cross_entropy_with_logits(pred, truth).item()
    }


def get_loader(subset, batch_size, shuffle=True):
    return LinkNeighborLoader(
        data=subset,
        num_neighbors=[20, 10],
        edge_label_index=(('paper','mention','software'), subset['paper','mention','software'].edge_label_index),
        edge_label=subset['paper','mention','software'].edge_label,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
    )

def print_label_distribution(loader, name=""):
    """检查加载器中正/负样本平衡的实用程序。"""
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


def predict_and_save_results(model, loader, device, data_df, paper, software_text, output_filename="test_predictions"):
    """
    在测试集上运行预测并将结果保存到CSV文件。
    
    Args:
        model (torch.nn.Module): 训练好的模型。
        loader (LinkNeighborLoader): 测试集的数据加载器。
        device (torch.device): 运行模型的设备。
        data_df (pd.DataFrame): 用于合并的原始完整数据框。
        paper (pd.DataFrame): 带有原始ID的论文数据框。
        software_text (pd.DataFrame): 带有原始ID的软件数据框。
        output_filename (str): 输出CSV文件的名称。
    """
    logging.info(f"开始在测试集上进行预测以保存结果...")
    model.eval()
    
    all_edge_indices = []
    all_preds = []
    all_truths = []

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="在测试集上预测"):
            batch = batch.to(device)
            out = model(batch)
            
            all_preds.append(out.cpu())
            all_truths.append(batch["paper", "mention", "software"].edge_label.cpu())
            all_edge_indices.append(batch["paper", "mention", "software"].edge_label_index.cpu())

    # 连接所有批次的结果
    preds_tensor = torch.cat(all_preds)
    truths_tensor = torch.cat(all_truths)
    edge_indices_tensor = torch.cat(all_edge_indices, dim=1)

    # 将预测转换为概率和标签
    probs = torch.sigmoid(preds_tensor)
    pred_labels = (probs > 0.5).long()

    # 创建反向ID映射
    paper_id_map = paper[['pmcid']].drop_duplicates().reset_index().rename(columns={'index': 'mappedID'})
    software_id_map = software_text[['ID']].drop_duplicates().reset_index().rename(columns={'index': 'mappedID'})
    
    paper_id_rev = paper_id_map.set_index('mappedID')['pmcid'].to_dict()
    software_id_rev = software_id_map.set_index('mappedID')['ID'].to_dict()

    # 将mappedID映射回原始ID
    paper_indices = edge_indices_tensor[0].numpy()
    software_indices = edge_indices_tensor[1].numpy()

    pmcids = [paper_id_rev.get(i) for i in paper_indices]
    software_ids = [software_id_rev.get(i) for i in software_indices]

    # 创建结果数据框
    results_df = pd.DataFrame({
        'pmcid': pmcids,
        'ID': software_ids,
        'ground_truth': truths_tensor.numpy(),
        'predicted_label': pred_labels.numpy(),
        'predicted_prob': probs.numpy()
    })

    # 删除映射失败的行（如果有）
    results_df.dropna(subset=['pmcid', 'ID'], inplace=True)

    # 与原始数据合并以获得完整上下文
    # 在原始df上使用drop_duplicates以避免创建额外的行
    original_unique_edges = data_df[['pmcid', 'ID', 'text', 'title', 'abstract']].drop_duplicates(subset=['pmcid', 'ID'])
    
    final_df = pd.merge(results_df, original_unique_edges, on=['pmcid', 'ID'], how='left')

    # 保存到CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_df.to_csv(f"{output_filename}_{timestamp}.csv", index=False)
    logging.info(f"测试预测结果已保存到 {output_filename}_{timestamp}.csv")


if __name__ == '__main__':
    DATA_PATH = '../datasets/single_graph_merged_data_label.csv'
    EMB_MODEL = 'all-MiniLM-L6-v2'
    BATCH_SIZE = 256
    HIDDEN_DIM = 128
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    NEG_SAMPLING_RATIO = 2.0

    NUM_WORKERS = min(8, os.cpu_count())
    config = {
        'lr': LEARNING_RATE,
        'hidden_dim': HIDDEN_DIM,
        'batch_size': BATCH_SIZE,
        'model': EMB_MODEL,
        'neg_sampling_ratio': NEG_SAMPLING_RATIO,
        'epochs': NUM_EPOCHS
    }
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    BEST_MODEL_PATH = f"best_model_{timestamp}.pt"

    try:
        wandb.login()
        wandb.init(project="HeteroGNN_Link_Prediction", entity="byfrfy", config=config)
    except Exception as e:
        logging.warning(f"无法初始化wandb: {e}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    data_df, paper, software_text = load_and_prepare_data(DATA_PATH)
    paper, software_text, emb_dim = compute_embeddings(EMB_MODEL, paper, software_text, device)
    data = build_heterodata(data_df, paper, software_text)

    data = T.ToUndirected()(data)
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
        edge_types=[('paper','mention','software')],
        neg_sampling_ratio=NEG_SAMPLING_RATIO
    )
    train_data, val_data, test_data = transform(data)

    for split, split_name in zip([train_data, val_data, test_data], ['train', 'val', 'test']):
        unique_vals = split["paper", "mention", "software"].edge_label.unique()
        print(f"{split_name} label values: {unique_vals.tolist()}")

    train_loader = get_loader(train_data, BATCH_SIZE, shuffle=True)
    val_loader   = get_loader(val_data,   BATCH_SIZE*2, shuffle=False)
    test_loader  = get_loader(test_data,  BATCH_SIZE*2, shuffle=False)

    print_label_distribution(train_loader, "Train")
    print_label_distribution(val_loader, "Val")
    print_label_distribution(test_loader, "Test")

    model = Model(
        emb_dim=emb_dim,
        hidden_dim=HIDDEN_DIM,
        metadata=data.metadata(),
        num_papers=data['paper'].num_nodes,
        num_software=data['software'].num_nodes
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = FocalLoss(alpha=1.0, gamma=2.0)

    best_val_loss = float('inf')
    patience = 0
    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        total_loss, total_count = 0, 0
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch:02d}")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            labels = batch['paper','mention','software'].edge_label.float()
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * out.numel()
            total_count += out.numel()
            pbar.set_postfix({'loss': total_loss / total_count})
        avg_loss = total_loss / total_count

        val_metrics = evaluate(model, val_loader, device)
        logging.info(
            f"Epoch {epoch:03d} | 训练损失: {avg_loss:.4f} | 验证 AUC: {val_metrics['auc']:.4f}, "
            f"验证 AP: {val_metrics['ap']:.4f}, 验证 F1: {val_metrics['f1']:.4f}, 验证 Recall: {val_metrics['recall']:.4f}" 
        )

        if wandb.run:
            wandb.log({
                "epoch": epoch, "train_loss": avg_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()}
            })

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            patience = 0
        else:
            patience += 1
            if patience > 30:
                logging.info("Early stopping triggered")
                break

    if best_val_loss < float('inf'):
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        test_metrics = evaluate(model, test_loader, device)
        logging.info(f"测试结果 --> AUC: {test_metrics['auc']:.4f}, AP: {test_metrics['ap']:.4f}, F1: {test_metrics['f1']:.4f}, Recall: {test_metrics['recall']:.4f}")
        if wandb.run:
            wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        predict_and_save_results(model, test_loader, device, data_df, paper, software_text)
    else:
        logging.warning("没有保存最佳模型，跳过最终测试。")

    if wandb.run:
        wandb.finish()
