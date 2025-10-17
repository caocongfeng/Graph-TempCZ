import pandas as pd
import multiprocessing
import requests
import re
import os
import json
from tqdm import tqdm

# 参数设置
BATCH_SIZE = 36
OUTPUT_FILE = "results/pmc_data_all.jsonl"

# 读取 PMCIDs
print("[INFO] Loading PMCIDs...")
df = pd.read_table("./disambiguated/comm_disambiguated.tsv.gz")
pmcid_list = list(set(df['pmcid']))
print(f"[INFO] Total unique PMCIDs: {len(pmcid_list)}")

# 划分批次
batches = [pmcid_list[i:i+BATCH_SIZE] for i in range(0, len(pmcid_list), BATCH_SIZE)]
print(f"[INFO] Total batches: {len(batches)} (batch size = {BATCH_SIZE})")

# 单个批次处理函数（不使用 cache）
def process_batch(pmcid_batch):
    results = []
    for pmcid in pmcid_batch:
        try:
            url = f'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC{pmcid}/fullTextXML'
            response = requests.get(url, timeout=20)
            html = response.text

            title_match = re.findall(r'<title-group.*?>(.*?)</title-group>', html, re.S)
            abstract_match = re.findall(r'<abstract.*?>(.*?)</abstract>', html, re.S)

            title = title_match[0] if title_match else ""
            abstract = abstract_match[0] if abstract_match else ""

            title = re.sub(r'<.*?>', ' ', title)
            title = re.sub(r'\s+', ' ', title).strip()
            abstract = re.sub(r'<.*?>', ' ', abstract)
            abstract = re.sub(r'\s+', ' ', abstract).strip()

            results.append({
                "ID": pmcid,
                "title": title,
                "abstract": abstract,
                "fulltext": html
            })

        except Exception as e:
            print(f"[ERROR] PMCID {pmcid}: {e}")
    return results

# 主函数：并行处理所有批次并写入文件
if __name__ == '__main__':
    os.makedirs("results", exist_ok=True)
    num_workers = multiprocessing.cpu_count()
    print(f"[INFO] Using {num_workers} CPU cores")

    with multiprocessing.Pool(processes=num_workers) as pool, open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for batch_result in tqdm(pool.imap_unordered(process_batch, batches), total=len(batches)):
            for item in batch_result:
                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"[✓] Done. Output written to {OUTPUT_FILE}")
