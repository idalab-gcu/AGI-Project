import re
import csv
import pandas as pd
from tqdm import tqdm
from radgraph import RadGraph
from collections import Counter

file_path = '' # input your mimic_cxr reports file path
output_node_file = 'extract_nodes.csv'
output_rel_file = 'extract_relations.csv'

rad_graph = RadGraph()

def read_csv_file(file_path):
    texts = []
    try:
        df = pd.read_csv(file_path)
        for item in df.iterrows():
            texts.append(item[1].__getitem__('text'))
        return texts
    except FileNotFoundError:
        print(f"File Not Found: {file_path}")
    except Exception as e:
        print(f"Exception: {e}")


def extract_annotations(text):
    sentences = re.split(r'[.;]\s*', text)
    annotations = rad_graph(sentences)
    return annotations


def set_triplet_csv(annotations, node_writer, rel_writer, node_set,
                    node_counter, rel_counter):
    """
    node_set: 이미 저장한 노드를 중복 방지용으로 관리
    """
    for key, val in annotations.items():
        entities = val['entities']
        for idx, val2 in entities.items():
            n1_tokens = val2["tokens"].lower()
            n1_label = val2["label"].replace("::", ":").replace(" ", "_")
            # 노드 저장
            node_key1 = (n1_tokens, n1_label)
            if node_key1 not in node_set:
                node_writer.writerow({"name": n1_tokens, "label": n1_label})
                node_set.add(node_key1)
            node_counter.update([n1_tokens])

            for rel in val2['relations']:
                rel_idx = rel[1]
                n2_tokens = entities[rel_idx]["tokens"].lower()
                n2_label = entities[rel_idx]["label"].replace("::", ":").replace(" ", "_")
                node_key2 = (n2_tokens, n2_label)
                if node_key2 not in node_set:
                    node_writer.writerow({"name": n2_tokens, "label": n2_label})
                    node_set.add(node_key2)
                node_counter.update([n2_tokens])

                # 관계 저장
                rel_writer.writerow({
                    "src": n1_tokens,
                    "src_label": n1_label,
                    "dst": n2_tokens,
                    "dst_label": n2_label,
                    "rel": rel[0]
                })
                rel_counter.update([rel[0]])


if __name__ == '__main__':
    texts = read_csv_file(file_path)

    # CSV 파일 오픈
    with open(output_node_file, "w", newline="", encoding="utf-8") as nf, \
         open(output_rel_file, "w", newline="", encoding="utf-8") as rf:
        node_writer = csv.DictWriter(nf, fieldnames=["name", "label"])
        node_writer.writeheader()
        rel_writer = csv.DictWriter(rf, fieldnames=["src", "src_label", "dst", "dst_label", "rel"])
        rel_writer.writeheader()

        node_set = set()
        node_counter = Counter()
        rel_counter = Counter()

        for text in tqdm(texts):
            annotations = extract_annotations(text)
            set_triplet_csv(annotations, node_writer, rel_writer, node_set, node_counter, rel_counter)

    print("save result csv file")