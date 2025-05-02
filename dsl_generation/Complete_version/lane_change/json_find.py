import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def find_similar_descriptions(query, json_path, top_k=3):
    """
    根据传入的 query 查找与之最相似的描述，并返回对应的数据集名称和相似度。

    :param query: 查询的场景描述
    :param json_path: 存储场景描述和数据集名称的 JSON 文件路径
    :param top_k: 返回最相似的前 k 个结果，默认为 3
    :return: 一个包含最相似描述及数据集名称和相似度的字典列表
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    descriptions = [item['original_description'] for item in data]
    file_names = [item['file_name'] for item in data]
    model = SentenceTransformer("all-MiniLM-L6-v2")

    query_embedding = model.encode([query])
    descriptions_embedding = model.encode(descriptions)

    # 计算查询与每个场景描述之间的余弦相似度
    similarities = cosine_similarity(query_embedding, descriptions_embedding)

    top_indices = np.argsort(similarities[0])[::-1][:top_k]
    result = []

    for idx in top_indices:
        result.append({
            "DataSet": file_names[idx],
            "Description": descriptions[idx],
            "Similarity": similarities[0][idx]
        })

    return result