from datasets import load_dataset
import json
import os
import torch
import numpy as np
import pandas as pd


def _load_hf_dataset_flexible(dataset_name):
    if "/" in dataset_name:
        first, second = dataset_name.split("/", 1)
        if first in {"glue", "super_glue"}:
            return load_dataset(first, second)
    return load_dataset(dataset_name)


def _pick_hf_split(data_hug):
    for split_name in ("validation", "test", "train"):
        if split_name in data_hug:
            return data_hug[split_name]
    return data_hug[list(data_hug.keys())[0]]


def _example_to_text(item, column_names, dataset_name):
    if dataset_name == "glue/qqp" or {"question1", "question2"}.issubset(column_names):
        return f"{item['question1']} [SEP] {item['question2']}"

    if {"sentence1", "sentence2"}.issubset(column_names):
        return f"{item['sentence1']} [SEP] {item['sentence2']}"

    for text_col in ("sentence", "text", "review", "content", "question", "title"):
        if text_col in column_names:
            return str(item[text_col])

    if "output" in column_names:
        return str(item["output"])

    for col in column_names:
        if isinstance(item[col], str):
            return str(item[col])

    raise ValueError(
        f"Cannot infer text columns for dataset {dataset_name}. "
        f"Available columns: {list(column_names)}"
    )

class Dataset(object):
    def __init__(self, dataset_name, dataset_type):
        if dataset_type == "local":
            if not os.path.exists(dataset_name):
                raise FileNotFoundError("{} dataset does not exist!!".format(dataset_name))
            if "ECHR" in dataset_name:
                self.data = []
                for root, dirs, files in os.walk(dataset_name):
                    files.sort()
                    for i, file in enumerate(files):
                        with open(os.path.join(dataset_name, file), 'r') as f:
                            d = json.load(f)
                        self.data.append(d["CONCLUSION"])
            else:
                f = open(dataset_name, 'r')
                self.data = json.load(f)
                f.close()

        elif dataset_type == "datasets":
            self.data = []
            self.labels = []

            data_hug = _load_hf_dataset_flexible(dataset_name)
            split = _pick_hf_split(data_hug)
            print("dataset size: {}".format(len(split)))
            column_names = set(split.column_names)
            has_label = "label" in column_names

            for item in split:
                self.data.append(_example_to_text(item, column_names, dataset_name))
                if has_label:
                    label = item["label"]
                    if label is not None and int(label) >= 0:
                        self.labels.append(int(label))

            if len(self.labels) != len(self.data):
                self.labels = None

        elif dataset_type == "github":
            if "skytrax-reviews-dataset" in dataset_name:
                if not os.path.exists(dataset_name):
                    raise FileNotFoundError("{} dataset does not exist!!".format(dataset_name))
                df = pd.read_csv(dataset_name)
                self.data = df['content'].tolist()
                self.labels = df['recommended'].tolist()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def get_data(self):
        return self.data

    def get_labels(self):
        return getattr(self, 'labels', None)
