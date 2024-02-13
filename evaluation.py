import os
import json
import numpy as np
import re
from collections import Counter
from utils import normalize_answer
from config import ID_ALIASES_PATH, QUERY_PATH

class Evaluator:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        if dataset_name == "2wiki":
            # pre-load for 2wiki dataset
            wid2alias = {}
            with open(ID_ALIASES_PATH, 'r') as fin:
                for l in fin:
                    l = json.loads(l)
                    wid2alias[l['Q_id']] = l['aliases']

            wiki_details = []
            with open(QUERY_PATH,"r") as f:
                for line in f:
                    wiki_details.append(json.loads(line))

            self.wiki_details = wiki_details
            self.wid2alias = wid2alias

    def get_all_alias(self, ground_truth_id):
        if ground_truth_id and ground_truth_id in self.wid2alias:
            return self.wid2alias[ground_truth_id]
        return []

    def exact_match_score(
        self,
        prediction: str,
        ground_truth: str,
        ground_truth_id: str = None
    ):  
        
        ground_truths = {ground_truth}
        if ground_truth_id:
            ground_truths.update(self.get_all_alias(ground_truth_id))
        correct = max([int(normalize_answer(prediction) == normalize_answer(gt)) for gt in ground_truths])
        
        return {'correct': correct, 'incorrect': 1 - correct}

    def f1_score(
        self,
        prediction: str,
        ground_truth: str,
        ground_truth_id: str = None
    ):
        ground_truths = {ground_truth}
        if ground_truth_id:
            ground_truths.update(self.get_all_alias(ground_truth_id))

        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        for ground_truth in ground_truths:
            normalized_prediction = normalize_answer(prediction)
            normalized_ground_truth = normalize_answer(ground_truth)
        
            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric


    def evaluate_item(self, question, pred, answer):
        answer_id = None
        if self.dataset_name == "2wiki":
            for i in self.wiki_details:
                if question == i['text']:
                    answer_id = i['metadata']['answer_id']
                    break        

        f1_metric = self.f1_score(pred, answer, ground_truth_id=answer_id)
        exact_metric = self.exact_match_score(pred, answer, ground_truth_id=answer_id)

        return {"em": exact_metric['correct'],
                "f1": f1_metric['f1'],
                "precision": f1_metric['precision'],
                "recall": f1_metric['recall']}


