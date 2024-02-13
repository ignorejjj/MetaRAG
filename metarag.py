import os
import json
from tqdm import tqdm
import datetime
import torch
import os
import json
import numpy as np
import re
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from utils import format_ref, check_answer
from evaluation import Evaluator
from generator import generator
from critic import critic
from monitor import monitor
from WikiSearcher import WikiSearcher
from llms import LLM
from config import DATASET2PATH, NLI_MODEL_PATH


class MetaRAG:
    def __init__(
        self, 
        llm_name:str, 
        dataset_name:str,
        save_dir:str,
        max_iter:int = 3,
        ref_num:int = 5,
        threshold:float = 0.3,
        expert_model:str = "t5",
        do_eval:bool = True,
        use_sample_num:int = 50
    ):
        
        self.llm_name = llm_name
        self.dataset_name = dataset_name
        self.max_iter = max_iter
        self.ref_num = ref_num
        self.threshold = threshold
        self.do_eval = do_eval

        self.evaluator = Evaluator(dataset_name)

        # preprocess save path 
        exp_name = f"{dataset_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
        output_dir = os.path.join(save_dir, exp_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "output.json")
        self.output_path = output_path
        self.output_dir = output_dir

        # load dataset
        dataset_path = DATASET2PATH[dataset_name]
        with open(dataset_path,"r",encoding="utf-8") as f:
            dataset = json.load(f)
        dataset = dataset[:use_sample_num]

        self.dataset = [{"question": item['question'], "answer": item['answer']}  for item in dataset]
       
        # load models 
        print("Loading LLM...")
        self.llm = LLM(llm_name)
        print("Loading NLI model...")
        self.nli_model = AutoModelForSeq2SeqLM.from_pretrained(NLI_MODEL_PATH, torch_dtype=torch.bfloat16).to("cuda:0")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH, use_fast=False)

        self.retriever = WikiSearcher()
        print("Loading monitor...")
        self.monitor = monitor(expert_model_name = expert_model)
        print("Loading generator...")
        self.generator = generator(llm = self.llm) 
        print("Loading critic...")
        self.critic = critic(llm = self.llm, nli_model=self.nli_model, nli_tokenizer=self.nli_tokenizer)

    def add_new_reference(self, question:str, reference:list ,single_log:dict, answer:str):
        rewrite_query = self.critic.rewrite(question, answer, reference)
        rewrite_query = rewrite_query[:64]
        new_reference = self.retriever.search(rewrite_query,100, self.ref_num)

        single_log['add_reference'] = new_reference

        reference = reference + new_reference
        reference = list(set(reference))

        return reference, single_log
    

    def predict(self, question:str):
        print(f"question: {question}")
        output_item = {'question': question}
        logs = []

        reference = self.retriever.search(question, 100, self.ref_num)
        output_item['reference'] = reference
        reason, answer = self.generator.answer(question, format_ref(reference), suggestion=None, hint=None)
        print(reason,answer)
        final_answer = answer

        while True:
            print(f"---iter: {len(logs) + 1}---")
            single_log = {"original_output": {"reason":reason,"answer":answer}}

            # monitor_result: {"pseduo_answer":...,"score":score from 0 to 1}
            monitor_result = self.monitor.judge(question, format_ref(reference), answer)
            single_log['monitor_result'] = monitor_result
            monitor_judge = True if monitor_result['score'] > self.threshold else False
            if monitor_judge or len(logs) >= self.max_iter:
                final_answer = answer
                logs.append(single_log)
                break

            # critic_result: {"internal_judgement":..., "external_judgement":..., "judgement":...,"feedback":/.}
            critic_result = self.critic.feedback(question, format_ref(reference), answer)
            if critic_result['judgement'] == "correct":
                hint = "Please think step by step."
            else:
                hint = critic_result['feedback']
            
            if critic_result['internal_judge']:
                if critic_result['external_judge']:
                    suggestion = "Carefully check your final answer. Do not add irrelevant content and answer questions accurately."
                    reason_support = self.judge_support(reason,answer)
                    critic_result['reason_support'] = reason_support
                    if not reason_support:
                        suggestion = "Please try to provide an answer by considering multi-step reasoning."
                else: 
                    suggestion = "If you feel that there is no suitable content in the reference, try relying on yourself to answer the question."                    # 添加新的reference
                    reference, single_log = self.add_new_reference(question, reference, single_log, answer)
                    output_item['reference'] = reference
            else:
                if critic_result['external_judge']:
                    suggestion = "You need to answer questions entirely based on reference, and do not use your own knowledge."
                else:
                    suggestion = "You can break down the question into sub questions to use reference."
                    reference, single_log = self.add_new_reference(question, reference, single_log, answer)
                    output_item['reference'] = reference
            
            new_reason, new_answer = self.generator.answer(question, format_ref(reference), suggestion, hint)
            if check_answer(new_answer):
                final_answer = new_answer
            single_log['critic_result'] = critic_result
            single_log['new_answer'] = {"reason":new_reason,"answer":new_answer}
            
            logs.append(single_log)
            reason,answer = new_reason,new_answer
            
        output_item['final_answer'] = final_answer
        output_item['interaction_log'] = logs

        return output_item

    def judge_support(self, reason: str, answer: str) -> bool:
        device = self.nli_model.device
        input_text = f"premise: {reason} hypothesis: {answer}"
        input_ids = self.nli_tokenizer(input_text, return_tensors="pt", max_length=512).input_ids.to(device)

        with torch.no_grad():
            outputs = self.nli_model.generate(input_ids, max_new_tokens=10)

        result = self.nli_tokenizer.decode(outputs[0], skip_special_tokens=True)
        inference = result == "1"
        
        return inference
    
    def run(self):
        output = []
        for item in tqdm(self.dataset):
            question = item['question']
            output_item = self.predict(question)
            output_item['answer'] = item['answer']

            output.append(output_item)
            
        if self.do_eval:
            output = self.evaluate(output)
        
        with open(self.output_path,"w",encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        return output
    
    def evaluate(self, output_list:list):
        all_f1, all_pre, all_rec, all_em = [], [], [], []

        for item in output_list:
            question = item['question']
            answer = item['answer']
            pred = item['final_answer']

            eval_result = self.evaluator.evaluate_item(question, pred, answer)
            item['eval_result'] = eval_result

            all_em.append(eval_result['em'])
            all_f1.append(eval_result['f1'])
            all_pre.append(eval_result['precision'])
            all_rec.append(eval_result['recall'])

        print(f"Exact acc: {100*np.mean(all_em)}")
        print(f"F1: {100*np.mean(all_f1)}")
        print(f"Precision: {100*np.mean(all_pre)}")
        print(f"Recall: {100*np.mean(all_rec)}")
    
        return output_list


    


        
        
        