import os
import torch
from config import CHECK_PROMPT, REWRITE_PROMPT, INTERNAL_PROMPT, EXTERNAL_PROMPT
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import re

def parse_llm_output_rewrite(output):
    pattern = r"(.*?The rewrite query is.*?)"
    # 使用正则表达式进行匹配
    match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
    if match:
        query = match.group(1).strip()
    else:
        query = output.replace("the rewrite query is","")
    return query

class critic:
    def __init__(self, llm, nli_model, nli_tokenizer):
        self.llm = llm
        self.nli_model = nli_model
        self.nli_tokenizer = nli_tokenizer

    def check(self,question, answer):
        check_prompt = CHECK_PROMPT.format(question = question, answer=answer)
        judge = self.llm.get_output(check_prompt, system_prompt="Act as an Evaluator & Critic System.")
        try:
            judge = eval(judge)
        except:
            print(f"error in critic check!\n {judge}")
            judge = {"judgement":"correct","feedback":""}

        return judge

    def judge_internal(self, question):
        prompt = INTERNAL_PROMPT.format(question = question)

        judge = self.llm.get_output(prompt, system_prompt="Act as an Evaluator & Critic System.")
        check_list = ["sorry","couldn't","don't have access","no"]
        for keyword in check_list:
            if keyword in judge:
                return False
        return True


    def judge_external(self, question, reference):
        prompt = EXTERNAL_PROMPT.format(question = question, reference = "\n".join(reference))
        output = self.llm.get_output(prompt, system_prompt="Act as an Evaluator & Critic System.").strip().lower()
        if "no" in output:
            return False
        else:
            return True

    def feedback(self, question, reference, answer):
        basic_judge = self.check(question, answer)
        
        internal_judge = self.judge_internal(question)
        external_judge = self.judge_external(question, reference)

        basic_judge['internal_judge'] = internal_judge
        basic_judge['external_judge'] = external_judge

        return basic_judge

    def rewrite(self, question, answer, reference):        
        rewrite_prompt = REWRITE_PROMPT.format(question = question, answer = answer, reference = "\n".join(reference))
        new_query = parse_llm_output_rewrite(self.llm.get_output(rewrite_prompt))
        return new_query