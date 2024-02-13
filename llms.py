import torch
import openai
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import json
from generator import generator
from critic import critic
from monitor import monitor
from fastchat.model import load_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModel


MAX_TRY_NUM = 5

class LLM:
    def __init__(self, llm_name:str):
        # initialize
        if llm_name == "llama2":
            model,tokenizer = load_model(LLAMA2_13B_PATH,'cuda', 1)
        elif llm_name == "vacuna":
            model,tokenizer = load_model(VICUNA_PATH,'cuda', 1)
        elif llm_name == "chatglm":
            model_path = CHATGLM2_PATH
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
            model = model.eval()
        elif llm_name == "gpt-3.5-turbo":
            model,tokenizer = None,None
        
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = llm_name

    def get_output(self, prompt, system_prompt="Act as an Evaluator & Critic System."):
        if self.model_name == 'gpt-3.5-turbo':
            return self.get_gpt_output(prompt, system_prompt)
        elif self.model_name == "llama2" or self.model_name == "vacuna":
            return self.get_llama_output([prompt], max_length=256)[0]
        elif self.model_name == "chatglm":
            return self.get_glm_output(prompt)
        else:
            assert False, "Not implemented"

    def get_gpt_output(self, prompt, system_prompt):
        try_num = 0

        while try_num < MAX_TRY_NUM:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                output = response['choices'][0]['message']['content']
                return output
            
            except Exception as e:
                print(e)
                try_num += 1

        print("Error in generator! Too many errors!")
        return '{"reason":"Unknown","answer":"Unknown"}'

    @torch.no_grad()
    def get_glm_output(self, prompt):
        response, history = self.model.chat(self.tokenizer, prompt, history=[])
        return response.strip()

    @torch.no_grad()
    def get_llama_output(self, instruction_qa_list, max_length=512):
        batch_size = 4
        responses = []
        for idx in range(0, len(instruction_qa_list), batch_size):
            batched_prompts = instruction_qa_list[idx:idx+batch_size]
            inputs = self.tokenizer(batched_prompts, return_tensors="pt", padding=True).to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=1.0,
                max_new_tokens=max_length
            )
            for i, generated_sequence in enumerate(outputs):
                input_ids = inputs['input_ids'][i]
                text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if input_ids is None:
                    prompt_length = 0
                else:
                    prompt_length = len(
                        self.tokenizer.decode(
                            input_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                        )
                    )
                new_text = text[prompt_length:]
                responses.append(new_text.strip())
        
        return responses