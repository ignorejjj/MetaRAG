from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
import torch
from fastchat.model import load_model
import torch.nn.functional as F
from config import EXTERNAL_PROMPT, EXPERT_MODEL_PATH, SIMILARITY_MODEL_PATH

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

@torch.no_grad()
def get_llama_answer(model, tokenizer, instruction_qa_list, max_length=512):
    batch_size = 4
    responses = []
    for idx in range(0, len(instruction_qa_list), batch_size):
        batched_prompts = instruction_qa_list[idx:idx+batch_size]
        inputs = tokenizer(batched_prompts, return_tensors="pt", padding=True).to(model.device)
        outputs = model.generate(
            **inputs,
            do_sample=False,
            temperature=1.0,
            max_new_tokens=max_length
        )
        for i, generated_sequence in enumerate(outputs):
            input_ids = inputs['input_ids'][i]
            text = tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if input_ids is None:
                prompt_length = 0
            else:
                prompt_length = len(
                    tokenizer.decode(
                        input_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                )
            new_text = text[prompt_length:]
            responses.append(new_text.strip())
    
    return responses

class monitor:
    def __init__(self, expert_model_name):
        self.device = torch.device("cuda:1")
        self.expert_model_name = expert_model_name
        model_path = EXPERT_MODEL_PATH[expert_model_name]

        if expert_model_name == "span-bert":
            self.generator_model = AutoModelForQuestionAnswering.from_pretrained(model_path, torch_dtype=torch.bfloat16)
            self.generator_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            self.generator_model = self.generator_model.to(self.device)

        elif expert_model_name == "t5":
            self.generator_model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
            self.generator_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            self.generator_model = self.generator_model.to(self.device)

        elif expert_model_name == "llama2":
            self.generator_model, self.generator_tokenizer = load_model(model_path,
                                                                        device='cuda', 
                                                                        num_gpus=1,
                                                                        max_gpu_memory='30GiB',
                                                                        load_8bit=False,
                                                                        cpu_offloading=False,
                                                                        debug=False,)
            self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
            self.generator_tokenizer.padding_side = "left"

        elif expert_model_name == "chatglm2":
            self.generator_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.generator_model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
            self.generator_model = self.generator_model.eval()

        else:
            assert False, "Not implemented"

        self.sim_tokenizer = AutoTokenizer.from_pretrained(SIMILARITY_MODEL_PATH)
        self.sim_model = AutoModel.from_pretrained(SIMILARITY_MODEL_PATH)
        self.sim_model = self.sim_model.to(self.device)

    def get_pseduo_answer_llama(self, question, reference):

        prompt = f"Current Question: {question}\nSearch results: {reference}\nCurrent Answer: "
        output = get_llama_answer(self.generator_model, self.generator_tokenizer, [prompt], max_length=512)[0]
        return output
    
    def get_pseduo_answer_bert(self, question, reference):

        inputs = self.generator_tokenizer(question ,reference, return_tensors="pt",max_length=512).to(self.device)
        with torch.no_grad():
            output = self.generator_model(**inputs)
        answer_start_index = output.start_logits.argmax()
        answer_end_index = output.end_logits.argmax()
        predict_answer_tokens = inputs.input_ids[0, answer_start_index:answer_end_index+1]
        answer = self.generator_tokenizer.decode(predict_answer_tokens,skip_special_tokens=True)
        return answer

    def get_pseduo_answer_t5(self, question, reference):

        # make input
        context = "Current Question: "
        context += question
        context += "\nSearch results:"
        all_contexts = [" ".join(context) for context in reference]
        for i, search_result in enumerate(all_contexts):
            context += "\n[%s]: " % (i+1)
        context += "\nCurrent Answer: "
        input_text = context
       
        input_ids = self.generator_tokenizer(input_text , return_tensors="pt",max_length=512,truncation=True).input_ids.to(self.device)
        with torch.inference_mode():
            outputs = self.generator_model.generate(input_ids, max_new_tokens=16)
        result = self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result

    def get_pseduo_answer_chatglm(self, question, reference):

        prompt = f"Current Question: {question}\nSearch results: {reference}\nCurrent Answer: "
        response, _ = self.generator_model.chat(self.generator_tokenizer, prompt, history=[])
        return response.strip()[:64]
    
    def judge(self, question, reference, answer):

        if self.expert_model_name == "t5":
            pseduo_answer = self.get_pseduo_answer_t5(question, reference)
        elif self.expert_model_name == "span-bert":
            pseduo_answer = self.get_pseduo_answer_bert(question,reference)
        elif self.expert_model_name == "llama2":
            pseduo_answer = self.get_pseduo_answer_llama(question, reference)
        elif self.expert_model_name == "chatglm2":
            pseduo_answer = self.get_pseduo_answer_chatglm(question, reference)
        else:
            assert False


        encoded_input = self.sim_tokenizer(
                                [pseduo_answer,answer], 
                                padding=True, 
                                truncation=True, 
                                return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            model_output = self.sim_model(**encoded_input)

        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        sim_score = sentence_embeddings[0,:] @ sentence_embeddings[1,:]
        sim_score = sim_score.item()

        output = {"pseduo_answer": pseduo_answer,
                  "score": sim_score}
        return output
