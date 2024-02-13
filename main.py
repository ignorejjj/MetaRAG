import argparse
from metarag import MetaRAG


parser = argparse.ArgumentParser()
parser.add_argument("--llm_name", 
                    type = str, 
                    default = "gpt-3.5-turbo", 
                    choices = ["gpt-3.5-turbo", "llama2", "vacuna", "chatglm"]
)
parser.add_argument("--dataset_name", 
                    type = str, 
                    default = "2wiki",
                    choices = ['2wiki', 'hotpotqa', 'bamboogle', 'musique']
)
parser.add_argument("--save_dir", 
                    type = str, 
                    default = "./output"
)
parser.add_argument("--max_iter", 
                    type = int, 
                    default = 3
)
parser.add_argument("--ref_num", 
                    type = int, 
                    default = 5
)
parser.add_argument("--threshold", 
                    type = float, 
                    default = 0.3
)
parser.add_argument("--expert_model", 
                    type = str, 
                    default = "t5",
                    choices=['span-bert', 't5', 'llama2', 'chatglm2']
)
parser.add_argument("--do_eval",
                    action = "store_true"
)
parser.add_argument("--use_sample_num",
                    type=int,
                    default=50)

if __name__ == "__main__":
    args = parser.parse_args()
    
    env = MetaRAG(
            llm_name = args.llm_name,
            dataset_name = args.dataset_name,
            save_dir = args.save_dir,
            max_iter = args.max_iter,
            ref_num = args.ref_num,
            threshold = args.threshold,
            expert_model = args.expert_model,
            do_eval = args.do_eval,
            use_sample_num = args.use_sample_num
        )
    
    env.run()
