# Metacognitive Retrieval-Augmented Large Language Models (MetaRAG)

This repository contains the code for the paper:
[Metacognitive Retrieval-Augmented Large Language Models](url)

## Quick start

### Install environment

Install all required libraries by running
```bash
pip install -r requirements.txt
```

### Download Wikipedia index

Our retrieval system is built on pre-built index in [pyserini library](https://github.com/castorini/pyserini), and the Wikipedia index will automatically download on the first run. If you want to use your own built index, you can modify the ```INDEX_PATH``` in ```config.py```.

### Setup Openai key

Replace ```"your API key"``` in ```config.py``` with your own OpenAI API key Please pay attention to the usage of APIs.

### Run

You need to put the data file into ```./data``` folder and modify the file path in ```config.py```. The dataset format is as follows:
```json
[
    {
        "question": "Who is the mother of the director of film Polish-Russian War (Film)?",
        "answer": "Magorzata Braunek"
    },
    {
        "question": "Which film came out first, Blind Shaft or The Mask Of Fu Manchu?",
        "answer": "The Mask Of Fu Manchu"
    }
]
```

Then use the following command to run MetaRAG in 2WikiMultihopQA dataset.
```bash
python main.py \
    --llm_name "gpt-3.5-turbo" \
    --dataset_name "2wiki" \
    --save_dir "./output/" \
    --max_iter 3 \
    --ref_num 3 \
    --threshold 0.3 \
    --expert_model "span-bert" \
    --do_eval \
    --use_sample_num 5
```

## Citation

If you find our paper or code useful, please cite the paper

```
@misc{zhou2024metacognitive,
      title={Metacognitive Retrieval-Augmented Large Language Models}, 
      author={Yujia Zhou and Zheng Liu and Jiajie Jin and Jian-Yun Nie and Zhicheng Dou},
      year={2024},
      eprint={2402.11626},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


