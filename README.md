# ExpRec

**This repository includes several methods of Explainable Recommendations (ExpRec): **

| Model  | Core              | Url                                   |
| ------ | ----------------- | ------------------------------------- |
| CER    | Transformer       | https://github.com/JMRaczynski/CER    |
| PETER  | Transformer       | https://github.com/lileipisces/PETER  |
| PEPLER | GPT-2             | https://github.com/lileipisces/PEPLER |
| RecLLM | GPT-2 and LLaMa-2 | https://github.com/XiuzeZhou/ExpRec   |

## Requirements

- python>=3.9
- numpy==1.26.4
- torch==2.2.0
- transformers==4.40.2

### **1. Prepare the code and the environment**

Git clone our repository, creating a python environment and coping it to .sh files. 

***Code Structure:***

```
├──data: datasets, download from step 3
    ├── ClothingShoesAndJewelry
    ├── MoviesAndTV
    ├── TripAdvisor
    ├── Yelp
├──llm: pretrained LLMs, download from step 2
    ├── gpt2-small
    ├── gpt2-medium
    ├── gpt2-large
    ├── llama-2
├──logs: training logs and results.
├──models: save the final model.
├──outputs: generated texts by trained models.
├──shell: .sh for running
    ├── train_mf.sh
    ├── train_recllm-gpt.sh
    ├── train_recllm-llm.sh
    ├── train_peter.sh
    ├── train_pepler.sh
    ├── train_cer.sh
```

### **2. Prepare the pretrained Vicuna weights**

- gpt2-small: https://huggingface.co/openai-community/gpt2
- gpt2-medium: https://huggingface.co/openai-community/gpt2-medium
- gpt2-large: https://huggingface.co/openai-community/gpt2-large
- llama-2: https://huggingface.co/meta-llama/Llama-2-7b

### **3. Prepare the Datasets**

Datasets to [download](https://drive.google.com/drive/folders/1yB-EFuApAOJ0RzTI0VfZ0pignytguU0_?usp=sharing)
- ClothingShoesAndJewelry: 
- MoviesAndTV: Amazon Movies & TV
- TripAdvisor: TripAdvisor Hong Kong
- Yelp: Yelp 2019

## Training
### 1. CER

```
sh ./shell/train_cer.sh 0
```

### 2. PETER

```
sh ./shell/train_peter.sh 0
```

## 3. PEPLER

```
sh ./shell/train_pepler.sh 0
```

### 4. RecLLM

The training of RecLLM contains two stages:

**1. Build user/item embeddings by MF**

```
sh ./shell/train_mf.sh 0
```

**2. LoRA tuning for all datasets**

GPT-2:

```
sh ./shell/train_recllm-gpt.sh 0
```

LLaMa-2:

```
sh ./shell/train_recllm-llama.sh 0
```

## Citations

Please cite our paper if they are helpful to your work!

```

```

