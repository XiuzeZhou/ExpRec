# ExpRec

**This repository is constructed based on GPT-2!**

## Requirements

- python>=3.9
- torch>=2.2.0
- transformers>=4.40.1

**1. Prepare the code and the environment**

```
├──data: datasets.
    ├── ClothingShoesAndJewelry
    ├── MoviesAndTV
    ├── TripAdvisor
    ├── Yelp
├──llm: llms.
    ├── gpt2-small
    ├── gpt2-medium
    ├── gpt2-large
├──logs: logs in training.
├──models: save the final model.
├──outputs: generated texts for prediction.
├──shell: .sh for running
    ├── train_mf_mlp.sh
    ├── train_recllm.sh

```

**2. Prepare the pretrained Vicuna weights**

- gpt2-small: https://huggingface.co/openai-community/gpt2
- gpt2-medium: https://huggingface.co/openai-community/gpt2-medium
- gpt2-large: https://huggingface.co/openai-community/gpt2-large

**3. Prepare the Datasets**

### Training
The training of RecLLM contains two stages:

**1. MF Embeddings**
```
sh ./shell/train_mf_mlp.sh 0
```

**2. LoRA Tuning**
```
sh ./shell/train_recllm.sh 0
```
