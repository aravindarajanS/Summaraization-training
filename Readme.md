# Summaraization training

This project is a training script to showcase the efficiency of the BART model in abstractive summaraization .



 ## Data
Samsum data used in this training,from following link

https://huggingface.co/datasets/samsum

there are other datsets given and most of the dataset are before the advancement of instruct based LLM , hence this is just to showcase the coding style and training of encoer-decoder model


## File Structure

- `src/dataset_utils.py`: to process the dataset from the hf.
- `src/trainer.py`: training component and arguments for the BART training.
- `src/main.py`: main script of the training
- `metrics/metrics.py` : Various non LLM based evaluation metrics that can be used.
 (to train the model picked rouge score. can choose any techniques from the metrics)
- `metrics/judge_llm.py` : to evaluate the summaraizarion using a LLm
- `README.md`: This file.
