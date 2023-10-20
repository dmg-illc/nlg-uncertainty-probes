# Evaluating Uncertainty in Neural Text Generators Against Human Production Variability
This repository contains code for the paper [What Comes Next? Evaluating Uncertainty
in Neural Text Generators Against Human Production Variability](https://arxiv.org/pdf/2305.11707.pdf).

## Data
- Machine Translation: [Newstext2014](https://github.com/facebookresearch/analyzing-uncertainty-nmt) (en-de)
- Text Simplification: [ASSET](https://github.com/facebookresearch/asset)
- Story Generation: [WritingPrompts](https://github.com/facebookresearch/fairseq/blob/main/examples/stories/README.md)
- Open-ended Dialogue: [DailyDialog++](https://iitmnlp.github.io/DailyDialog-plusplus/)

## Models
- Machine Translation: [Helsinki-NLP/opus-mt-en-de](https://huggingface.co/Helsinki-NLP/opus-mt-en-de)
- Text Simplification: [Flan-T5 large](https://huggingface.co/google/flan-t5-large) fine-tuned on the ASSET training set
- Story Generation: [GPT-2 large](https://huggingface.co/gpt2-large) fine-tuned on the WritingPrompts training set
- Open-ended Dialogue: [DialoGPT medium](https://huggingface.co/microsoft/DialoGPT-medium)

## Data Pre-Processing
We provide several methods in `utils.py` to transform the WMT, ASSET, and DailyDialog++ datasets into a unified format. 
WritingPrompts can be converted using the python file `convert_writingprompts.py`, which also allows to filter data instances 
according to minimum number of story completions available for each prompt.

## Fine-tuning
For convenience and reproducibility, we provide the fine-tuning scripts we used for text simplification and 
story generation: `finetune_flanT5.py` and `finetune_gpt.py`. We recommend using up-to-date Huggingface code; see, for
example, the Huggingface [trainers documentation](https://huggingface.co/docs/transformers/training) and 
[finetuning script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py).

## Generation
You can launch the generation script by providing a configuration file in json format:
```
python src/generate.py --configs_path generation_config.json
```

This is what a configuration file looks like:
```
{
    "config_name": "FlanT5-large finetuned typical d=0.95", 
    "task": "simplification", 
    "context_key": "source",   # the key to access the context/input in the data file
    "data_path": "data/asset/val.json", 
    "out_path": "data/asset/flanT5_large_finetuned-typical_095-val.json",
    "model_name": "models/asset/flan-t5-large", 
    "do_sample": true, 
    "n_samples": 50, 
    "max_length": 100, 
    "typical_p": 0.95,
    "temperature": null, 
    "top_k": null, 
    "top_p": null, 
    "n_beams": null
}
```

Alternatively, you can pass arguments directly (e.g., `--data_path PATH --out_path PATH --model_name MODEL`). 
See all available arguments in `src/generate.py`.

## Computing similarity scores
To compute similarity scores between two systems, you can use the script `src/write_scores.py`. For example, the following configuration will output similarity scores between the outputs of one NLG system `cand_system_name`:
```
python src/write_scores.py \
    --cand_system "cand_system_name" \
    --tgt_system "cand_system_name" \
    --cand_path "/path/to/cand_system_outputs.json" \
    --tgt_path "/path/to/tgt_system_outputs.json" \
    --cand_response_key "generation" \
    --tgt_response_key "generation" \
    --context_key "input" \
    --out_path "/path/to/output/" \
    --n_cand_samples "10" \
    --n_tgt_samples "10" \
    --n_contexts "1000" \
    --dataset "asset" \
    --max_response_length "100"
```
Precommputed scores for a large variety of models and decoding algorithms are available at: [https://doi.org/10.5281/zenodo.10025272](https://doi.org/10.5281/zenodo.10025272).

## Probing Representations of Uncertainty and Assessing Statistical Fitness 
We assess the statistical fitness of the candidate system by analysing its outputs w.r.t. itself (self-variability) or a system known to be plausible (human references) for different divergence measures. This analysis is done in `fitness_analysis.ipynb`. This notebook also contains the plots included in the paper. 
