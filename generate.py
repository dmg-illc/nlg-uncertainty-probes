import json
import argparse
from datetime import datetime
from pathlib import Path

import torch.cuda
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    FSMTForConditionalGeneration,
    FSMTTokenizer,
    T5Tokenizer,
)

from utils import set_seeds, load_jsonl, load_wmt14_data


def main(args):
    print(args)
    print("Initializing model and loading data...")
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    print(device)
    if args.task == "translation":
        try:
            # FairSeq Machine Translation
            tokenizer = FSMTTokenizer.from_pretrained(args.model_name)
            model = FSMTForConditionalGeneration.from_pretrained(args.model_name).to(device)
        except ValueError:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)
    elif args.task in ["dialogue", "story_generation"]:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    elif args.task == "simplification":
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name, device_map="auto")
    else:
        raise ValueError("Invalid task")

    if args.task == "translation" and args.data_path.endswith(".extr_refs"):
        data = load_wmt14_data(args.data_path)
    else:
        data = load_jsonl(args.data_path)
    out_path = Path(args.out_path)

    print("Generating responses...")
    responses = list()

    for datum in tqdm(data[: args.debug_instances]):
        if args.task == "dialogue":
            context = "".join(
                [utterance + tokenizer.eos_token for utterance in datum[args.context_key]]
            )
        elif args.task == "simplification":
            context = f"Simplify: {datum[args.context_key]}"
        elif args.task in ["translation", "story_generation"]:
            context = datum[args.context_key]

        context_ids = tokenizer.encode(context, return_tensors="pt").to(device)
        response_ids = model.generate(
            context_ids,
            max_new_tokens=args.max_length,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=bool(args.do_sample),
            top_k=args.top_k,
            top_p=args.top_p,
            typical_p=args.typical_p,
            num_beams=args.n_beams,
            temperature=args.temperature,
            num_return_sequences=args.n_samples,
        )
        # some models return input_output, others just output
        response_ids = (
            response_ids[:, context_ids.shape[-1] :]
            if args.task in ["dialogue", "story_generation"]
            else response_ids
        )
        decoded_responses = [
            tokenizer.decode(response, skip_special_tokens=True) for response in response_ids
        ]
        responses.append({**datum, "generated_responses": decoded_responses})

    print("Writing responses to file...")
    write_responses(out_path, responses)
    write_params(args)


def write_params(args):
    with open(Path(args.out_path).with_suffix(".info"), "w") as f:
        json.dump({**vars(args), "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S")}, f)


def write_responses(out_path, responses):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for response in responses:
            json.dump(response, f)
            f.write("\n")


if __name__ == "__main__":
    set_seeds(0)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs_path",
        type=str,
        help="Path to jsonl file with one config json per line.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to json file with multiple references.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        help="Path to output txt file containing one response per line.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the huggingface model to be used.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether the decoding algorithm requires sampling.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="The number of samples to generate given one context.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="The maximum number tokens of a response.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="The top k value for sampling. Defaults to 50.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="If set to float < 1, only the most probable tokens with "
        "probabilities that add up to top_p or higher are kept for "
        "generation. Defaults to 1.0.",
    )
    parser.add_argument(
        "--typical_p",
        type=float,
        default=None,
        help="The amount of probability mass from the original distribution "
        "to be considered in typical decoding. If set to 1.0 (default) "
        "it takes no effect. See "
        "[this paper](https://arxiv.org/pdf/2202.00666.pdf) for more details.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="The value used to module the next token probabilities. " "Defaults to 1.0.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=None,
        help="The number of beams for beam search. Defaults to None (sampling).",
    )
    parser.add_argument(
        "--debug_instances",
        type=int,
        default=int(10e10),
        help="For test runs with only a few instances",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="The task to be performed: 'dialogue', 'simplification', or 'story_generation'",
    )
    parser.add_argument(
        "--context_key",
        type=str,
        default=None,
        help="The key of the context in the data file.",
    )

    args = parser.parse_args()
    if args.configs_path:
        configs = load_jsonl(args.configs_path, return_obj=True)
        for config in configs:
            config.debug_instances = args.debug_instances
            main(config)
    else:
        main(args)
