import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Any

import json
import csv
from pprint import pprint
from tqdm import tqdm

from scorer import Scorer
from utils import load_jsonl, NpEncoder


def main(args):
    print("Configuration...")
    pprint(vars(args))

    print(f"Loading data...")
    cand_data = load_jsonl(args.cand_path)
    tgt_data = load_jsonl(args.tgt_path)
    is_same_system = args.cand_path == args.tgt_path

    print("Initializing scorer...")
    scorer = Scorer(lang=args.lang)

    print("Computing scores...")
    col_names = [
        "task",
        "cand_system",
        "tgt_system",
        "score_name",
        "context_id",
        "candidate_id",
        "target_id",
        "score_value",
    ]
    scores = [col_names]
    base_row = [args.dataset, args.cand_system, args.tgt_system]

    n_contexts = args.n_contexts if args.n_contexts else len(cand_data)
    for context_idx in tqdm(range(n_contexts), total=n_contexts):
        cand_instance = cand_data[context_idx]
        tgt_instance = tgt_data[context_idx]
        assert cand_data[context_idx]["id"] == tgt_data[context_idx]["id"], (
            cand_data[context_idx]["id"],
            tgt_data[context_idx]["id"],
        )

        # Extract data from json blobs
        cand_responses = cand_instance[args.cand_response_key][: args.n_cand_samples]
        cand_context = cand_instance[args.context_key]
        tgt_responses = tgt_instance[args.tgt_response_key][: args.n_tgt_samples]
        tgt_context = tgt_instance[args.context_key]
        if type(cand_context) is list:
            cand_context = " ".join(cand_context)
        if type(tgt_context) is list:
            tgt_context = " ".join(tgt_context)
        assert cand_context == tgt_context

        # Compute lexical and syntactic overlap for different ngram sizes
        for n in range(1, 4):
            for pos_bool, pos_str in [(False, ""), (True, "_pos")]:
                score_name = f"{n}gram{pos_str}_overlap"

                # Response-response overlap
                for cand_idx, cand_response in enumerate(cand_responses):
                    for tgt_idx, tgt_response in enumerate(tgt_responses):
                        if not is_same_system or (is_same_system and (cand_idx < tgt_idx)):
                            score = scorer.ngram_overlap(
                                tgt_response,
                                cand_response,
                                max_len1=args.max_response_length,
                                max_len2=args.max_response_length,
                                n=n,
                                pos=pos_bool,
                            )
                            scores.append(
                                base_row
                                + [f"{score_name}_yy", context_idx, cand_idx, tgt_idx, score]
                            )

                    # Context-response overlap (only when comparing the same system, e.g., human-human)
                    if is_same_system:
                        score = scorer.ngram_overlap(
                            cand_context,
                            cand_response,
                            max_len1=args.max_context_length,
                            max_len2=args.max_response_length,
                            n=n,
                            pos=pos_bool,
                        )
                        scores.append(
                            base_row
                            + [f"{score_name}_xy", context_idx, cand_idx, float("nan"), score]
                        )

        # Compute semantic distance
        cand_embeddings = scorer.compute_embeddings(
            cand_responses, max_len=args.max_response_length
        )
        tgt_embeddings = scorer.compute_embeddings(tgt_responses, max_len=args.max_response_length)
        context_embedding = scorer.compute_embeddings(
            [cand_context], max_len=args.max_context_length
        )
        for score_name, score_func in [
            ("cosine_similarity", scorer.cosine_similarity),
            ("euclidean_similarity", scorer.euclidean_similarity),
        ]:
            for cand_idx in range(len(cand_responses)):
                for tgt_idx in range(len(tgt_responses)):
                    if not is_same_system or (is_same_system and (cand_idx < tgt_idx)):
                        score = score_func(cand_embeddings[cand_idx, :], tgt_embeddings[tgt_idx, :])
                        scores.append(
                            base_row + [f"{score_name}_yy", context_idx, cand_idx, tgt_idx, score]
                        )

                if is_same_system:
                    score = score_func(cand_embeddings[cand_idx, :], context_embedding.squeeze(0))
                    scores.append(
                        base_row + [f"{score_name}_xy", context_idx, cand_idx, float("nan"), score]
                    )

        # Compute lengths
        if is_same_system:
            scores.append(
                base_row
                + ["length_x", context_idx, float("nan"), float("nan"), scorer.length(cand_context)]
            )
            for cand_idx, cand_response in enumerate(cand_responses):
                scores.append(
                    base_row
                    + [
                        "length_y",
                        context_idx,
                        cand_idx,
                        float("nan"),
                        scorer.length(cand_response),
                    ]
                )

    write_params(args)
    write_scores(args.out_path, scores)


def write_params(args):
    print(f"Writing parameters...")
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_path).with_suffix(".info"), "w") as f:
        json.dump({**vars(args), "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S")}, f)


def write_scores(path, scores: List[List[Any]]):
    print(f"Writing scores...")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(Path(f"{path}").with_suffix(".csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(scores[0])
        writer.writerows(scores[1:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name is the dataset, e.g., 'dailydialog++'.",
    )
    parser.add_argument(
        "--cand_system",
        type=str,
        required=True,
        help="Name of the system that generated the candidates",
    )
    parser.add_argument(
        "--tgt_system",
        type=str,
        required=True,
        help="Name of the system that generated the targets",
    )
    parser.add_argument(
        "--cand_path",
        type=str,
        required=True,
        help="Path to candidate response samples",
    )
    parser.add_argument(
        "--tgt_path",
        type=str,
        required=True,
        help="Path to target response samples",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to output file ",
    )
    parser.add_argument(
        "--cand_response_key",
        type=str,
        required=True,
        help="json key to index the responses (agnostic of the human/machine source)",
    )
    parser.add_argument(
        "--tgt_response_key",
        type=str,
        required=True,
        help="json key to index the responses (agnostic of the human/machine source)",
    )
    parser.add_argument(
        "--context_key",
        type=str,
        required=True,
        help="json key to index the context",
    )
    parser.add_argument(
        "--n_cand_samples",
        type=int,
        help="The number of samples (in case of multiple references or generated samples) "
        "for the same context to use to compute utilities.",
    )
    parser.add_argument(
        "--n_tgt_samples",
        type=int,
        default=1,
        help="The number of samples (in case of multiple references or generated samples) "
        "for the same context to use to compute utilities.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en-sent",
        help="The target language to use for computing utility scores.",
    )
    parser.add_argument(
        "--n_contexts",
        type=int,
        help="Control the number of instances we use.",
    )
    parser.add_argument(
        "--max_response_length",
        type=int,
        help="The maximum number of response tokens to consider.",
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        help="The maximum number of context tokens to consider.",
    )
    _args = parser.parse_args()
    main(_args)
