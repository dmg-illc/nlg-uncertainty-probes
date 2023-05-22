import json
from types import SimpleNamespace
from typing import List, Dict, Any

import numpy as np
import torch
import random
import numpy
import re
import argparse
import pathlib


def set_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def load_jsonl(path: str, return_obj=False) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(
                json.loads(line, object_hook=lambda d: SimpleNamespace(**d))
                if return_obj
                else json.loads(line)
            )
    return data


def load_wmt14_data(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        _id, source, original_reference, additional_references = None, None, None, []
        for line in f.readlines():
            line = line.strip().split("\t")
            sentence_type, sentence = line
            if sentence_type.startswith("S"):  # new source sentence
                if source:
                    data.append(
                        {
                            "id": _id,
                            "source": source,
                            "original_reference": original_reference,
                            "additional_references": additional_references,
                        }
                    )
                    original_reference, additional_references = None, []
                _id = sentence_type.split("-")[1]
                source = sentence.strip()
            elif sentence_type.startswith("T"):  # original translation
                original_reference = sentence.strip()
            elif sentence_type.startswith("R"):  # a single reference
                additional_references.append(sentence.strip())
        # Add trailing data point
        data.append(
            {
                "id": _id,
                "source": source,
                "original_reference": original_reference,
                "additional_references": additional_references,
            }
        )
    return data


def transform_wmt13_data(input_path: str, output_path: str) -> None:
    data = load_wmt14_data(input_path)
    with open(output_path, "w") as out_file:
        for entry in data:
            json.dump(entry, out_file)
            out_file.write("\n")


def _transform_dailydialog(lines: List[str]) -> List[Dict[str, Any]]:
    data = list()
    for i, line in enumerate(lines):
        utterances = line.split("__eou__")
        utterances = [
            re.sub(r"""\s([?.!,:"'](?:\s|$))""", r"\1", utterance).strip()
            for utterance in utterances
        ]
        data.append(
            {
                "context": utterances[:-2],
                "positive_responses": [utterances[-2]],
                "id": i,
            }
        )
    return data


def transform_dailydialog(input_path: str, output_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as in_file:
        data = _transform_dailydialog(in_file.readlines())
    with open(output_path, "w") as out_file:
        for entry in data:
            json.dump(entry, out_file)
            out_file.write("\n")
    return data


def transform_asset_dataset(
    input_dir: str, output_dir: str, n_refs: int = 10
) -> List[Dict[str, Any]]:
    for split in ["test", "valid"]:
        with open(pathlib.Path(input_dir) / f"asset.{split}.orig", "r") as source_file:
            source = source_file.readlines()
            references = list()
            for i in range(n_refs):
                with open(pathlib.Path(input_dir) / f"asset.{split}.simp.{i}", "r") as ref_file:
                    references.append(ref_file.readlines())
        data = list()
        for i, sent in enumerate(source):
            data.append(
                {
                    "id": i,
                    "source": sent.strip(),
                    "references": [references[r][i].strip("\n") for r in range(n_refs)],
                }
            )
        out_path = pathlib.Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / f'{split if split=="test" else "val"}.json', "w") as out_file:
            for entry in data:
                json.dump(entry, out_file)
                out_file.write("\n")
    return data


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to text samples to compute utility scores on",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to output file ",
    )
    parser.add_argument("--transform_dataset", type=str, required=True, help="Dataset to transform")
    args = parser.parse_args()
    if args.transform_dataset == "asset":
        data = transform_asset_dataset(args.data_path, args.out_path)
