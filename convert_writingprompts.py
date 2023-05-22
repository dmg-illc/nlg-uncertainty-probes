import argparse
import json
from collections import defaultdict

import numpy as np


def main(args):
    with open(args.source_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines()]
    with open(args.target_path, "r", encoding="utf-8") as f:
        stories = [line.strip() for line in f.readlines()]

    assert len(prompts) == len(stories)
    print(f"Dataset contains {len(prompts)} prompt-story pairs.")

    data = defaultdict(list)
    for prompt, story in zip(prompts, stories):
        # remove "[ WP ]" from the beginning of the prompt string
        prompt = prompt[7:]
        story = story.replace("<newline>", "")
        story = " ".join(story.split())
        if story not in data[prompt]:
            data[prompt].append(story)

    print(
        f"Dataset contains {len(data)} unique prompts, for a total "
        f"of {sum([len(lst) for lst in data.values()])} stories"
    )

    json_data = []
    n_refs = []
    for i, (prompt, multiple_refs) in enumerate(data.items()):
        if len(multiple_refs) >= args.min_multiple_refs:
            json_data.append(
                {
                    "id": i,
                    "prompt": prompt,
                    "stories": multiple_refs,
                }
            )
            n_refs.append(len(multiple_refs))

    with open(args.output_path, "w") as out_file:
        for entry in json_data:
            json.dump(entry, out_file)
            out_file.write("\n")

    print(
        f"{len(json_data)} with at least {args.min_multiple_refs} references written to disk: {args.output_path}"
    )
    print(f"Maximum number of references: {np.max(n_refs)}")
    print(
        f"Mean: {np.mean(n_refs)}  Standard deviation: {np.std(n_refs)}  Median: {np.median(n_refs)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="Path to story prompts file (wp_source)",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        required=True,
        help="Path to story completions file (wp_target)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output json file containing one prompt (and multiple stories) per line",
    )
    parser.add_argument(
        "--min_multiple_refs",
        type=int,
        default=5,
        help="The minimum number of story completions per prompt. "
        "Prompts with less than `min_multiple_refs` completions are discared.",
    )
    _args = parser.parse_args()
    main(_args)
