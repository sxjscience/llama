# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import numpy as np
import tiktoken
import matplotlib.pyplot as plt

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def plot_inference_speed_info(speed_info, model_type):
    fig = plt.figure(figsize = (12, 5)) 
    default_x_ticks = range(len(speed_info))

    tps = [round(ele['tps'], 2) for ele in speed_info]
    # creating the bar plot
    bars = plt.bar(default_x_ticks, tps, color ='maroon', width = 0.4)
    plt.bar_label(bars)
    plt.rc('font', size=14)

    plt.xlabel("Inference batch size")
    plt.ylabel("Tokens per second (TPS)")
    plt.xticks(default_x_ticks, [ele['inference_batch_size'] for ele in speed_info])
    plt.ylim(0, 500)
    # plt.yticks(np.arange(np.min(tps) // 2, np.max(tps) + np.min(tps) // 2, 50))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.title("Inference speed of LLaMA-7B. #Prompt=32. #Prompt Tokens=1083.")
    plt.savefig(f"{model_type}_inference_speed.png")

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:""",
        """Translate English to French:

sea otter => loutre de mer

peppermint => menthe poivrée

plush girafe => girafe peluche

cheese =>""",
    ]

    all_prompts = (prompts * 30)[:32]
    # all_prompts = prompts
    # "tps (normalized via len_prompt + len_gen)": (total_prompt_real_token_length + total_decoded_real_token_length) / (end - start),
    # "tps (normalized via len_gen)": (total_prompt_real_token_length + total_decoded_real_token_length) / (end - start),
    # "tps (normalized via padded_len_gen)": (total_decoded_token_length) / (end - start)

    all_speed_info = []

    for batch_size in [1, 2, 4, 8, 16, 32]:
        results = []
        total_time = 0
        min_prompt_length = 1E8
        max_prompt_length = 0
        total_prompt_padded_token_length = 0
        total_prompt_token_length = 0
        total_padded_token_length = 0
        total_token_length = 0 # Truncate based on EOS ID 

        for start in range(0, len(all_prompts), batch_size):
            end = min(len(all_prompts), start + batch_size)
            torch.cuda.synchronize()
            tic = time.time()
            partial_results, other_info = generator.generate(
                all_prompts[start:end], max_gen_len=256, temperature=temperature, top_p=top_p
            )
            total_time += time.time() - tic
            total_prompt_padded_token_length += other_info["total_prompt_padded_token_length"]
            total_prompt_token_length += other_info["total_prompt_token_length"]
            total_padded_token_length += other_info["total_padded_token_length"]
            total_token_length += other_info["total_token_length"]
            min_prompt_length = min(min_prompt_length, other_info["min_prompt_length"])
            max_prompt_length = max(max_prompt_length, other_info["max_prompt_length"])
            results.extend(partial_results)
            
        speed_info = {
            "inference_batch_size": batch_size, 
            "total_num_prompt": len(all_prompts),
            "min_prompt_length": min_prompt_length, 
            "max_prompt_length": max_prompt_length,
            "total_prompt_token_length": total_prompt_token_length,
            "total_token_length": total_token_length,
            "tps": total_token_length / total_time,
        }
        # Evaluate the total token count via tiktoken
        for openai_model in ["text-davinci-003", "gpt-3.5-turbo"]:
            encoding = tiktoken.encoding_for_model(openai_model)
            speed_info[f"openai_{openai_model}_ntoken"] = sum([len(encoding.encode(ele)) for ele in results])
        print(speed_info)
        all_speed_info.append(speed_info)
    model_type =  "_".join(os.path.abspath(ckpt_dir).split(os.sep)[-2:])
    with open(f"speed_info_{model_type}.json", "w") as f:
        f.write(json.dumps(all_speed_info, indent=4, sort_keys=True))
    plot_inference_speed_info(all_speed_info, model_type)


if __name__ == "__main__":
    fire.Fire(main)
