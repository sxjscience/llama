# Other scripts related to LLaMA

Inference speed profiling

```shell
# 7B model
CKPT_DIR=/home/ubuntu/data/models/LLaMA/7B
TOKENIZER_PATH=/home/ubuntu/data/models/LLaMA/tokenizer.model

torchrun --nproc_per_node 1 speed_profile.py --ckpt_dir ${CKPT_DIR} --tokenizer_path ${TOKENIZER_PATH}
```
