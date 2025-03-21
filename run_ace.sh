#!/bin/bash

# # rinna/japanese-gpt-neox-3.6b
# python average_causal_effects.py \
#   --model_name "rinna/japanese-gpt-neox-3.6b" \
#   --arch "rinna_japanese-gpt-neox-3.6b" \
#   --archname "GPT-NEOX-3.6B" \
#   --data_pattern "known"

# EleutherAI/gpt-j-6B
python average_causal_effects.py \
  --model_name "EleutherAI/gpt-j-6B" \
  --arch "EleutherAI_gpt-j-6B" \
  --archname "GPT-J-6B" \
  --data_pattern "known"

# meta-llama/Llama-3.2-3B
python average_causal_effects.py \
  --model_name "meta-llama/Llama-3.2-3B" \
  --arch "meta-llama_Llama-3.2-3B" \
  --archname "Llama-3.2-3B" \
  --data_pattern "known"