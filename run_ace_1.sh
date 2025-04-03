#!/bin/bash

# # ishigaki_2
# python average_causal_effects.py \
#   --model_name "SakanaAI/TinySwallow-1.5B" \
#   --arch "SakanaAI_TinySwallow-1.5B" \
#   --archname "TinySwallow-1.5B" \
#   --dataset_type "ja_question"

# # ishigaki_2
# python average_causal_effects.py \
#   --model_name "rinna/japanese-gpt-neox-3.6b" \
#   --arch "rinna_japanese-gpt-neox-3.6b" \
#   --archname "GPT-NEOX-3.6B" \
#   --dataset_type "ja_question"

# ishigaki_1_new
# python average_causal_effects.py \
#   --model_name "EleutherAI/gpt-j-6B" \
#   --arch "EleutherAI_gpt-j-6B" \
#   --archname "GPT-J-6B" \
#   --dataset_type "prompt"

python average_causal_effects.py \
  --model_name "EleutherAI/gpt-j-6B" \
  --arch "EleutherAI_gpt-j-6B" \
  --archname "GPT-J-6B" \
  --dataset_type "question"

# python average_causal_effects.py \
#   --model_name "meta-llama/Llama-3.2-3B" \
#   --arch "meta-llama_Llama-3.2-3B" \
#   --archname "Llama-3.2-3B" \
#   --dataset_type "prompt"

# python average_causal_effects.py \
#   --model_name "meta-llama/Llama-3.2-3B" \
#   --arch "meta-llama_Llama-3.2-3B" \
#   --archname "Llama-3.2-3B" \
#   --dataset_type "question"