import os
import json
import torch
import pandas as pd

# 必要なクラス・関数のインポート
from causal_trace import ModelAndTokenizer, make_inputs, decode_tokens
from globals import DATA_DIR

# 勾配計算は不要なので無効化
torch.set_grad_enabled(False)

# ---------------------------
# モデルとデータセットの対応設定
# ---------------------------
model_dataset_config = [
    {
        "model_name": "rinna/japanese-gpt-neox-3.6b",
        "dataset_type": "ja_question"
    },
    {
        "model_name": "SakanaAI/TinySwallow-1.5B",
        "dataset_type": "ja_question"
    },
    {
        "model_name": "EleutherAI/gpt-j-6B",
        "dataset_type": "prompt"
    },
    {
        "model_name": "meta-llama/Llama-3.2-3B",
        "dataset_type": "prompt"
    }
]

# ---------------------------
# 共通のサンプル処理関数
# ---------------------------
def predict_from_input(model, inp):
    out = model(**inp).logits
    probs = torch.softmax(out[:, -1, :], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p

def predict_from_input_jp(model, inp):
    # output_ids = model.generate(**inp, max_new_tokens=50)
    output_ids = model.generate(
        **inp,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.1
    )
    return output_ids

def process_dataset(data_iter, dataset_label, model_name, mt, input_key):
    """
    data_iter: JSONファイルからロードした各サンプルのリストのイテレータ
    dataset_label: ログ用のデータセット名
    model_name: モデル名（ログ保存にも使用）
    mt: 既に初期化された ModelAndTokenizer インスタンス
    input_key: 入力として使用するキー ("ja_question" または "prompt")
    """
    dataset_results = {}
    reject = 0
    for i, knowledge in enumerate(data_iter):
        subject = knowledge.get("subject", "")
        attribute = knowledge.get("attribute", "")
        if input_key not in knowledge:
            print(f"[{model_name} - {dataset_label}] サンプル {i}: 指定された入力キー '{input_key}' が存在しません")
            reject += 1
            continue
        input_text = knowledge[input_key]

        # # 日本語モデルの場合、[INST] タグで囲む
        # if model_name in ["rinna/japanese-gpt-neox-3.6b", "SakanaAI/TinySwallow-1.5B"]:
        #     input_text = f"[INST]{input_text}[/INST]"
        # モデル入力の作成
        inputs = make_inputs(mt.tokenizer, [input_text])
        
        try:
            # if model_name in ["rinna/japanese-gpt-neox-3.6b", "SakanaAI/TinySwallow-1.5B"]:
            #     predictions = predict_from_input_jp(mt.model, inputs)
            #     output = mt.tokenizer.decode(predictions[0], skip_special_tokens=True)
            if model_name in ["rinna/japanese-gpt-neox-3.6b"]:
                token_ids = mt.tokenizer.encode(f"Q: {input_text}\nA:", add_special_tokens=False, return_tensors="pt")
                with torch.no_grad():
                    output_ids = mt.model.generate(
                        token_ids.to(mt.model.device),
                        max_new_tokens=100,
                        min_new_tokens=100,
                        do_sample=True,
                        temperature=0.8,
                        pad_token_id=mt.tokenizer.pad_token_id,
                        bos_token_id=mt.tokenizer.bos_token_id,
                        eos_token_id=mt.tokenizer.eos_token_id
                    )
                output = mt.tokenizer.decode(output_ids.tolist()[0])
                print(output)
            elif model_name in ["SakanaAI/TinySwallow-1.5B"]:
                # TinySwallow 用の出力形式（Q&A形式でプロンプトを組む）
                prompt = f"Q: {input_text}\nA:"
                token_ids = mt.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
                with torch.no_grad():
                    output_ids = mt.model.generate(
                        token_ids.to(mt.model.device),
                        max_new_tokens=100,
                        min_new_tokens=100,
                        do_sample=True,
                        temperature=0.8,
                        pad_token_id=mt.tokenizer.pad_token_id or mt.tokenizer.eos_token_id,
                        bos_token_id=mt.tokenizer.bos_token_id,
                        eos_token_id=mt.tokenizer.eos_token_id
                    )
                output = mt.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                print(output)
            else:
                preds, _ = predict_from_input(mt.model, inputs)
                output = decode_tokens(mt.tokenizer, [preds])[0]
        except Exception as e:
            print(f"[{model_name} - {dataset_label}] サンプル {i} の処理中にエラー: {e}")
            continue

        dataset_results[f"sample_{i}"] = {
            "input_text": input_text,
            "output": output,
            "subject": subject,
            "attribute": attribute,
        }
        print(f"[{model_name} - {dataset_label}] サンプル {i} を記録")
    print(f"reject: {reject}")
    return dataset_results

# ---------------------------
# JSONからデータを読み込み
# ---------------------------
json_path = os.path.join("data", "known_1000_questions_ja.json")
if os.path.exists(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    # json_data がリスト形式になっていることを前提とする
    data_iter = json_data
    dataset_label = os.path.basename(json_path)
else:
    print(f"JSONファイル {json_path} が存在しません。")
    data_iter = []
    dataset_label = "None"

# ---------------------------
# 全結果を格納する辞書
# ---------------------------
all_results = {}

# ---------------------------
# 各モデルごとの処理
# ---------------------------
for config in model_dataset_config:
    model_name = config["model_name"]
    # config の dataset_type をそのまま入力キーとして利用
    input_key = config["dataset_type"]
    print(f"==== モデル: {model_name} の処理を開始（入力キー: {input_key}） ====")
    mt = ModelAndTokenizer(
        model_name,
        torch_dtype=(torch.float16 if "20b" in model_name else None),
    )
    # 各サンプルの処理
    model_results = process_dataset(data_iter, dataset_label, model_name, mt, input_key)
    all_results[model_name] = {
        "dataset": dataset_label,
        "input_key": input_key,
        "results": model_results
    }
    print(f"==== モデル: {model_name} の処理が完了 ====\n")
    del mt
    torch.cuda.empty_cache()

# ---------------------------
# 全結果を JSON に保存
# ---------------------------
output_json_path = 'output_results_all_models.json'
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)
print(f"全ての結果を {output_json_path} に保存しました。")