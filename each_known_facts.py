import os
import json
import torch
import pandas as pd

# 必要なクラス・関数のインポート
from causal_trace import ModelAndTokenizer, make_inputs, decode_tokens
from knowns import KnownsDataset
from globals import DATA_DIR

# 勾配計算は不要なので無効化
torch.set_grad_enabled(False)

# ---------------------------
# モデルとデータセットの対応設定
# ---------------------------
# ※ rinna は CSV, EleutherAI と meta-llama は KnownsDataset を使用する
model_dataset_config = [
    {
        "model_name": "rinna/japanese-gpt-neox-3.6b",
        "dataset_type": "csv",
        "csv_path": "data/en2jp_data.csv"
    },
    {
        "model_name": "SakanaAI/TinySwallow-1.5B",
        "dataset_type": "csv",
        "csv_path": "data/en2jp_data.csv"
    },
    {
        "model_name": "EleutherAI/gpt-j-6B",
        "dataset_type": "knowns"
    },
    {
        "model_name": "meta-llama/Llama-3.2-3B",
        "dataset_type": "knowns"
    }
]

# ---------------------------
# 共通のサンプル処理関数
# ---------------------------
def predict_from_input(model, inp):
    out = model(**inp).logits
    probs = torch.softmax(out[:, -1, :], dim=1)
    p, preds =  torch.max(probs, dim=1)
    return preds, p
def predict_from_input_rinnna(model, inp):
    # モデルによる生成（ビームサーチやトップk/top-pサンプリングもオプションで設定可能）
    output_ids = model.generate(**inp, max_new_tokens=10)
    return output_ids

def process_dataset(data_iter, dataset_label, model_name, mt):
    """
    data_iter: 各サンプルのイテレータ（pandas.DataFrame.iterrows() や enumerate(knowns) など）
    dataset_label: ログ用のデータセット名
    model_name: モデル名（ログ保存にも使用）
    mt: 既に初期化された ModelAndTokenizer インスタンス
    """
    dataset_results = {}
    reject = 0
    for i, knowledge in data_iter:
        # pandas の場合は knowledge が Series として渡される
        if isinstance(knowledge, pd.Series):
            prompt = knowledge["prompt"]
            attribute = knowledge["attribute"]
            subject = knowledge["subject"]
        else:
            # KnownsDataset の場合は辞書形式と仮定
            prompt = knowledge["prompt"]
            attribute = knowledge["attribute"]
            subject = knowledge.get("subject", "")

        # モデル入力の作成
        inputs = make_inputs(mt.tokenizer, [prompt])
        # 出力検証用のキーとして attribute を利用（例: 英語の場合、出力に attribute がそのまま含まれているか）
        o = attribute

        try:
            # predict_from_input は (予測トークン, 基本スコア) のタプルのリストを返す
            if model_name == "rinna/japanese-gpt-neox-3.6b":
                predictions = predict_from_input_rinnna(mt.model, inputs)
            else:
                predictions = predict_from_input(mt.model, inputs)
            answer_t, *_ = predictions[0]
            output = decode_tokens(mt.tokenizer, [answer_t])[0]
        except Exception as e:
            print(f"[{model_name} - {dataset_label}] サンプル {i} の処理中にエラー: {e}")
            continue

        # 英語プロンプトの場合は、ASCII 判定で出力内に attribute が含まれているかチェック
        if prompt.isascii():
            if attribute not in output:
                print(f"[{model_name} - {dataset_label}] サンプル {i}: 英語プロンプトで attribute が出力に含まれていません")
                reject += 1
                continue

        dataset_results[f"sample_{i}"] = {
            "prompt": prompt,
            "output": output,
            "subject": subject,
            "attribute": attribute,
        }
        print(f"[{model_name} - {dataset_label}] サンプル {i} を記録")
    print(f"reject: {reject}")
    return dataset_results

# ---------------------------
# 全結果を格納する辞書
# ---------------------------
all_results = {}

# ---------------------------
# 各モデルごとの処理
# ---------------------------
for config in model_dataset_config:
    model_name = config["model_name"]
    print(f"==== モデル: {model_name} の処理を開始 ====")
    # モデル・トークナイザーの初期化
    mt = ModelAndTokenizer(
        model_name,
        torch_dtype=(torch.float16 if "20b" in model_name else None),
    )
    
    # データセットの読み込みとサンプルイテレータの準備
    if config["dataset_type"] == "csv":
        csv_path = config["csv_path"]
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            data_iter = df.iterrows()
            dataset_label = os.path.basename(csv_path)
        else:
            print(f"CSVファイル {csv_path} が存在しません。")
            continue
    elif config["dataset_type"] == "knowns":
        try:
            knowns = KnownsDataset(DATA_DIR)
            data_iter = enumerate(knowns)
            dataset_label = "KnownsDataset"
        except Exception as e:
            print(f"KnownsDataset の読み込みに失敗しました: {e}")
            continue
    else:
        print("不明な dataset_type")
        continue

    # 各サンプルの処理
    model_results = process_dataset(data_iter, dataset_label, model_name, mt)
    all_results[model_name] = {
        "dataset": dataset_label,
        "results": model_results
    }
    print(f"==== モデル: {model_name} の処理が完了 ====\n")
    # 各モデルごとの処理が完了した後
    del mt
    torch.cuda.empty_cache()

# ---------------------------
# 全結果を JSON に保存
# ---------------------------
output_json_path = 'output_results_all_models.json'
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)
print(f"全ての結果を {output_json_path} に保存しました。")