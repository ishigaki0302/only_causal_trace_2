import json
from openai import OpenAI

class PromptChanger:
    def __init__(self):
        self.client = OpenAI(
            # OPENAI_API_KEY の環境変数が使用されるか，
            # 直接 api_key を指定してください．
            api_key="",
        )

    def convert_to_question(self, prompt, subject, attribute):
        """
        与えられた prompt を英語の疑問文に書き換えます．
        出力は厳密に JSON 形式（{"question": "Generated question"}）とします．
        """
        input_text = f"""
### INSTRUCTIONS ###
Rewrite the given prompt as an English question such that its answer is the attribute.

### RULES ###
- Do not change the subject; it must appear exactly as provided.
- The output must be in the EXACT JSON format below with no extra text.
- The JSON object must have exactly one key: "question".

### EXAMPLE ###
## Input ##
prompt: Beats Music is owned by
subject: Beats Music
attribute: Apple
## Output ##
{{"question": "Which company owns Beats Music?"}}

### INPUT ###
prompt: {prompt}
subject: {subject}
attribute: {attribute}
### OUTPUT ###
"""
        count = 0
        while True:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "system", "content": input_text}],
                model="gpt-4",
            )
            output = chat_completion.choices[0].message.content.strip()
            if count >= 5:
                return {"question": ""}
            try:
                result = json.loads(output)
                # 生成された疑問文に subject が含まれているか検証
                if subject in result.get("question", ""):
                    return result
                else:
                    count += 1
            except Exception:
                count += 1
                continue

    def translate_to_japanese(self, english_question, subject, attribute):
        """
        与えられた英語の疑問文を日本語に翻訳します．
        出力は全て全角文字とし，subject は必ず与えられた通りに含むものとします．
        出力は厳密に JSON 形式（{"question": "日本語の疑問文"}）とします．
        """
        input_text = f"""
### 指示 ###
以下の英語の疑問文を、日本語に翻訳しなさい。

### ルール ###
- prompt の主語は、与えられた subject と全く同じつづりで必ず含むこと。
- 出力はすべて全角文字で行いなさい。
- 出力は以下のEXACTなJSONフォーマットのみとし、余計なテキストを含んではならない。
- JSONオブジェクトは、キー "question" のみを持つこと。

### 例 ###
## 入力 ##
subject: Beats Music
attribute: Apple
prompt: Which company owns Beats Music?
## 出力 ##
{{"question": "Beats Musicを所有しているのはどこですか？"}}

### 入力 ###
subject: {subject}
attribute: {attribute}
prompt: {english_question}
### 出力 ###
"""
        count = 0
        while True:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "system", "content": input_text}],
                model="gpt-4",
            )
            output = chat_completion.choices[0].message.content.strip()
            if count >= 5:
                return {"question": ""}
            try:
                result = json.loads(output)
                # 生成された日本語疑問文に subject が含まれているか検証
                if subject in result.get("question", ""):
                    return result
                else:
                    count += 1
            except Exception:
                count += 1
                continue

def main():
    # data/counterfact.json を読み込む
    with open("data/counterfact.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    changer = PromptChanger()
    
    for record in data:
        # requested_rewrite の変換
        requested = record["requested_rewrite"]
        subject = requested["subject"]
        # prompt の {} を subject で埋める
        prompt_template = requested["prompt"]
        english_prompt = prompt_template.format(subject)
        # attribute には target_new の "str" を使用
        attribute = requested["target_new"]["str"]
        
        eng_result = changer.convert_to_question(english_prompt, subject, attribute)
        record["question"] = eng_result.get("question", "")
        ja_result = changer.translate_to_japanese(record["question"], subject, attribute)
        record["ja_question"] = ja_result.get("question", "")
        
        # 各 prompt リストの変換（英語, 日本語それぞれ）
        prompt_lists = {
            "paraphrase_prompts": "paraphrase_questions",
            "neighborhood_prompts": "neighborhood_questions",
            "attribute_prompts": "attribute_questions",
            "generation_prompts": "generation_questions",
        }
        for key, out_key in prompt_lists.items():
            eng_questions = []
            ja_questions = []
            for prompt in record.get(key, []):
                # ここでは元の prompt に subject を含むことを前提とするので，
                # 直接変換処理を実施
                eng_res = changer.convert_to_question(prompt, subject, attribute)
                eng_q = eng_res.get("question", "")
                eng_questions.append(eng_q)
                
                ja_res = changer.translate_to_japanese(eng_q, subject, attribute)
                ja_q = ja_res.get("question", "")
                ja_questions.append(ja_q)
            record[out_key] = eng_questions
            # 日本語版のフィールド名は先頭に "ja_" を付ける
            record["ja_" + out_key] = ja_questions

    # 英語疑問文のみのファイルを作成（日本語版のフィールドは除去）
    english_data = []
    for rec in data:
        rec_eng = rec.copy()
        # 日本語フィールドの削除（"ja_question", "ja_paraphrase_questions", など）
        keys_to_remove = [k for k in rec_eng.keys() if k.startswith("ja_")]
        for k in keys_to_remove:
            del rec_eng[k]
        english_data.append(rec_eng)
        
    with open("counterfact_question.json", "w", encoding="utf-8") as f:
        json.dump(english_data, f, ensure_ascii=False, indent=2)
    
    # 日本語疑問文のみのファイルを作成（英語版のフィールドは除去）
    japanese_data = []
    for rec in data:
        rec_jp = rec.copy()
        # 英語フィールド（"question", "paraphrase_questions", etc.）を削除
        keys_to_remove = [k for k in rec_jp.keys() if not k.startswith("ja_") and k not in ["case_id", "pararel_idx", "requested_rewrite", 
                                                                                           "paraphrase_prompts", "neighborhood_prompts", 
                                                                                           "attribute_prompts", "generation_prompts"]]
        for k in keys_to_remove:
            del rec_jp[k]
        japanese_data.append(rec_jp)
    
    with open("counterfact_question_jp.json", "w", encoding="utf-8") as f:
        json.dump(japanese_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()