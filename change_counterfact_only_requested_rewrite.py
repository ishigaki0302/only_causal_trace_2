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

def main():
    # data/counterfact.json を読み込む
    with open("data/counterfact.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    changer = PromptChanger()
    
    # 各レコードについて，requested_rewrite の prompt を対象に変換する
    for record in data:
        requested = record["requested_rewrite"]
        subject = requested["subject"]
        # prompt の {} を subject で埋める
        prompt_template = requested["prompt"]
        english_prompt = prompt_template.format(subject)
        # attribute には target_new の "str" を使用
        attribute = requested["target_new"]["str"]
        
        eng_result = changer.convert_to_question(english_prompt, subject, attribute)
        record["question"] = eng_result.get("question", "")
    
    # 結果を counterfact_question.json として出力
    with open("counterfact_question.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()