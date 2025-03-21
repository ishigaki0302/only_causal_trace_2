import json
from openai import OpenAI

class PromptChanger:
    def __init__(self):
        self.client = OpenAI(
            # Defaults to os.environ.get("OPENAI_API_KEY")
            api_key="",
        )

    def convert_to_question(self, prompt, subject, attribute):
        """
        Convert the given prompt into an English question about the attribute.
        The output is strictly in JSON format: {"question": "Generated question"}.
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
                messages=[
                    {"role": "system", "content": input_text},
                ],
                model="gpt-4",
            )
            output = chat_completion.choices[0].message.content.strip()
            if count >= 5:
                return {"question": ""}
            try:
                result = json.loads(output)
                # Validate that the generated question contains the subject exactly
                if subject in result.get("question", ""):
                    return result
                else:
                    count += 1
            except Exception:
                count += 1
                continue

    def translate_to_japanese(self, english_question, subject, attribute):
        """
        Translate the given English question into Japanese.
        The output must be in full-width characters and include the subject exactly.
        The output is strictly in JSON format: {"question": "Japanese question"}.
        """
        input_text = f"""
### 指示 ###
以下の英語の疑問文を、日本語に翻訳しなさい。

### ルール ###
- promptの主語は、与えられたsubjectと全く同じつづりで必ず含むこと。
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
                messages=[
                    {"role": "system", "content": input_text},
                ],
                model="gpt-4",
            )
            output = chat_completion.choices[0].message.content.strip()
            if count >= 5:
                return {"question": ""}
            try:
                result = json.loads(output)
                # Validate that the translated question contains the subject exactly
                if subject in result.get("question", ""):
                    return result
                else:
                    count += 1
            except Exception:
                count += 1
                continue

def main():
    # Load the JSON data from data/known_1000.json
    with open("data/known_1000.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    changer = PromptChanger()
    
    # Process each record: convert the prompt to an English question, then translate it to Japanese.
    for item in data:
        subject = item["subject"]
        attribute = item["attribute"]
        prompt_text = item["prompt"]
        
        eng_result = changer.convert_to_question(prompt_text, subject, attribute)
        item["question"] = eng_result.get("question", "")
        
        ja_result = changer.translate_to_japanese(item["question"], subject, attribute)
        item["ja_question"] = ja_result.get("question", "")
    
    # Write the results into a new JSON file
    with open("data/known_1000_questions_ja.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()