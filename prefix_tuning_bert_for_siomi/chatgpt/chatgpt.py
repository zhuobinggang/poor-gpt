import openai
import os
from reader import read_regular_ds_zip
import re

openai.api_key = os.getenv("OPENAI_API_KEY")

def send(content, role='user', max_tokens=200):
    if content == '' or content is None:
        return None
    else:
        return openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                max_tokens=max_tokens,
                messages=[{'role': role, 'content': content}],
        )

def only_contents(ress):
    return [item['choices'][0]['message']['content'] for item in ress]


query = """Q: Of the two Japanese words "天気" or "雲行き", which is more poetic?
Meaning: weather, atmosphere
天気: common, neutral
雲行き: specific, poetic
Answer: 雲行き
Q: Of the two Japanese words "おめでた" or "妊娠", which is more poetic?
Meaning: pregnancy
おめでた: casual, congratulatory
妊娠: formal, medical
Answer: おめでた
"""

def test1():
    ds = read_regular_ds_zip()
    bad, good = ds[100]
    content = query + f'Q: Of the two Japanese words "{bad}" or "{good}", which is more poetic?\n'
    res = send(content)

def run():
    ds = read_regular_ds_zip()
    contents = []
    for bad, good in ds:
        content = query + f'Q: Of the two Japanese words "{bad}" or "{good}", which is more poetic?\n'
        contents.append(content)
    ress = [send(content) for content in contents]
    # 处理结果
    f = open('exp_log/chatgpt_res_content.txt')
    lines = f.readlines()
    f.close()
    answers = []
    for line in lines:
        if line.startswith('Answer:'):
            answers.append(line)
    answers2 = []
    # 处理结果
    for line in answers:
        a = re.findall('Answer: (.*?)\n', line)[0]
        answers2.append(a)
    # 储存结果
    f = open('exp_log/chatgpt_answers_processed.txt', 'w')
    for line in answers2:
        f.write(line + '\n')
    f.close()
    # 机器检测
    f = open('exp_log/good_words.txt')
    trues = f.readlines()
    f.close()
    f = open('exp_log/chatgpt_answers_processed.txt')
    preds = f.readlines()
    f.close()
    pairs = []
    count = 0
    for pred, true in zip(preds, trues):
        if pred != true:
            count += 1
            pairs.append((pred, true))
    
