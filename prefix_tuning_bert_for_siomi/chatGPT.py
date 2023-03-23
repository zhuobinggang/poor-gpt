import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def send(content, role='user', max_tokens=500):
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


############

# 构造能够直接发送的请求
def create_request_content(ld):
    requests = []
    for ss, labels in ld:
        text = ''
        for idx, s in enumerate(ss):
            if idx == 2:
                s = '🔪' + s
            if s is not None:
                text += s.strip()
        requests.append(text)
    requests = [f'Please judge whether the marked sentence is the beginning of a paragraph: {text}' for text in requests]
    return requests
