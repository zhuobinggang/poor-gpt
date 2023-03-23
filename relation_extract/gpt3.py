import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        max_tokens=1000,
        messages=[{'role': 'user', 'content': '持有武器为冰属性皇家骑士剑的等级为10的玩家桐人在湖边遭遇了等级为12的冰霜哥布林精英，战斗结果为惨败，请你构思战斗情节'}],
)

prompt = '持有武器为冰属性皇家骑士剑的等级为10的玩家桐人在湖边遭遇了等级为12的冰霜哥布林精英，战斗结果为惨败，以下为战斗情节:'

openai.Completion.create(
        model='babbage',
        # model='ada',
        max_tokens=1000,
        prompt=prompt
)
