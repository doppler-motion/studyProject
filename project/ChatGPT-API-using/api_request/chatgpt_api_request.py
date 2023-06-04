import requests
import json


class chatgptAPIRequest(object):
    def __init__(self, key, model_name, request_url):
        super().__init__()
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.model_name = model_name
        self.request_url = request_url

    def post_request(self, message):
        data = {
            "model": self.model_name,
            "messages": message
        }

        data = json.dumps(data)

        response = requests.post(self.request_url, headers=self.headers, data=data)

        return response


if __name__ == "__main__":
    keys = "Your OpenAI API keys"
    model_name = "gpt-3.5-turbo"
    request_url = "https://api.openai.com/v1/chat/completions"
    api_request = chatgptAPIRequest(key=keys, model_name=model_name, request_url=request_url)

    while True:
        user_input = input("user input: ")  # 需要输入[{}]这种形式的数据
        user_input = json.loads(user_input)
        res = api_request.post_request(user_input)

        response = res.json()["choices"][0]["message"]["content"]

        print(f"chatGPT: {response}")
