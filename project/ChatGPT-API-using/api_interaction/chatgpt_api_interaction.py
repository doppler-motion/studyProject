import time

from loguru import logger

# 引用本地模块
from config.config import config_dict
from api_request.chatgpt_api_request import chatgptAPIRequest
from utils.common_func import load_config


class apiInteractionManager:
    def __init__(self):
        # load config
        config = load_config(config_dict=config_dict)
        keys = config.Access_config.api_key
        model_name = config.Request_config.model_name
        request_url = config.Request_config.request_url

        self.api_request = chatgptAPIRequest(key=keys, model_name=model_name, request_url=request_url)

    def send_msg(self, user_input):

        start_time = time.time()

        res = self.api_request.post_request(user_input)
        end_time = time.time()

        logger.info(f"post request time cost: {end_time - start_time}")

        if res.status_code == 200:
            logger.info(res.json())
            response = res.json()['choices'][0]['message']['content']
            return response
        else:
            logger.info(res.json())
            return "!!! This request go wrong, please check !!!"