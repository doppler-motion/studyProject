# 导入必要的包
from flask import Flask, render_template
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
from loguru import logger

# 导入本地包
from api_interaction.chatgpt_api_interaction import apiInteractionManager

app = Flask(__name__)
api = Api(app)
CORS(app)

parser = reqparse.RequestParser()
parser.add_argument("user_input", type=str, location="json")


# 与chatgpt api 交互的函数
api_interactor = apiInteractionManager()


class requestOpenAI(Resource):

    def get(self):

        res_dict = {"code": 0, "message": "success", "res": ""}
        logger.info(res_dict)
        args = parser.parse_args()
        user_request_input = args["user_input"]
        logger.info(user_request_input)
        try:
            res = api_interactor.send_msg(user_input=user_request_input)
            res_dict["res"] = res
        except Exception as e:
            res_dict["code"] = -1
            res_dict["message"] = "request failed"
            res_dict["res"] = f"!!! This request is bad, please check the running log! error : {e}"
        return res_dict


@app.route("/")
def index():
    return render_template("index.html")


api.add_resource(requestOpenAI, "/request_openai")

if __name__ == "__main__":
    host, port = "0.0.0.0", 6600
    logger.info(f"running at host: {host}, port: {port}")
    app.run(host=host, port=port, debug=False)
