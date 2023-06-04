import json


class confConstruction:

    def __init__(self, dict_obj):
        self.__dict__.update(dict_obj)


def load_config(config_dict):
    """
    此函数可以将字典格式的配置信息转换为对象的形式，并可以以属性的方式来访问和修改配置信息
    :param config_dict:
    :return:
    """
    return json.loads(json.dumps(config_dict), object_hook=confConstruction)
