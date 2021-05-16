"""
実験設定関連
yamlファイル + dataclassで設定を読み込む
"""
import dataclasses
import yaml
import argparse
from typing import Any, Dict, Tuple
import os
__all__ = ["get_config"]


#設定ファイルの引数を受け取る
def setting_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""
        train a network.
        """
    )
    parser.add_argument("--inst",type=str)
    parser.add_argument("--config", type=str, required=True,help="path of a config file")
    parser.add_argument("--summary", type=bool,default=False,help="Output a summary of the experiment to ./result")
    return parser.parse_args()


#実験のディレクトリ構成チェック


#config.yamlをdataclassに変換
@dataclasses.dataclass(frozen=True)
class Config:

    model: str
    train_data: str
    test_data: str
    batch_size: int = 1
    max_epochs: int = 1
    patience: int = 10
    early_stopping: bool = False 
    rate: float = 0.2
    learning_rate: float = 0.001

    def __post_init__(self) -> None:
        #self._type_check()
        #self._value_check()

        print("-" * 10, "Experiment Configuration", "-" * 10)
        #pprint.pprint(dataclasses.asdict(self), width=1)
    """
    def _value_check(self) -> None:
        #if not os.path.exists(self.model):
            #raise FileNotFoundError("model path is not found.")
        if not os.path.exists(self.train_data):
            raise FileNotFoundError("train_data path is not found.")
        #if not os.path.exists(self.test_data):
        #    raise FileNotFoundError("test_data path is not found.")
    """


def convert_list2tuple(_dict: Dict[str, Any]) -> Dict[str, Any]:
    # cannot use list in dataclass because mutable defaults are not allowed.
    for key, val in _dict.items():
        if isinstance(val, list):
            _dict[key] = tuple(val)

    return _dict

def get_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    config_dict = convert_list2tuple(config_dict)
    config = Config(**config_dict)
    return config
