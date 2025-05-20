from typing import Text, Any, Mapping
import yaml

from tools import yaml_loader


def read_file(file_path: Text) -> Text:
  with open(file_path, "r") as f:
    content = f.read()
  return content


def read_config(config: Text, with_include=True
                ) -> Mapping[Text, Any]:
  loader_used = yaml_loader.Loader if with_include else yaml.Loader
  with open(config, "r") as f:
    return yaml.load(f, loader_used)
