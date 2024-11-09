import pandas as pd
import datasets
import os
import logging

# 数据集路径设置
TRAIN_META_DATA_PATH = "/Users/yxizhong/workspace/visual-tuning/lora/mmseg_datasets/train.jsonl"
VAL_META_DATA_PATH = "/Users/yxizhong/workspace/visual-tuning/lora/mmseg_datasets/val.jsonl"
IMAGE_DIR = ""
LABEL_DIR = ""


# 定义数据集中有哪些特征，及其类型
_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "label": datasets.Image(),
    },
)


# 定义数据集
class DMCode(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [datasets.BuilderConfig(name="default", version=datasets.Version("0.0.2"))]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description="None",
            features=_FEATURES,
            supervised_keys=None,
            homepage="None",
            license="None",
            citation="None",
        )

    def _split_generators(self, dl_manager):

        return [
            datasets.SplitGenerator(
                # 控制train/test/val
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": TRAIN_META_DATA_PATH,
                    "images_dir": IMAGE_DIR,
                    "labels_dir": LABEL_DIR,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": VAL_META_DATA_PATH,
                    "images_dir": IMAGE_DIR,
                    "labels_dir": LABEL_DIR,
                },
            ),
        ]

    def _generate_examples(self, metadata_path, images_dir, labels_dir):
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            image_path = os.path.join(images_dir, row["img"])
            # 打开文件错误时直接跳过
            try:
                image = open(image_path, "rb").read()
            except Exception as e:
                logging.error(e)
                continue

            label_path = os.path.join(labels_dir, row["seg"])
            # 打开文件错误直接跳过
            try:
                label = open(label_path, "rb").read()
            except Exception as e:
                logging.error(e)
                continue

            yield row["img"], {
                "image": {
                    "path": image_path,
                    "bytes": image,
                },
                "label": {
                    "path": label_path,
                    "bytes": label,
                },
            }
