import json

import datasets
from datasets.tasks import QuestionAnsweringExtractive

logger = datasets.logging.get_logger(__name__)


class SlotTagConfig(datasets.BuilderConfig):
    train_file: str = None
    eval_file: str = None

    def __init__(self, **kwargs):
        super(SlotTagConfig, self).__init__(**kwargs)


class SlotTagDataset(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        SlotTagConfig(
            name="train",
            version=datasets.Version("1.0.0", ""),
            description="Json file",
        ),
        SlotTagConfig(
            name="eval",
            version=datasets.Version("1.0.0", ""),
            description="Json file",
        ),
        SlotTagConfig(
            name="test",
            version=datasets.Version("1.0.0", ""),
            description="Json file",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="Seq Cls dataset",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(
                        datasets.Value("string"),
                    ),
                    "tags": datasets.Sequence(
                        datasets.Value("string"),
                    ),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": self.config.train_file,
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": self.config.eval_file,
                }
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        is_test = self.config.name == "test"

        key = 0
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for instance in data:
                yield key, {
                    **instance
                }
                key += 1

