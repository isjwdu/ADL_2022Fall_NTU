import json

import datasets
from datasets.tasks import QuestionAnsweringExtractive

logger = datasets.logging.get_logger(__name__)


class QAConfig(datasets.BuilderConfig):
    """BuilderConfig for QA dataset."""
    question_file: str = None
    context_file: str = None

    def __init__(self, **kwargs):
        """BuilderConfig for QA dataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(QAConfig, self).__init__(**kwargs)


class QADataset(datasets.GeneratorBasedBuilder):
    """QA dataset: The ADL@NTU Homework 2 Dataset. Version 1.0."""

    BUILDER_CONFIGS = [
        QAConfig(
            name="train",
            version=datasets.Version("1.0.0", ""),
            description="Json file",
        ),
        QAConfig(
            name="eval",
            version=datasets.Version("1.0.0", ""),
            description="Json file",
        ),
        QAConfig(
            name="test",
            version=datasets.Version("1.0.0", ""),
            description="Json file",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="QA dataset",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            task_templates=[
                QuestionAnsweringExtractive(
                    question_column="question", context_column="context", answers_column="answers"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        split_name = {
            "train": datasets.Split.TRAIN,
            "eval": datasets.Split.VALIDATION,
            "test": datasets.Split.TEST,
        }
        return [
            datasets.SplitGenerator(
                name=split_name[self.config.name],
                gen_kwargs={
                    "question_file": self.config.question_file,
                    "context_file": self.config.context_file
                }
            ),
        ]

    def _generate_examples(self, question_file, context_file):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", question_file)
        is_test = self.config.name == "test"

        key = 0
        with open(question_file, encoding="utf-8") as f, open(context_file, encoding="utf-8") as ctx_file:
            squad = json.load(f)
            contexts = json.load(ctx_file)

            if not is_test:
                for article in squad:
                    ctx_id = article["relevant"]
                    context = contexts[ctx_id]
                    answer = article["answer"]
                    yield key, {
                        "id": article["id"],
                        "context": context,
                        "question": article["question"],
                        "answers": {
                            "answer_start": [answer["start"]],
                            "text": [answer["text"]],
                        },
                    }
                    # yield key, {
                    #     "id": article["id"],
                    #     "context": context if not is_test else article["paragraphs"],
                    #     "question": article["question"],
                    #     "answers": {
                    #         "answer_start": [answer["start"]] if not is_test else [],
                    #         "text": [answer["text"]] if not is_test else [],
                    #     },
                    # }
                    key += 1
            else:
                for article in squad:
                    yield key, {
                        "id": article["id"],
                        "question": article["question"],
                        # "paragraphs": article["paragraphs"],
                        "context": "",
                        "answers": {
                            "answer_start": [],
                            "text": [],
                        },
                    }
                    key += 1
