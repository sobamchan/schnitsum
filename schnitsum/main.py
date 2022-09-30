from typing import List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAMES = ["sobamchan/bart-large-scitldr"]


class SchnitSum:
    def __init__(self, model_name: str = None, use_gpu=False):
        """Easy to use summarization package.
        model_name (str): Model name to use.
        use_gpu (bool): Whether use GPU for inference or not.
        """

        if model_name not in MODEL_NAMES:
            model_names_str = "\n".join([f"- {name}" for name in MODEL_NAMES])
            raise ValueError(
                f"{model_name} is not currently supported. Pick one from\n"
                f"{model_names_str}"
            )

        print(f"Initializing a {model_name} ...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if use_gpu:
            model = model.to("cuda")

        self.use_gpu = use_gpu
        self.model = model
        self.model_name = model_name
        print("loaded!")

    def summarize_batch(self, texts: List[str]) -> List[str]:
        inputs = self.tokenizer(
            texts, padding="max_length", truncation=True, return_tensors="pt"
        )
        summary_ids = self.model.generate(
            inputs["input_ids"].to("cuda") if self.use_gpu else inputs["input_ids"],
            max_length=250,
            num_beams=1,
            early_stopping=True,
        )
        return self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

    def __call__(self, texts: List[str]) -> List[str]:
        return self.summarize_batch(texts)
