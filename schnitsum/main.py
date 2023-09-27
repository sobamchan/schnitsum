from math import ceil
from typing import List

from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAMES = [
    "sobamchan/bart-large-scitldr",
    "sobamchan/bart-large-scitldr-distilled-3-3",
    "sobamchan/bart-large-scitldr-distilled-12-3",
    "sobamchan/mbart-large-xscitldr-de",
]
LANG2CODE = {
    "en": "en_XX",
    "de": "de_DE",
    "ja": "ja_XX",
    "it": "it_IT",
    "zh": "zh_CN",
}


class SchnitSum:
    def __init__(
        self, model_name: str = None, tgt_lang: str = "en", use_gpu: bool = False
    ):

        """Easy to use summarization package.
        model_name (str):
        use_gpu (bool): Whether use GPU for inference or not.

        Parameters
        ----------
        model_name : str
            Model name to use.

        use_gpu : bool
            If use GPU or not

        tgt_lang : str
            Language for output summary (default = `en`)
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
        self.tgt_lang = tgt_lang
        print("loaded!")

    def summarize_batch(self, texts: List[str]) -> List[str]:
        inputs = self.tokenizer(
            texts, padding="max_length", truncation=True, return_tensors="pt"
        )
        summary_ids = self.model.generate(
            inputs["input_ids"].to("cuda") if self.use_gpu else inputs["input_ids"],
            max_length=250,
            num_beams=1,
            repetition_penalty=1.0,
            early_stopping=True,
            decoder_start_token_id=self.tokenizer.lang_code_to_id[
                LANG2CODE[self.tgt_lang]
            ]
            if self.tgt_lang != "en"
            else None,
        )
        return self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

    def __call__(self, texts: List[str], batch_size: int = 4) -> List[str]:
        summaries = []
        for i in tqdm(
            range(0, len(texts), batch_size), total=ceil(len(texts) / batch_size)
        ):
            texts_batch = texts[i : i + batch_size]
            summaries += self.summarize_batch(texts_batch)
        return summaries
