# Schnitsum: Easy to use neural network based summarization models

This package enables to generate summaries of you documents of interests.

Currently, we support following models,

- [BART (large)](https://aclanthology.org/2020.acl-main.703) fine-tuned on computer science papers (ref. [SciTLDR](https://aclanthology.org/2020.findings-emnlp.428)).
  - Model name: `sobamchan/bart-large-scitldr`
- [BART (large)](https://aclanthology.org/2020.acl-main.703) fine-tuned on computer science papers (ref. [SciTLDR](https://aclanthology.org/2020.findings-emnlp.428)). Then distilled (by [`shrink and fine-tune`](http://arxiv.org/abs/2010.13002)) to have 65% parameters less.
  - Model name: `sobamchan/bart-large-scitldr-distilled-3-3`
- [BART (large)](https://aclanthology.org/2020.acl-main.703) fine-tuned on computer science papers (ref. [SciTLDR](https://aclanthology.org/2020.findings-emnlp.428)). Then distilled (by [`shrink and fine-tune`](http://arxiv.org/abs/2010.13002)) to have 37% parameters less.
  - Model name: `sobamchan/bart-large-scitldr-distilled-12-3`

we are planning to expand coverage soon to other sizes, domains, languages, models soon.


# Installation

```bash
pip install schnitsum  # or poetry add schnitsum
```

This will let you generate summaries with CPUs only, if you want to utilize your GPUs, please follow the instruction by PyTorch, [here](https://pytorch.org/get-started/locally/).


# Usage

## From Command Line
Pass document as an argument and print the summary
```sh
> schnitsum --model-name sobamchan/bart-large-scitldr-distilled-3-3 --text "Text to summarize"
```

Pass documents as a file and save summaries in a file.
Input file needs to contain documents line by line. [example](https://github.com/sobamchan/schnitsum/blob/main/examples/docs.txt)
```sh
> schnitsum --model-name sobamchan/bart-large-scitldr-distilled-3-3 --file docs.txt --opath sums.txt
```

## From Python
```py3
from schnitsum import SchnitSum

model = SchnitSum("sobamchan/bart-large-scitldr-distilled-3-3")
docs = [
    "Document you want to summarize."
]
summaries = model(docs)
print(summaries)
```
