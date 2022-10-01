import fire
import sienna

from schnitsum.main import SchnitSum


def summarize(
    model_name: str,
    file: str = None,
    text: str = None,
    opath: str = None,
    batch_size: int = 4,
    use_gpu: bool = False,
):
    ssum = SchnitSum(model_name, use_gpu)

    if file and text:
        raise ValueError("Only one of file or text can be provided.")

    if file:
        docs = sienna.load(file)
    elif text:
        docs = [text]
    else:
        raise ValueError("file or text needs to be provided.")

    summaries = ssum(docs, batch_size)

    if opath:
        sienna.save(summaries, opath)
    else:
        print("\n".join(summaries))


def cli():
    fire.Fire(summarize)
