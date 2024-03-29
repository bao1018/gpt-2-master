
# gpt-2-master

Tensorflow and GPT 2 based Spelling Association Tool


> #Related Repo
> https://github.com/bao1018/seifer


## Overall Tech Arch Diagram

![Image of Arch Design](https://i.imgur.com/I1Y3GiG.png)

# gpt-2

Code from the paper ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

We have currently released small (124M parameter), medium (355M parameter), and large (774M parameter) versions of GPT-2<sup>*</sup>, with only the full model as of yet unreleased.  We have also [released a dataset](https://github.com/openai/gpt-2-output-dataset) for researchers to study their behaviors.

You can read about GPT-2 and release decisions in our [original blog post](https://blog.openai.com/better-language-models/) and [6 month follow-up post](https://openai.com/blog/gpt-2-6-month-follow-up/).

<sup>*</sup> *Note that our original parameter counts were wrong due to an error (in our previous blog posts and paper).  Thus you may have seen small referred to as 117M and medium referred to as 345M.*

## Usage

This repository is meant to be a starting point for researchers and engineers to experiment with GPT-2.

For basic information, see our [model card](./model_card.md).

### Some caveats

- GPT-2 models' robustness and worst case behaviors are not well-understood.  As with any machine-learned model, carefully evaluate GPT-2 for your use case, especially if used without fine-tuning or in safety-critical applications where reliability is important.
- The dataset our GPT-2 models were trained on contains many texts with [biases](https://twitter.com/TomerUllman/status/1101485289720242177) and factual inaccuracies, and thus GPT-2 models are likely to be biased and inaccurate as well.
- To avoid having samples mistaken as human-written, we recommend clearly labeling samples as synthetic before wide dissemination.  Our models are often incoherent or inaccurate in subtle ways, which takes more than a quick read for a human to notice.

### Work with us

Please [let us know](mailto:languagequestions@openai.com) if you’re doing interesting research with or working on applications of GPT-2!  We’re especially interested in hearing from and potentially working with those who are studying
- Potential malicious use cases and defenses against them (e.g. the detectability of synthetic text)
- The extent of problematic content (e.g. bias) being baked into the models and effective mitigations

## Development

See [DEVELOPERS.md](./DEVELOPERS.md)

## Contributors

See [CONTRIBUTORS.md](./CONTRIBUTORS.md)

## Citation

Please use the following bibtex entry:
```
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}
```

## Future work

We may release code for evaluating the models on various benchmarks.

We are still considering release of the larger models.


## Setup Steps

> Run base on Python 3

1. Install the python dependencies
```shell
pip install numpy
pip install tensorflow
pip install -r requirements.txt
```
2. Download the models

```shell
python download_model.py 124m
```
3. Download the serving model
TBD，send email to github repo owner for getting that data


## Test GPT 2.0 NLP feature

```shell
# Linux
python3 src/generate_unconditional_samples.py  | tee samples

# With param 
python3 src/generate_unconditional_samples.py --top_k 40 --temperature 0.7 | tee samples

# Interactive mode
python3 src/interactive_conditional_samples.py --top_k 40

```

## Run the GPT 2 Service

```shell
### the path of serving: models should be replaced by your actual codebase path
docker run -t --rm -p 8501:8501 -v /Users/jbao009/Documents/github/wb/gpt-2-master/serving:/models -e MODEL_NAME=use tensorflow/serving
```









[MIT](./LICENSE)
