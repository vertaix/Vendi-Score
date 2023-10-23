# The Vendi Score: A Diversity Evaluation Metric for Machine Learning

This repository contains the implementation of the Vendi Score (VS), a metric for evaluating diversity in machine learning.
The input to metric is a collection of samples and a pairwise similarity function, and the output is a number, which can be interpreted as the effective number of unique elements in the sample.
Specifically, given a positive semi-definite matrix $K \in \mathbb{R}^{n \times n}$ of similarity scores, the score is defined as:
$$\mathrm{VS}(K) = \exp(-\mathrm{tr}(K/n \log K/n)) = \exp(-\sum_{i=1}^n \lambda_i \log \lambda_i),$$
where $\lambda_i$ are the eigenvalues of $K/n$ and $0 \log 0 = 0$.
That is, the Vendi Score is equal to the exponential of the von Neumann entropy of $K/n$, or the Shannon entropy of the eigenvalues, which is also known as the effective rank.

For more information, please see our paper, [The Vendi Score: A Diversity Evaluation Metric for Machine Learning](https://arxiv.org/abs/2210.02410) and our follow-up paper [Cousins of the Vendi Score: A Family of Similarity-Based Diversity Metrics For Science And Machine Learning](https://arxiv.org/abs/2310.12952).

## Installation

You can install `vendi_score` from pip:
```
pip install vendi_score
```
or by cloning this repository:
```
git clone https://github.com/vertaix/Vendi-Score.git
cd Vendi-Score
pip install -e .
```
`vendi_score` includes some optional dependencies for computing predefined similarity score between images, text, or molecules. You can install these dependencies with a command as in the following:
```
pip install vendi_score[images]
pip install vendi_score[text,molecules]
pip install vendi_score[all]
```

## Usage

The input to `vendi_score` is a list of samples and a similarity function, `k`, mapping a pair of elements to a similarity score. `k` should be symmetric, and `k(x, x) = 1`:
```python
import numpy as np
from vendi_score import vendi

samples = [0, 0, 10, 10, 20, 20]
k = lambda a, b: np.exp(-np.abs(a - b))

vendi.score(samples, k)

# 2.9999
```
If you already have precomputed a similarity matrix:
```python
K = np.array([[1.0, 0.9, 0.0],
              [0.9, 1.0, 0.0],
              [0.0, 0.0, 1.0]])
vendi.score_K(K)

# 2.1573
```
One can also compute Vendi Scores of different orders $q$. Large orders measure diversity with a greater emphasis on common elements. See our latest [pre-print](https://arxiv.org/abs/2310.12952) for more details on the behavior of the Vendi Score with different orders $q$.   

```python
vendi.score(samples, k, q=1.)
```

If your similarity function is a dot product between normalized
embeddings $X\in\mathbb{R}^{n\times d}$, and $d < n$, it is faster
to compute the Vendi score using the covariance matrix,
$\frac{1}{n} \sum_i x_i x_i^{\top}$:
```python
vendi.score_dual(X)
```
If the rows of $X$ are not normalized, set `normalize = True`.


### Similarity functions

Some similarity functions are provided in `vendi_score.image_utils`, `vendi_score.text_utils`, and `vendi_score.molecule_utils`. For example:

Images:
```python
from torchvision import datasets
from vendi_score import image_utils

mnist = datasets.MNIST("data/mnist", train=False, download=True)
digits = [[x for x, y in mnist if y == c] for c in range(10)]
pixel_vs = [image_utils.pixel_vendi_score(imgs) for imgs in digits]
# The default embeddings are from the pool-2048 layer of the torchvision
# Inception v3 model.
inception_vs = [image_utils.embedding_vendi_score(imgs, device="cuda") for imgs in digits]
for y, (pvs, ivs) in enumerate(zip(pixel_vs, inception_vs)): print(f"{y}\t{pvs:.02f}\t{ivs:02f}")

# Output:
# 0       7.68    3.45
# 1       5.31    3.50
# 2       12.18   3.62
# 3       9.97    2.97
# 4       11.10   3.75
# 5       13.51   3.16
# 6       9.06    3.63
# 7       9.58    4.07
# 8       9.69    3.74
# 9       8.56    3.43
```

Text:
```python
from vendi_score import text_utils

sents = ["Look, Jane.",
         "See Spot.",
         "See Spot run.",
         "Run, Spot, run.",
	 "Jane sees Spot run."]
ngram_vs = text_utils.ngram_vendi_score(sents, ns=[1, 2])
bert_vs = text_utils.embedding_vendi_score(sents, model_path="bert-base-uncased")
simcse_vs = text_utils.embedding_vendi_score(sents, model_path="princeton-nlp/unsup-simcse-bert-base-uncased")
print(f"N-grams: {ngram_vs:.02f}, BERT: {bert_vs:.02f}, SimCSE: {simcse_vs:.02f}")

# N-grams: 3.91, BERT: 1.21, SimCSE: 2.81
```

More examples are illustrated in Jupyter notebooks in the `examples/` folder.

## Citation
```bibtex
@article{friedman2022vendi,
  title={The Vendi Score: A Diversity Evaluation Metric for Machine Learning},
  author={Friedman, Dan and Dieng, Adji Bousso},
  journal={arXiv preprint arXiv:2210.02410},
  year={2022}
}
```

```bibtex
@article{pasarkar2023cousins,
      title={Cousins Of The Vendi Score: A Family Of Similarity-Based Diversity Metrics For Science And Machine Learning}, 
      author={Pasarkar, Amey P and Dieng, Adji Bousso},
      journal={arXiv preprint arXiv:2310.12952},
      year={2023},
}
```