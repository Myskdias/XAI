# Positional Artefacts in Masked Language Model Embeddings

This repository contains our XAI project on **positional artefacts in masked language model embeddings**, based on a partial reproduction of the paper *Positional Artefacts Propagate Through Masked Language Model Embeddings* by Luo, Kulmizev, and Mao (2021).

**Project report:** [read the report in the repository](./RapportXAI.pdf) · [download the PDF](https://raw.githubusercontent.com/Myskdias/XAI/main/RapportXAI.pdf)

## Overview

Pretrained transformer language models such as BERT produce contextualized token embeddings that are highly useful for downstream NLP tasks. However, these representations are not geometrically neutral: they often contain **outlier dimensions** that dominate vector norms and cosine similarity computations.

The original paper studied in this project argues that some of these outlier dimensions are not primarily semantic. Instead, they are linked to **positional information** and propagate through the transformer layers, partly due to the structure of BERT's embeddings and normalization layers.

This project reimplements and analyzes several of the paper's core experiments under a reduced computational setup. The goal is to verify whether the main qualitative phenomena can still be observed when using a smaller corpus and more limited compute.

## Project objective

The project focuses on the following questions:

- Do BERT hidden states contain a dominant outlier dimension?
- Is this dimension connected to positional information?
- Does removing this dimension change the geometry of the embedding space?
- Does this intervention improve semantic similarity or word sense discrimination?

The experiments are conducted with **bert-base-uncased** from Hugging Face Transformers. BERT is used as a fixed encoder: it is not fine-tuned. The analysis is performed directly on hidden states extracted from the model.

## Experimental setup

The main experiments use:

- **Model:** `bert-base-uncased`
- **Dataset for representation analysis:** SST-2 from GLUE
- **Dataset for word sense evaluation:** WiC / Word-in-Context
- **Frameworks:** PyTorch and Hugging Face Transformers
- **Representation size:** 768 dimensions
- **BERT depth:** embedding layer + 12 transformer layers

Due to compute constraints, the reproduction uses a reduced setup compared to the original paper. Most experiments are run on a cropped SST-2 subset, and the position probe is trained for fewer epochs than in the paper.

## Implemented experiments

### 1. Hidden state extraction

The first step is to extract contextualized token representations from BERT.

For each batch of sentences:

1. the input is tokenized with the BERT tokenizer;
2. BERT is run with `output_hidden_states=True`;
3. hidden states are collected for each layer;
4. padding tokens are filtered out;
5. valid token embeddings are flattened into matrices of shape `(N, d)`.

This extraction pipeline makes it possible to analyze individual embedding dimensions across tokens and layers.

### 2. Outlier dimension detection

For each token embedding, we identify the dimensions corresponding to the minimum and maximum activation values. We then count how often each dimension appears as the `argmin` or `argmax` across the dataset.

A dimension is considered an outlier when it appears as an extreme value for a disproportionately large fraction of tokens.

In our reproduction, we find a strong dominant outlier dimension: dimension **308** appears as the minimum activation in at least 80% of the sampled texts across the first layers. This differs from the exact dimension reported in the original paper, but the qualitative phenomenon is the same: one coordinate dominates the hidden representation space.

### 3. Position probing

To test whether BERT hidden states encode positional information, we train a simple linear probe. The probe receives a token embedding and predicts the position of the token in the input sequence.

The probe is trained independently on several layers. The results show that positional information is highly accessible in early layers and becomes weaker in deeper layers.

| Layer | Loss | Accuracy |
| ---: | ---: | ---: |
| 1 | 0.0135 | 0.999 |
| 3 | 0.0605 | 0.995 |
| 6 | 0.6714 | 0.774 |
| 9 | 1.3186 | 0.519 |
| 12 | 1.9153 | 0.373 |

This supports the interpretation that the outlier dimension is linked to structural information, especially token position, rather than to purely semantic content.

### 4. Clipping outlier dimensions

We then perform a direct intervention: the dominant outlier dimension is set to zero in the hidden representations. This operation is referred to as **clipping**.

The goal is to measure whether removing the outlier changes the geometry of the embedding space. We analyze three effects:

- anisotropy;
- self-similarity;
- word sense discrimination on WiC.

### 5. Effect on anisotropy

BERT embeddings are known to be anisotropic: unrelated token vectors can have high cosine similarity because they share a common dominant direction.

After clipping the outlier dimension, anisotropy decreases consistently across layers.

| Layer | Before clipping | After clipping | Difference |
| ---: | ---: | ---: | ---: |
| 1 | 0.1876 | 0.0969 | -0.0906 |
| 2 | 0.2451 | 0.0994 | -0.1457 |
| 3 | 0.2231 | 0.1128 | -0.1103 |
| 4 | 0.2471 | 0.1231 | -0.1240 |

The reduction is weaker than in the original paper, but the direction of the effect is consistent: clipping removes part of the common geometric bias in the embedding space.

### 6. Effect on self-similarity

Self-similarity measures whether occurrences of the same word in different contexts remain close in embedding space.

In our experiments, clipping the outlier dimension generally increases self-similarity. This suggests that the outlier dimension injects a structural component that can interfere with lexical comparison across contexts.

### 7. Word sense discrimination with WiC

The WiC task asks whether a target word has the same meaning in two different sentences. We evaluate this using cosine similarity between the contextualized representations of the target word.

Unlike the original paper, our reproduction does not show an improvement after clipping. Accuracy slightly decreases in our setup. This does not necessarily contradict the main interpretation, because the reported improvement in the original paper was small and our experiment uses a fixed threshold rather than an exhaustive threshold search.

The conclusion is therefore more nuanced: clipping has a clear geometric effect, but its downstream semantic benefit is limited and sensitive to evaluation details.

## Main findings

- BERT hidden states contain a dominant outlier dimension.
- The exact index of the outlier can differ from the original paper, but the phenomenon remains stable.
- Positional information is strongly accessible in early BERT layers and weaker in deeper layers.
- Clipping the outlier dimension reduces anisotropy.
- Clipping increases self-similarity in most tested settings.
- Clipping does not improve WiC accuracy in our reproduction.
- The outlier dimension appears to act more as a **structural positional bias** than as a directly semantic feature.

## Interpretation

The main interpretation is that a highly salient neuron in a transformer representation is not necessarily semantically meaningful.

In this project, the dominant outlier dimension seems to encode a systematic property of the input sequence, especially positional structure. Because this signal is shared across many tokens, it creates a common direction in representation space. This affects cosine similarity, anisotropy, and self-similarity measurements.

From an XAI perspective, this is important: representation analysis can be misleading if we interpret dominant dimensions as semantic features without first checking whether they encode structural artefacts.

## Limitations

This is a partial reproduction, not a full replication of the original paper.

The main limitations are:

- the original paper did not provide an implementation;
- the experimental setup had to be reimplemented from scratch;
- the compute budget was significantly smaller than in the original work;
- the SST-2 experiments were run on a reduced subset;
- the probing experiment used fewer epochs;
- the WiC experiment used a fixed threshold;
- the pretraining experiment was attempted only at much smaller scale and gave inconsistent results.

These limitations mainly affect quantitative comparison with the original paper. The main qualitative conclusions remain supported.

## Repository contents

The repository contains the code and experiments used for the reproduction. The implementation is organized around reusable source modules and a notebook that runs the experiments and plots the results.

The project report is available here:

```text
RapportXAI.pdf
```

## Reference

This project is based on:

```text
Luo, Ziyang, Artur Kulmizev, and Xiaoxi Mao. 2021.
Positional Artefacts Propagate Through Masked Language Model Embeddings.
Proceedings of EACL 2021.
```

Related references include BERT, RoBERTa, contextual representation anisotropy, and the WiC dataset.

## Conclusion

This project reproduces the main qualitative claim of Luo, Kulmizev, and Mao (2021): masked language model embeddings contain dominant outlier dimensions linked to positional artefacts. These dimensions strongly affect the geometry of BERT representations.

Removing the dominant outlier dimension makes the embedding space less anisotropic and increases self-similarity, but it does not necessarily improve downstream semantic discrimination. The result highlights the importance of separating structural artefacts from semantic information when interpreting transformer representations.
