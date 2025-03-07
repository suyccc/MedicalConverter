# Converting Open-ended Questions to Multiple-choice Questions Simplifies Biomedical Vision-Language Model Evaluation

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![Pytorch](https://img.shields.io/badge/Pytorch-2.5-red.svg)](https://pytorch.org/get-started/previous-versions/#v25)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

[**🌐 Homepage**](https://suyccc.github.io/MedicalConverter-Website/) | [**🤗 Dataset**](https://huggingface.co/datasets/suyc21/MedicalConverter) | [**💎 Poster**](assets/ML4H_poster.pdf) | [**📖 PDF**](assets/224_Converting_Open_ended_Ques.pdf)

This repo provides the PyTorch source code and pdf version of our paper: [Converting Open-ended Questions to Multiple-choice Questions Simplifies Biomedical Vision-Language Model Evaluation](.) (Machine Learning for Health (ML4H) 2024).

## 🔮 Introduction

Vision-language models (VLMs) show promise in medicine, but their evaluation remains challenging due to their open-ended nature. Current metrics often fail to capture nuances in human judgment,  while model-based evaluations are computationally expensive and unstable. We propose converting open-ended questions into multiple-choice format to address these limitations. Using an agent-based framework with GPT-4, we transform questions through iterative refinement. Our results demonstrate strong correlation between multiple-choice and open-ended performance across three datasets. We evaluate 18 models on these converted datasets, showing improved capability discrimination. Case studies illustrate our approach’s success where rule-based evaluations fail. This work contributes a novel evaluation framework, aiming to enable easier and more consistent VLM evaluation in medicine.

<img src="assets/1.png"></img>
**Overview.** We discovered challenges in open-ended medical VQA evaluation and we chose converting to multi-choice format as an alternative solution. Given an open-ended format question, answer, and corresponding image, we aim to output three challenge distractors. Then, combine question, answer and distrators as a multiple-choice format question.

## 🛠️ Usage

Check out [main.py](main.py) for the implementation of MedicalConverter pipeline.

```
python main.py --api_key your_api_key --dataset_path your_dataset_path --output_path your_output_path
```

## 🤗 Dataset

Dataset is available at [Huggingface Datasets](https://huggingface.co/datasets/suyc21/MedicalConverter).

## 🎯 Citation

If you use this repo in your research, please cite it as follows:

```
@article{MedicalConverter 
title={Converting Open-ended Questions to Multiple-choice Questions Simplifies Biomedical Vision-Language Model Evaluation}, author={Yuchang Su and Yuhui Zhang and Yiming Liu and Ludwig Schmidt and Serena Yeung-Levy}, 
journal={ML4H 2024}, year={2024} }
```
