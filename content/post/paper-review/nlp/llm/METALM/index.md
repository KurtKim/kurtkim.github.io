+++
author = "Kurt"
title = "METALM"
date = "2024-02-08"
description = "Language Models are General-Purpose Interfaces"
categories = [
    "Paper Review"
]
tags = [
    "NLP",
    "LLM",
]
draft = true
+++

## Abstract

파운데이션 모델은 다양한 응용 분야에서 효과적이기 때문에 주목받고 있다. 이 연구에서는 언어 모델을 다양한 파운데이션 모델에 대한 일반적인 인터페이스로 사용하는 것을 제안한다. 이는 causal 모델링과 non-causal 모델링의 장점을 동시에 가져와,  bidirectional encoder의 사용으로 미세조정이 쉽고, 문맥 내 학습이나 지시 수행 등을 가능하게 한다. 실험 결과, METALM 모델은 미세조정, zero-shot 일반화, few-shot 학습 등에서 전문 모델들과 경쟁력을 가지거나 능가하는 성능을 보여주었다.

---

## Introduction: Design Principles

---

## Reference

* [Paper](https://arxiv.org/pdf/2206.06336.pdf)
* [Github](https://github.com/microsoft/unilm)