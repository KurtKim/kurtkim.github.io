+++
author = "Kurt"
title = "Switch Transformers"
date = "2023-12-28"
description = "Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
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

딥러닝에서 대부분의 모델은 동일한 parameter를 재사용하지만, Mixture of Experts (MoE) 모델은 각 입력에 대해 다른 parameter를 선택한다. 이로 인해 많은 parameter를 가진 sparsely-activated 모델이 만들어지지만, 계산 비용은 일정하다. 그러나 복잡성과 통신 비용, 학습의 불안정성 때문에 MoE는 널리 적용되지 못하였다.

이 문제를 해결하기 위해 Switch Transformer를 도입했고, MoE 라우팅 알고리즘을 단순화하고 통신 및 계산 비용을 줄인 개선된 모델을 설계하였다. 이를 통해 처음으로 lower precision(bfloat16)로 large sparse 모델을 학습시킬 수 있었다.

이러한 개선을 통해, 동일한 계산 자원을 사용하면서 사전 학습 속도를 최대 7배 향상시켰고, 모든 101개 언어에서 mT5-Base 버전보다 성능을 향상시켰다. 또한, 최대 1조 개의 parameter 모델을 사전 학습하여 언어 모델의 규모를 확장하였고, T5-XXL 모델보다 4배 빠른 속도를 달성하였다.

---

## Introduction

---

## Reference

* [Paper](https://arxiv.org/pdf/2101.03961.pdf)
* [Github](https://github.com/kyegomez/SwitchTransformers)