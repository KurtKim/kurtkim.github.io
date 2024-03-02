+++
author = "Kurt"
title = "LLaMA"
date = "2024-01-18"
description = "Open and Efficient Foundation Language Models"
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

LLaMA는 7B에서 65B parameter의 기본 언어 모델 컬렉션이다. 이 모델들은 수조 개의 토큰에 대해 학습되었고, 공개적으로 사용 가능한 데이터셋만을 사용하여 최고 수준의 모델을 학습시킬 수 있음을 보여준다. 특히, LLaMA-13B는 대부분의 벤치마크에서 GPT-3를 능가하며, LLaMA-65B는 최고의 모델과 경쟁력이 있다. 이 모델들은 모두 연구 커뮤니티에 공개되었다.

---

## Introduction

거대 언어 모델(Large Languages Models, LLMs)은 텍스트 지시나 소수의 예제를 통해 새로운 작업을 수행할 수 있다. 이런 능력은 모델 규모를 충분히 확대할 때 나타났고, 이를 더 확대하려는 연구가 진행되고 있다. 하지만 최근 연구에서는 더 많은 parameter가 더 나은 성능을 가져다 주지 않는다는 것을 보여주었다. 오히려 주어진 컴퓨팅 예산 내에서 더 많은 데이터로 학습된 작은 모델이 최상의 성능을 보여주었다.

Hoffmann et al.의 연구는 학습 예산에 따라 데이터셋과 모델 크기를 어떻게 최적화할지에 초점을 맞추고 있다. 하지만 추론 예산을 고려하지 않았고, 이는 대규모 언어 모델을 서비스하는 데 중요하다. 특정 성능 목표가 있을 때, 학습 속도보다는 추론 속도가 더 빠른 모델이 선호되며, 큰 모델을 학습하는 것보다 작은 모델을 오래 학습하는 것이 추론에서 더 저렴하다는 것이 확인되었다. 10B 모델을 200B 토큰에서 훈련하는 것을 권장하지만, 1T 토큰 이후에도 7B 모델의 성능이 계속 향상되는 것을 발견하였다.

이 연구는 일반적으로 사용하는 것보다 더 많은 토큰으로 학습하여 다양한 추론 예산에서 최고의 성능을 달성하는 언어 모델, LLaMA를 개발했다. 이 모델은 7B에서 65B의 parameter를 가지며, 기존 최고의 언어 모델과 경쟁력이 있다. 예를 들어, 10배 작은 LLaMA-13B는 대부분의 벤치마크에서 GPT-3를 능가한다. 이 모델은 단일 GPU에서 실행될 수 있어 언어 모델의 접근성과 연구를 민주화(democratize)하는데 도움이 될 것이다. 또한, 65B parameter 모델은 최고의 거대 언어 모델과도 경쟁력이 있다.

이 연구는 공개적으로 이용 가능한 데이터만을 사용하여 Chinchilla, PaLM, GPT-3와 달리 오픈 소스와 호환성이 있다. 대부분의 기존 모델들은 공개적으로 이용 가능하지 않거나 문서화되지 않은 데이터에 의존한다. OPT, GPT-NeoX, BLOOM, GLM 등의 일부 예외가 있지만, 이들 중 어느 것도 PaLM-62B나 Chinchilla와 경쟁력이 없다.


---

## Reference

* [Paper](https://arxiv.org/pdf/2302.13971.pdf)
* [Github](https://github.com/facebookresearch/llama)