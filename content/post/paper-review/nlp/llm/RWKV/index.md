+++
author = "Kurt"
title = "RWKV"
date = "2024-03-25"
description = "Reinventing RNNs for the Transformer Era"
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

Transformer는 NLP 작업을 변화시켰으나 시퀀스 길이에 따라 복잡성이 증가하는 문제가 있고, RNN은 선형 확장성은 있으나 transformer만큼의 성능을 내기 어렵다. 이에 transformer의 병렬 학습과 RNN의 효율적 추론을 결합한 새로운 모델, Receptance Weighted Key Value(RWKV)를 제안한다.

이 연구의 방식은 linear attention 메커니즘을 통해 모델을 transformer나 RNN으로 구현할 수 있게 하여, 학습 시 병렬 계산과 추론 시 일정한 복잡성을 유지한다. 14B 개 parameter로 확장된 이 모델은 역대 가장 큰 RNN이며, RWKV는 비슷한 크기의 transformer와 동등한 성능을 보여준다. 이는 계산 효율과 성능 균형을 맞추는데 있어 중요한 진전이다.

---

## Introduction




---

## Reference

* [Paper](https://arxiv.org/pdf/2305.13048.pdf)
* [GitHub](https://github.com/BlinkDL/RWKV-LM)