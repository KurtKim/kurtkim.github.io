+++
author = "Kurt"
title = "Direct Preference Optimization"
date = "2024-03-28"
description = "Your Language Model is Secretly a Reward Model"
categories = [
    "Paper Review"
]
tags = [
    "NLP",
    "LLM",
]
+++

## Abstract

이 논문에서는 대규모 unsupervised 언어 모델(LM)의 행동을 인간의 선호도에 맞추어 정밀하게 제어하는 새로운 방법, Direct Preference Optimization(DPO)를 소개한다. 기존의 복잡하고 불안정한 reinforcement learning from human feedback(RLHF) 대신, DPO는 보상 모델의 새로운 parameterization를 통해 단순한 분류 손실만으로 LM을 미세 조정할 수 있게 하여, 안정적이고 계산적으로 가벼운 방법을 제공한다. 실험 결과, DPO는 기존 방법들을 뛰어넘거나 일치하는 수준으로 인간의 선호도에 부합하게 LM을 미세 조정하며, 특히 생성물의 감정 제어와 요약, 단일 턴 대화의 응답 품질에서 우수한 성능을 보여준다.

---

## Introduction





---

## Reference

* [Paper](https://arxiv.org/pdf/2305.13048.pdf)
* [GitHub](https://github.com/BlinkDL/RWKV-LM)