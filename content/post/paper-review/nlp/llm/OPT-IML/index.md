+++
author = "Kurt"
title = "OPT-IML"
date = "2024-03-04"
description = "Scaling Language Model Instruction Meta Learning through the Lens of Generalization"
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

최근의 연구에서는 대규모 사전 학습된 언어 모델을 instruction-tuning하면 보이지 않는 작업에 대한 일반화 성능이 향상된다는 것이 확인되었다. 하지만, instruction-tuning 과정에서의 다양한 결정들이 성능에 어떤 트레이드오프를 가져오는지에 대한 이해는 아직 제한적이다. 이 논문에서는 각종 결정들이 언어 모델의 성능에 어떤 영향을 미치는지를 분석하고, 이를 바탕으로 OPT-IML 30B와 175B를 학습시켰다. 이 모델들은 다양한 작업과 입력 형식을 가진 네 가지 벤치마크에서 모두 뛰어난 일반화 성능을 보였다. 이 결과는 모든 벤치마크에서 OPT를 크게 능가하며, 특정 벤치마크에서 미세 조정된 기존 모델과도 경쟁력을 가진다. 이 연구의 결과와 평가 프레임워크는 공개되었다.

---

## Introduction



---

## Reference

* [Paper](https://arxiv.org/pdf/2212.12017.pdf)
