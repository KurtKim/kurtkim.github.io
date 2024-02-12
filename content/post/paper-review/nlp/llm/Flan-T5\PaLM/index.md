+++
author = "Kurt"
title = "Flan-T5/PaLM"
date = "2024-02-15"
description = "Scaling Instruction-Finetuned Language Models"
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

명령문 형태의 데이터셋을 사용한 언어 모델 미세 조정은 모델 성능 향상과 새로운 작업에 대한 일반화 능력을 향상시킨다. 이 논문에서는 작업 수, 모델 크기 확장, 생각의 흐름 데이터에 대한 미세 조정에 초점을 맞춘다. 이러한 방법은 다양한 모델 클래스, 프롬프팅 설정, 평가 벤치마크에서 성능을 크게 향상시킨다. 예를 들어, 1.8K 작업에 대해 미세 조정된 Flan-PaLM 540B는 평균적으로 9.4% 향상된 성능을 보인다. 또한, Flan-T5 체크포인트를 공개하여 대형 모델에 비해 강력한 성능을 보여준다. 결론적으로, instruction ﬁnetuning은 사전 학습된 언어 모델의 성능과 사용성을 향상시키는 일반적인 방법이다.

---

## Reference

* [Paper](https://arxiv.org/pdf/2210.11416.pdf)