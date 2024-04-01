+++
author = "Kurt"
title = "LLM-Adapters"
date = "2024-04-05"
description = "An Adapter Family for Parameter-Efficient Fine-Tuning of
Large Language Models"
categories = [
    "Paper Review"
]
tags = [
    "NLP",
    "PEFT",
]
draft = true
+++

## Abstract

GPT-4 및 ChatGPT와 같은 대규모 언어 모델(LLMs)의 성공에 힘입어, 특정 과제 데이터나 지시 데이터로 LLMs를 미세 조정하여 비용 효율적이고 접근성 높은 대안들이 개발되었다. adapter 기반의 parameter-efficient fine-tuning(PEFT)은 이 중에서도 전체 모델 대신 몇몇 parameter만 조정하여 우수한 성능을 내는 매력적인 방법이다. 본 논문은 LLM에 다양한 adapter를 통합하고, 이를 이용해 다른 과제에 적용할 수 있는 LLMAdapters 프레임워크를 제시한다. 이를 통해 adapter의 유형, 배치, hyper-parameter의 최적 설계를 연구하고, 산술 추론과 상식 추론 과제에서 뛰어난 성능을 입증하였다. 결과적으로, 소규모 LLMs에서도 adapter 기반 PEFT를 사용하여 몇 가지 추가 parameter만으로도 강력한 LLMs와 동등하거나 우수한 성능을 보여주었다.

---

## Introduction



---

## Reference

* [Paper](https://arxiv.org/pdf/2304.01933.pdf)
* [GitHub](https://github.com/AGI-Edgerunners/LLM-Adapters)