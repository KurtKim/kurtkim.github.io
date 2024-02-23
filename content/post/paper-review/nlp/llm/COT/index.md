+++
author = "Kurt"
title = "COT"
date = "2024-01-15"
description = "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
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

대형 언어 모델의 복잡한 추론 능력은 chain of thought, 즉 중간 추론 단계의 일련의 과정을 생성함으로써 크게 향상될 수 있다. 특히, 충분히 큰 언어 모델에서는 "chain of thought" 라는 간단한 방법을 통해 이러한 추론 능력이 자연스럽게 나타난다. 이 방법은 몇 가지 chain of thought를 프롬프팅의 예시로 제공하는 것이다.

세 가지 대형 언어 모델에 대한 실험은 사고의 연결 고리 프롬프팅이 산술, 상식, 심볼릭 추론 과제에서 성능을 향상시키는 것을 보여준다. 예를 들어, 여덟 가지 사고의 연결 고리 예시만을 사용해 PaLM 540B를 프롬프트하면, 수학 단어 문제 벤치마크인 GSM8K에서 최고의 정확도를 달성하며, 심지어 튜닝된 GPT-3를 뛰어넘는다.

---

## Introduction



---

## Reference

* [Paper](https://arxiv.org/pdf/2201.11903.pdf)