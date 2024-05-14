+++
author = "Kurt"
title = "Jamba"
date = "2024-05-20"
description = "A Hybrid Transformer-Mamba Language Model"
categories = [
    "Paper Review"
]
tags = [
    "LLM",
    "Milestone",
]
draft = true
+++

## Abstract

Jamba는 새로운 하이브리드 Transformer-Mamba mixture-of-experts(MoE) 아키텍처를 기반으로 한 대규모 언어 모델이다. 이 아키텍처는 Transformer와 Mamba layer를 교차 배치하고, 일부 레이어에 MoE를 추가하여 모델 용량을 확장하면서도 활성 파라미터 사용을 효율적으로 관리한다. 단일 80GB GPU에 맞는 강력한 모델로, 높은 처리량과 작은 메모리 사용량을 제공하며 표준 벤치마크와 긴 컨텍스트 평가에서 최신 성능을 달성한다. 최대 256K 토큰 컨텍스트 길이에서 강력한 결과를 보여주며, 아키텍처의 결정적인 요소를 연구하고 이 아키텍처의 흥미로운 특성을 공유할 계획이다. 

---

## Introduction



---

## Reference

* [Paper](https://arxiv.org/pdf/2403.19887)
* [HuggingFace](https://huggingface.co/ai21labs/Jamba-v0.1)