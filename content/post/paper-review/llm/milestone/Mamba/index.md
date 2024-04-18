+++
author = "Kurt"
title = "Mamba"
date = "2024-04-20"
description = "Linear-Time Sequence Modeling with Selective State Spaces"
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

기초 모델, 주로 transformer 구조에 기반한 현재 딥러닝 응용 프로그램은 긴 시퀀스 처리의 계산 비효율성과 언어 같은 핵심 모달리티에서의 성능 문제를 해결하기 위해 여러 하위 2차 시간 구조를 개발하였다. 이러한 모델의 내용 기반 추론 능력 부족을 개선하기 위해, SSM parameter를 입력에 따라 조정하고 선택적 SSM을 주의나 MLP 블록 없는 Mamba라는 단순화된 신경망 구조에 통합하였다. Mamba는 빠른 추론 속도와 시퀀스 길이에 대한 선형 스케일링을 제공하며, 여러 모달리티에서 최고 수준의 성능을 달성하고, 언어 모델링에서는 비슷한 크기의 transformer를 능가한다.

---

## Introduction




---

## Reference

* [Paper](https://arxiv.org/ftp/arxiv/papers/2312/2312.00752.pdf)
* [GitHub](https://github.com/state-spaces/mamba)