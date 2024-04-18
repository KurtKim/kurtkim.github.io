+++
author = "Kurt"
title = "QLoRA"
date = "2024-04-20"
description = "Efficient Finetuning of Quantized LLMs"
categories = [
    "Paper Review"
]
tags = [
    "LLM",
    "Quantization",
]
draft = true
+++

## Abstract

QLoRA는 단일 48GB GPU에서 65B parameter 모델을 효율적으로 미세 조정 할 수 있는 방법으로, 메모리 사용량을 줄이면서도 전체 16비트 미세 조정 성능을 유지한다. 이 방법은 동결된 4비트 양자화 사전 학습 모델을 통해 gradient를 LoRA로 backpropagate 한다. Guanaco 모델은 Vicuna 벤치마크에서 기존 모델들을 능가하며, 단일 GPU를 사용해 24시간만에 ChatGPT의 99.3% 성능에 도달한다. QLoRA는 성능 저하 없이 메모리를 절약하는 여러 혁신을 도입했으며, 1,000개 이상의 모델을 미세 조정 해 다양한 데이터셋과 모델 규모에서 상세한 분석을 제공한다. 이 연구는 작은 고품질 데이터셋에서 최신 기술 결과를 달성할 수 있음을 보여주며, GPT-4 평가가 인간 평가의 합리적 대안임을 보여준다. 또한, 현재 챗봇 벤치마크의 신뢰성 문제를 지적하고, 모든 모델과 코드를 공개한다.

---

## Introduction




---

## Reference

* [Paper](https://arxiv.org/pdf/2305.14314.pdf)
* [GitHub](https://github.com/artidoro/qlora)