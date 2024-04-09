+++
author = "Kurt"
title = "LLM.int8()"
date = "2024-04-20"
description = "8-bit Matrix Multiplication for Transformers at Scale"
categories = [
    "Paper Review"
]
tags = [
    "NLP",
    "Quantization",
]
draft = true
+++

## Abstract

대규모 언어 모델의 GPU 메모리 요구를 줄이기 위해, transformer의 feed-forward 및 attention projection layer에 대한 Int8 행렬 곱셈 절차를 개발하였다. 이 방법은 추론에 필요한 메모리를 절반으로 줄이면서 전체 정밀도 성능을 유지한다. 175B parameter 모델을 Int8로 변환하여 성능 저하 없이 사용할 수 있음을 보여주며, 이를 위해 벡터별 양자화와 mixed-precision decomposition 방식을 포함하는 LLM.int8() 양자화 절차를 개발했하였다. 이 결과로 OPT-175B/BLOOM 같은 모델을 소비자 GPU를 갖춘 단일 서버에서 사용할 수 있게 되었다.

## Introduction




---

## Reference

* [Paper](https://arxiv.org/pdf/2208.07339.pdf)