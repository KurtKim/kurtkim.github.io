+++
author = "Kurt"
title = "Pythia"
date = "2024-03-16"
description = "A Suite for Analyzing Large Language Models Across Training and Scaling"
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

Pythia는 70M에서 12B parameter까지 다양한 크기의 16개 거대 언어 모델(LLM)을 포함한 도구 모음으로, 이 모델들은 모두 동일한 순서로 공개 데이터에 대해 학습되었다. 이 모델들의 학습 과정과 진화를 탐구하고자 하며, 각 모델에 대해 154개 체크포인트와 정확한 학습 데이터 로더를 공개적으로 제공한다. Pythia는 기억력, 단어 빈도의 few-shot 성능 영향, 성별 편향 감소 등 여러 연구 분야에서 새로운 발견을 가능하게 하기 위해 설계되었다. 이는 LLM의 학습 역학에 대한 새로운 통찰을 얻기 위한 엄격하게 통제된 설정을 제공한다.

---

## Introduction

최근 몇 년 간, 대규모 transformer 모델들이 자연어 처리를 비롯한 다양한 분야에서 생성적 과제의 선두주자로 자리매김하였다. 텍스트-이미지 합성, 단백질 모델링, 컴퓨터 프로그래밍 등에서 큰 성공을 거두었음에도, 이 모델들의 성공 원인과 방법에 대해서는 아직 명확히 알려진 바가 적다.

transformer 모델이 학습과 스케일링에 따라 어떻게 변화하는지 이해하는 것은 중요하다. 이 모델들이 커질 때 나타나는 규칙적인 패턴은 잘 알려져 있지만, 이러한 스케일링 법칙과 모델의 학습 과정을 연결 짓는 연구는 많지 않다. 이 연구 부족은 적절한 모델을 테스트할 수 있는 자원의 부족 때문인데, 많은 대규모 언어 모델들이 공개되어 있음에도 불구하고 연구자들의 필요를 충족시키지 못하였다. 이런 연구는 대부분 비공개 모델에서 이루어져, 과학 연구를 위한 공개 모델 스위트의 필요성을 강조한다.

이 논문에서는 과학 연구를 위해 특별히 설계된 70M부터 12B parameter의 decoder-only autoregressive 언어 모델인 Pythia를 소개한다. Pythia는 세 가지 핵심 특성을 갖춘 유일한 공개 대규모 언어 모델이다.

1. 모델은 여러 크기의 스케일을 아우른다.
2. 모든 모델은 동일한 데이터에 대해 같은 순서로 학습되었다.
3. 데이터와 중간 체크포인트는 공부를 위해 공개적으로 이용 가능하다.




---

## Reference

* [Paper](https://arxiv.org/pdf/2304.01373.pdf)
* [GitHub](https://github.com/EleutherAI/pythia)