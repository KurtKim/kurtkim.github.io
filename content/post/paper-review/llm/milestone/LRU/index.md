+++
author = "Kurt"
title = "LRU"
date = "2024-03-11"
description = "Resurrecting Recurrent Neural Networks for Long Sequences"
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

Recurrent Neural Networks(RNNs)는 긴 시퀀스에서 빠른 추론을 제공하지만 학습 속도가 느리고 최적화가 어렵다. 최근 Deep state-space models(SSMs)은 긴 시퀀스 모델링에서 뛰어난 성능과 빠른 훈련 속도를 보여주었다. 이 논문에서는 표준 신호 전파 기법을 통해 깊은 RNN의 성능을 개선하고 학습 속도를 높일 수 있음을 보여준다. 이를 위해 RNN의 회귀를 선형화하고 대각화하며, 매개변수화와 초기화를 개선하고, 순방향 전파를 적절히 정규화하는 방법을 사용하였다. 결과적으로 Long Range Arena 벤치마크에서 SSMs와 동일한 성능과 효율성을 갖춘 'Linear Recurrent Unit'을 제안한다.

---

## Introduction

순환 신경망(RNNs)은 딥러닝 초기부터 순차적 데이터 모델링에 중요한 역할을 해왔으나, 기울기 소실 및 기울기 폭주 문제로 인해 학습이 어렵고 장기 종속성 학습에 한계가 있다. 이를 해결하기 위해 LSTM, GRU 등 여러 기술이 개발되었지만, 여전히 계산이 순차적이어서 최적화가 느리고 확장하기 어렵다.

최근 Transformer 모델은 순차 데이터 처리에서 큰 성공을 거두었으며, RNNs보다 학습 시 확장성과 병렬화가 용이하다. 그러나 Transformer는 계산 및 메모리 비용이 시퀀스 길이에 따라 제곱적으로 증가하기 때문에 긴 시퀀스에 대해 비용이 많이 들 수 있다. 반면, RNNs는 시퀀스 길이에 선형적으로 비례하여, 비교적 짧은 시퀀스에서는 추론 속도가 더 빠를 수 있다.

Gu et al. (2021a)은 S4 모델을 소개하였다. S4는 deep state-space model(SSM)로, Long Range Arena (LRA) 벤치마크에서 뛰어난 성능을 보인다. S4는 연속 시간 선형 상태공간 모델에서 영감을 받았으며, attention 레이어의 $O(L^2)$ 병목 현상을 숨겨진 상태를 통해 극복한다. 이 모델은 RNN처럼 레이어를 전개하여 추론 시 효율적이며, 시간 차원에서 선형적이어서 학습 중 병렬화가 용이하다.

S4 모델은 추론 시 RNN과 유사하지만, 학습 중에는 미분 방정식의 연속 시간 시스템을 이산화하여 매개변수화하고, 다항식 투영 이론에 기반한 초기화를 사용한다. 그러나 후속 연구에서는 이러한 초기화가 성능에 필수적이지 않으며, 이론과 다른 이산화 규칙이 더 나은 성능을 낼 수 있다고 제안하였다. 따라서, S4의 독특한 특성과 그 단순화 가능성은 아직 명확하지 않다.

RNN과 deep state-space model(SSMs)의 유사성을 바탕으로, 긴 거리 추론을 위한 깊은 아키텍처에서 RNN의 성능과 한계를 연구하여 그 기본 메커니즘을 이해하는 것이 이 연구의 주요 목표이다.

> "deep state-space model(SSMs)의 성능과 효율성을 deep RNN으로 맞출 수 있을까?""

기본 RNN에 작은 변화를 주어 S4와 같은 deep state-space model의 성능과 효율성을 달성할 수 있음을 보여준다. 이 새로운 RNN 모델을 'Linear Recurrent Unit' 또는 줄여서 LRU라고 부른다.

**Main Steps.** 이 논문에서는 성능이 뛰어나고 효율적인 RNN 모델을 만드는 주요 단계를 설명하며, 새로운 관점과 신중한 분석을 통해 깊은 RNN의 훈련과 초기화에서의 도전 과제와 최선의 방법을 제시한다.

* **Linear Recurrences.** SSM 레이어를 기본 RNN 레이어로 교체하면 LRA에서 성능이 크게 떨어지지만, 선형 회귀를 사용하면 테스트 정확도가 향상된다. 선형 RNN과 비선형 MLP 블록을 조합하면 비선형 시퀀스 매핑을 효과적으로 모델링할 수 있으며, 비선형성을 제거하면 기울기 문제를 제어하고 학습을 병렬화할 수 있다. 이러한 점이 깊은 SSM의 성공을 부분적으로 설명한다.

* **Complex Diagonal Recurrent Matrices.** 밀집 선형 RNN 레이어는 대각 형태로 재매개변수화할 수 있으며, 이는 학습 속도를 크게 향상시킨다. 이러한 접근은 이전 SSM들에서도 사용되었으며, 선형 RNN 레이어의 효율성도 개선한다.

* **Stable Exponential Parameterization.** 대각 순환 행렬에 지수 매개변수를 사용하면 학습 중 안정성을 보장하고 긴 거리 추론을 촉진할 수 있음을 보여준다. 초기화 시 순환 레이어의 고유값 분포가 긴 거리 추론의 성능을 결정한다.

* **Normalization.** 긴 거리 의존성이 있는 작업에서 순방향 전파의 활성화 정규화가 중요하다고 설명하며, 이 수정으로 RNN이 LRA 벤치마크에서 깊은 SSM과 같은 성능을 발휘할 수 있음을 보여준다. 또한, 이 정규화가 S4의 이산화 구조와 어떻게 연결되는지 설명한다.

이 논문에서는 깊은 선형 순환 유닛(LRU) 아키텍처와 각 단계의 성능 영향을 요약한다. 주된 목적은 S4 모델의 성능을 초월하는 것이 아니라, 적절히 초기화된 단순 RNN이 긴 거리 추론에서 강력한 성능을 발휘할 수 있음을 보여주는 것이다.

---

## Preliminaries

RNNs과 SSMs의 주요 구성 요소를 비교하고, 방법론과 실험 설정을 설명한다.

### Recap of recurrent block structures


---

## Reference

* [Paper](https://arxiv.org/pdf/2303.06349)
* [Github](https://github.com/Gothos/LRU-pytorch)