+++
author = "Kurt"
title = "RWKV"
date = "2024-03-25"
description = "Reinventing RNNs for the Transformer Era"
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

Transformer는 NLP 작업을 변화시켰으나 시퀀스 길이에 따라 복잡성이 증가하는 문제가 있고, RNN은 선형 확장성은 있으나 transformer만큼의 성능을 내기 어렵다. 이에 transformer의 병렬 학습과 RNN의 효율적 추론을 결합한 새로운 모델, Receptance Weighted Key Value(RWKV)를 제안한다.

이 연구의 방식은 linear attention 메커니즘을 통해 모델을 transformer나 RNN으로 구현할 수 있게 하여, 학습 시 병렬 계산과 추론 시 일정한 복잡성을 유지한다. 14B 개 parameter로 확장된 이 모델은 역대 가장 큰 RNN이며, RWKV는 비슷한 크기의 transformer와 동등한 성능을 보여준다. 이는 계산 효율과 성능 균형을 맞추는데 있어 중요한 진전이다.

---

## Introduction

딥러닝은 자연어 이해, 대화형 AI, 시계열 분석 등 복잡한 순차 데이터 처리에 혁신을 가져왔다. 이 분야에서 RNN과 transformer는 주요 기술로 활용되고 있으며, RNN은 긴 시퀀스 처리에 유리하나 그래디언트 소실 문제와 학습의 비병렬성으로 인한 확장성 제한이 단점이다.

transformer는 NLP에서 병렬화된 학습과 의존성 처리에 탁월함을 보이며, 모델들이 이 기술의 잠재력을 증명하였다. 그러나 self-attention 메커니즘의 복잡도는 긴 시퀀스 처리에 어려움을 주며, 이는 확장성 개선 연구로 이어지고 있다.

![](images/table1.png)

RWKV 모델은 RNN과 transformer의 장점을 결합하고 주요 단점을 피하며, 메모리 병목 현상과 transformer의 2차 스케일링 문제를 효율적인 linear 스케일링으로 완화한다. RWKV는 전통적인 dot-product 토큰 상호작용 대신 더 효과적인 channel-directed attention을 사용하는 linear attention의 변형을 통해 attention 메커니즘을 재정의한다. 이 구현은 근사치 없이 최소한의 계산 및 메모리 복잡성을 제공한다.

RWKV는 계산 효율성과 표현력 사이의 균형을 목표로 하며, 대규모 모델을 효과적으로 처리하여 낮은 계산 비용으로 경쟁력 있는 성능을 제공한다. 이는 AI의 확장성과 순차 데이터 처리의 도전을 해결하며, 더 지속 가능하고 효율적인 AI 모델로의 발전을 지향한다.

이 논문에서의 기여는 다음과 같다:

* RNN과 transformer의 장점을 결합하고 한계를 완화하는 새로운 구조인 RWKV의 도입.
* 대규모 모델을 위한 벤치마크 데이터셋에서 RWKV의 성능과 효율성을 입증하는 상세한 실험.
* Pile에서 학습된 169M에서 14B 개의 paramter를 가진 사전 학습된 모델의 공개.

---

## Background



---

## Reference

* [Paper](https://arxiv.org/pdf/2305.13048.pdf)
* [GitHub](https://github.com/BlinkDL/RWKV-LM)