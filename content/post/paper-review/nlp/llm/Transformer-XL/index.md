+++
author = "Kurt"
title = "Transformer-XL"
date = "2023-12-06"
description = "Attentive Language Models Beyond a Fixed-Length Context"
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

transformer는 long-term dependency를 학습할 수 있지만, 언어 모델링에서는 고정된 길이의 컨텍스트에 제한된다. 이를 해결하기 위해, 시간적 일관성을 해치지 않고 고정 길이를 넘어선 의존성을 학습할 수 있는 새로운 아키텍처인 Transformer-XL을 제안한다. 이는 세그먼트 수준의 재발 메커니즘과 새로운 위치 인코딩 체계를 포함하며, long-term dependency를 포착하고 컨텍스트 조각화 문제를 해결한다. 결과적으로 Transformer-XL은 RNN보다 80% 더 긴, 기본 transformer보다 450% 더 긴 의존성을 학습하고, 평가 시간에서 기본 transformer보다 최대 1,800+ 배 더 빠르며, 여러 텍스트 dataset에서 state-of-the-art를 달성하였다.

## Introduction

언어 모델링은 long-term dependency을 처리해야하는 중요한 과제로, 이는 RNN과 LSTM을 통해 다루어지곤 했다. 그러나 RNN 기반 모델들은 gradient vanishing과 explosion 등의 문제로 인해 최적화에 어려움이 있었다. LSTM은 평균적으로 200개의 문맥 단어를 사용하는데, 이는 아직 개선의 여지가 있다는 것을 보여준다.

attention mechanism은 장거리 단어 쌍 간의 직접적인 연결을 통해 최적화를 쉽게하고 long-term dependency를 학습하는 데 도움이 된다. 그러나 이 기법은 ﬁxed-length context를 가진 세그먼트에서만 작동하며, 이로 인해 long-term dependency를 충분히 포착하지 못하고, 문맥 파편화(context fragmentation)라는 문제를 야기한다. 이는 모델이 처음 몇 개의 기호를 잘 예측하는 데 필요한 문맥 정보가 부족하게 되어 최적화가 비효율적이고 성능이 떨어지게 된다.

ﬁxed-length context의 제한을 해결하기 위해, Transformer-XL이라는 새로운 아키텍처가 제안되었다. 이는 이전 세그먼트의 은닉 상태를 재사용하고, 이를 통해 세그먼트 간에 순환 연결을 만들어 long-term dependency를 모델링하게 된다. 또한, relative positional encodings을 사용하여 temporal confusion를 일으키지 않고 상태 재사용을 가능하게 한다. 이러한 접근법은 문맥 파편화 문제를 해결하고, 더 긴 주의 길이로 일반화할 수 있는 새로운 relative positional encodings 공식을 도입하였다.

Transformer-XL은 단어와 문자 수준 언어 모델링에서 뛰어난 결과를 보였다. 이 모델은 오직 100M 토큰만을 학습하여 상대적으로 일관된 긴 텍스트를 생성할 수 있다. 이 모델의 주요 기여는 reuse의 개념을 도입하고 새로운 positional encodings 방식을 개발한 것이다. 이 두 기술은 고정 길이 문맥 문제를 해결하기 위한 완벽한 솔루션을 제공하며, 이들 중 하나만으로는 충분하지 않다. Transformer-XL은 문자와 단어 수준 언어 모델링에서 RNN보다 더 나은 성능을 보여주는 첫 번째 self-attention 모델이다.

## Reference

* [Paper](https://arxiv.org/pdf/1901.02860.pdf)
* [Github](https://github.com/kimiyoung/transformer-xl)