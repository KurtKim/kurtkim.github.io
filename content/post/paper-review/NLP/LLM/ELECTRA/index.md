+++
author = "Kurt"
title = "ELECTRA"
date = "2024-01-10"
description = "Pre-training Text Encoders as Discriminators Rather Than Generators"
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

BERT와 같은 masked language modeling(MLM) 방법은 일부 토큰을 [MASK]로 바꿔 원래 토큰을 재구성하는 모델을 훈련하지만, 이를 효과적으로 하려면 많은 계산이 필요하다. 대신에, replaced token detection 라는 더 효율적인 사전 학습 작업을 제안한다. 이 방법은 일부 토큰을 가능성 있는 다른 토큰으로 바꾸는 방식으로 입력을 변형하고, 변형된 토큰의 원래 값을 예측하는 대신 각 토큰이 생성자 샘플로 대체되었는지 여부를 예측하는 모델을 학습시킨다. 이 방법은 모든 입력 토큰에 대해 작업을 정의하므로 MLM보다 효율적이다. 결과적으로, 이 방식으로 학습한 컨텍스트 표현은 동일한 모델 크기, 데이터, 계산량을 가진 BERT보다 월등히 높은 성능을 보여준다. 특히, 작은 모델에서는 GLUE 자연어 이해 벤치마크에서 GPT를 능가하는 모델을 한 GPU에서 4일 동안 훈련할 수 있다. 대규모에서도 이 방법은 RoBERTa와 XLNet의 성능에 비교할 수 있으며, 더 적은 계산량을 사용하면서도 동일한 계산량을 사용할 때 그 모델들을 능가한다.

## Reference

* [Paper](https://arxiv.org/pdf/2003.10555.pdf)
* [Code](https://github.com/google-research/electra)