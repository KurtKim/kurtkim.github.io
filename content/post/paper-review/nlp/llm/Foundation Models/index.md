+++
author = "Kurt"
title = "Foundation Models"
date = "2024-01-01"
description = "On the Opportunities and Risks of Foundation Models"
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

AI는 넓은 범위의 데이터에서 학습된 다양한 모델(BERT, DALL-E, GPT-3 등)이 등장하며 패러다임 변화를 겪고 있다. 이런 모델들을 "foundation model"이라 부르며, 그들의 중요성과 불완전성을 강조한다. 이 모델들은 많은 작업에 효과적이어서 동질화를 촉진하지만, 그 결함은 모든 하위 모델에 상속되므로 주의가 필요하다. foundation model의 작동 방식, 실패 시점, 그리고 가능성에 대한 명확한 이해는 아직 부족하며, 이에 대한 연구는 그들의 사회기술적 성질에 맞게 깊은 학제간 협력을 필요로 한다.

---

## Introduction

이 보고서는 "foundation model"이라는 일반적인 AI 모델 클래스를 기반으로 하는 시스템 구축에 대한 새로운 패러다임을 조사한다. foundation model은 광범위한 데이터에서 학습되어 다양한 downstream task에 적용될 수 있다. 이는 기존의 deep neural network와 self-supervised 학습에 기반하지만, 그 규모와 범위는 우리의 상상력을 넘어섰다. 예를 들어, GPT-3 같은 모델은 막대한 parameter를 가지고 있고, 특정 작업에 대한 명확한 학습 없이도 다양한 작업을 수행할 수 있다. 그러나, 이런 모델들의 특성은 잘 이해되지 않았고, 잠재적인 해를 악화시킬 수 있다. 이러한 모델들이 널리 배포될 예정이므로, 심도 있는 조사가 필요하다.

### Emergence and homogenization

foundation model의 중요성은 "emergence"과 "homogenization" 두 가지 개념으로 요약된다. "emergence"는 시스템 행동이 명시적으로 구축되지 않고 암시적으로 유도되는 것을 의미하며, 이는 흥분과 예상치 못한 결과에 대한 불안을 가져온다. "homogenization"는 다양한 응용 분야에서 기계 학습 시스템 구축 방법론을 통합하는 것을 가리키며, 많은 작업에 대한 leverage를 제공하지만 단일 실패 지점을 만들 수 있다. 이 두 개념은 최근 30년 동안 AI 연구에서 점점 중요해지고 있다.

![](images/figure1.png)

**Machine learning.** 오늘날 대부분의 AI 시스템은 과거 데이터를 학습하여 미래를 예측하는 기계 학습에 의해 구동된다. 1990년대부터 시작된 AI에서의 기계 학습의 부상은 작업 해결 방법을 명시하는 대신 데이터를 바탕으로 학습 알고리즘이 이를 유도하는 새로운 방식을 나타냈다. 이는 "how"가 학습 과정에서 발생하는 것을 의미한다. 또한, 기계 학습은 로지스틱 회귀와 같은 일반적인 학습 알고리즘을 통해 다양한 응용 분야를 통합하는 동질화로 나아가는 한 걸음을 나타냈다.

AI에서 기계 학습이 널리 사용되지만, 자연어 처리나 컴퓨터 비전 등의 복잡한 작업에서는 도메인 전문가가 "feature engineering"을 수행해야 했다. 이는 raw 데이터를 더 고수준의 특징으로 변환하는 과정으로, 기계 학습 방법에 더 적합하게 만드는 역할을 한다.

**Deep learning.** 2010년경, 딥러닝이라는 이름 아래에 deep neural network의 부활이 이루어졌다. 이는 더 큰 데이터셋, 더 많은 계산 능력, 그리고 대담함에 의해 추진되었다. raq 입력에 대한 학습을 통해 고수준의 특징이 발생하였고, 이는 성능 향상을 가져왔다. 딥러닝은 homogenization를 향한 또 다른 전환을 나타냈는데, 맞춤형 feature engineering 대신 동일한 deep neural network 아키텍처가 다양한 응용 프로그램에 사용될 수 있게 되었다.

**Foundation models.**

---

## Reference

* [Paper](https://arxiv.org/pdf/2108.07258.pdf)