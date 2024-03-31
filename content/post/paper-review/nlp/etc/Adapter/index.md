+++
author = "Kurt"
title = "Adapter"
date = "2024-04-02"
description = "Towards a Unified View of Parameter-Efficient Transfer Learning"
categories = [
    "Paper Review"
]
tags = [
    "NLP",
    "PEFT",
]
draft = true
+++

## Abstract

이 논문은 사전 학습된 언어 모델의 parameter-efﬁcient 전이 학습 방법을 분석하고, 이들 간의 연결을 확립하는 통합 프레임워크를 제시한다. 특정 숨겨진 상태를 수정하는 방식으로 다양한 방법을 재정의하고, 중요한 설계 차원을 정의한다. 종합적인 경험적 연구를 통해 중요한 설계 선택을 식별하고, 이를 통해 이전 방법보다 더 적은 parameter를 사용하면서도 더 효과적인 새로운 미세 조정 방법을 개발했으며, 이는 모든 parameter를 미세 조정한 결과와 비교할 수 있는 성능을 보여준다.

---

## Introduction

사전 학습된 언어 모델(PLMs)을 이용한 전이 학습은 자연어 처리 분야에서 주류가 되었으며, 다양한 작업에서 뛰어난 성능을 보여준다. 가장 일반적인 접근 방식은 모델의 전체 parameter를 미세 조정하는 것이지만, 이는 각 작업마다 별도의 모델 parameter 사본을 생성해야 하므로, 많은 작업을 수행하는 데에 많은 비용이 들게 된다. 특히, PLMs의 parameter 수가 수백만에서 수조에 이르기까지 증가함에 따라 이 문제는 더욱 심각해지고 있다.

![](images/figure1.png)

사전 학습된 언어 모델의 parameter를 대부분 동결하고 소수의 추가 parameter만 업데이트하는 가벼운 튜닝 방법들이 제안되었다. 이러한 방법으로는 adapter tuning, preﬁx 및 prompt tuning, 그리고 ow-rank matrice 학습 등이 있으며, 이들은 모델 매개변수의 1% 미만을 업데이트하면서도 전체 미세 조정과 비슷한 성능을 보여주었다. 이런 접근법은 parameter를 절약하고, 빠르게 새로운 작업에 적응할 수 있게 하며, 분포 외 평가에서의 견고성을 높인다.

이 논문은 parameter-efﬁcient tuning 방법들의 성공 요인과 이들 간의 연결성이 명확히 이해되지 않음을 지적하며, 세 가지 주요 질문을 탐구한다: (1) 이 방법들은 어떻게 연결되어 있는가, (2) 공유하는 핵심 설계 요소는 무엇인가, (3) 효과적인 요소들을 다른 방법으로 전달해 더 나은 변형을 만들 수 있는가?

이 연구에서는 adapter와 밀접하게 연결된 프리픽스 튜닝의 대안 형태를 제시하고, 동결된 PLMs의 숨겨진 표현을 수정하는 방식으로 구성된 통합 프레임워크를 개발하였다. 이 프레임워크는 수정 기능, 적용 위치, 통합 방법 등의 설계 차원을 따라 기존 방법들을 분해하고, 새로운 변형을 제안한다. 실험을 통해, 이 접근법은 기존의 parameter-efﬁcient tuning 방법보다 적은 parameter를 사용하면서, 네 가지 NLP 작업에서 전체 미세 조정과 동등한 성능을 달성함을 입증한다.

---

## Peliminaries




---

## Reference

* [Paper](https://arxiv.org/pdf/2110.04366.pdf)
* [GitHub](https://github.com/jxhe/unify-parameter-efficient-tuning)