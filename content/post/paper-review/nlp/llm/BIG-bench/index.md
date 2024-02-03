+++
author = "Kurt"
title = "BIG-bench"
date = "2024-02-03"
description = "Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models"
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

언어 모델의 규모 증가는 양적 개선과 새로운 질적 능력을 동시에 가져온다. 이 새로운 능력들은 아직 잘 이해되지 않았지만, 그들의 잠재적인 영향력 때문에 중요하다. 미래 연구를 위해, 새로운 파괴적인 능력에 대비하고, 사회적으로 불이익한 효과를 완화하기 위해서는 현재와 가까운 미래의 언어 모델의 능력과 한계를 이해하는 것이 필수적이다.

the Beyond the Imitation Game benchmark(BIG-bench)는 현재 언어 모델의 능력을 넘어서는 다양한 작업으로 구성되어 있다. 이 벤치마크는 여러 기관의 저자들이 기여한 204개의 작업을 포함하며, 이 작업들은 언어학, 아동 발달, 수학, 상식 추론, 생물학, 물리학, 사회적 편향, 소프트웨어 개발 등 다양한 주제를 다룬다. 이 벤치마크를 사용하여 다양한 크기의 언어 모델을 평가하였고, 인간 전문 평가자 팀이 모든 작업을 수행하여 기준선을 제공하였다. 결과적으로, 모델 성능과 보정은 규모가 증가함에 따라 개선되었지만 절대적인 수치에서는 불만족스럽다. 또한, 모델 간 성능은 놀랍게도 유사하며, 희소성에서 이점을 얻었다. 그러나 사회적 편향은 모호한 맥락에서 규모와 함께 증가하는 경향이 있지만, 이는 프롬프팅으로 개선될 수 있다.

---

## Introduction

> *An important feature of a learning machine is that its teacher will often be very largely ignorant of quite what is going on inside.* (A.M. Turing, Computing Machinery and Intelligence, 1950)

생성적 언어 모델은 텍스트 시퀀스의 가장 적절한 연속 부분을 만드는 능력을 가지고 있다. 이 능력은 텍스트를 통해 설명하고 수행될 수 있는 모든 작업을 포함하므로, 이메일, 채팅, 웹 포럼 등에서의 문제 해결에도 사용될 수 있다.

최근 연구에서는 생성적 언어 모델이 더 크고 많은 데이터로 학습될수록 예측 가능한 방법으로 성능이 향상됨을 보여주고 있다. 이러한 발전에 따라, 언어 모델은 1 trillion 개 이상의 parameter로 확장되었고, 앞으로도 더욱 커질 것으로 예상되며, 아키텍처와 학습 방법의 개선을 통해 성능 향상도 계속될 것으로 보인다.

### Quantity has a quality all its own

---

## Reference

* [Paper](https://arxiv.org/pdf/2206.04615.pdf)
* [Github](https://github.com/google/BIG-bench)