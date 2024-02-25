+++
author = "Kurt"
title = "BLOOM"
date = "2024-02-29"
description = "A 176B-Parameter Open-Access Multilingual Language Model"
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

거대 언어 모델(LLM)은 새로운 작업을 수행할 수 있는 능력을 보여주었다. 그러나 대부분은 자원이 풍부한 조직에 의해 개발되고 비공개 상태였다. 이를 민주화하기 위해, 176B-parameter의 공개 접근 언어 모델 BLOOM을 소개한다. 이는 다양한 언어를 포함하는 ROOTS 말뭉치에서 학습되었으며, 다양한 벤치마크에서 경쟁력 있는 성능을 보였다. 이 모델과 코드를 공개함으로써 LLM을 사용한 미래의 연구와 응용을 촉진한다.

---

## Introduction

사전 학습된 언어 모델은 적은 양의 라벨 데이터로도 높은 성능을 내는 특성 때문에 현대 자연어 처리(NLP)의 핵심 요소가 되었다. 이런 모델들은 추가적인 학습 없이도 유용한 작업을 수행할 수 있다. 하지만, 이러한 모델의 학습 비용과 환경적 부담은 커서 대부분의 연구 커뮤니티가 이들의 개발에서 배제되었고, 대부분의 언어 모델은 주로 영어 텍스트에 대해 학습되었다.

수백 명의 연구자들이 협력하여 개발하고 공개한 BigScience Large Open-science Open-access Multilingual Language Model (BLOOM)을 제시한다. 이 모델은 46개의 자연 언어와 13개의 프로그래밍 언어에 대해 학습되었다. BLOOM을 구축하기 위해, 학습 데이터셋, 모델 아키텍처와 학습 목표, 그리고 분산 학습을 위한 엔지니어링 전략에 대한 철저한 설계 과정을 거쳤다. 이 논문의 목적은 BLOOM의 설계 단계에 대한 고수준 개요를 제공하는 것이다.

---

## Background

---

## Reference

* [Paper](https://arxiv.org/pdf/2211.05100.pdf)