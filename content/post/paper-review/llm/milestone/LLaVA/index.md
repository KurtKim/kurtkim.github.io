+++
author = "Kurt"
title = "LLaVA"
date = "2024-09-10"
description = "Visual Instruction Tuning"
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

기계 생성 지침 데이터를 사용해 대형 언어 모델(LLM)을 지침 조정하는 새로운 접근 방식을 제시한다. GPT-4를 이용해 멀티모달 언어-이미지 지침 따르기 데이터를 생성하고, 이를 기반으로 LLaVA: Large Language and Vision Assistant 모델을 개발하였다. LLaVA는 비전 인코더와 LLM을 연결한 멀티모달 모델로, 다양한 도전적인 작업을 평가할 수 있는 두 개의 벤치마크를 구성하였다. 실험 결과, LLaVA는 멀티모달 GPT-4와 유사한 성능을 보이며, GPT-4와 비교해 85.1%의 상대 점수를 기록하였다. Science QA에서는 92.53%의 최고 정확도를 달성하였다.

---



---

## Reference

* [Paper](https://arxiv.org/pdf/2304.08485)
* [Github](https://github.com/haotian-liu/LLaVA)