+++
author = "Kurt"
title = "WebGPT"
date = "2024-01-09"
description = "Browser-assisted question-answering with human feedback"
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

웹 검색과 탐색 기능을 갖춘 GPT-3를 미세 조정하여 장문의 질문에 대답하도록 했다. 인간이 수행 가능한 작업 설정을 통해 모델을 학습시키고, 인간의 피드백으로 답변 품질을 최적화하였다. 사실 확인을 위해 모델은 브라우징 중 참조 정보를 수집한다. 이 방식은 Reddit의 질문 데이터셋 ELI5에서 효과적이었으며, 최적의 모델은 인간의 선호도를 예측하는 보상 모델을 통해 얻어졌다. 이 모델의 답변은 인간 평가자와 Reddit의 최고 투표 답변에 비해 각각 56%, 69%의 경우에 선호된다.

---

## Introduction

---

## Reference

* [Paper](https://arxiv.org/pdf/2112.09332.pdf)