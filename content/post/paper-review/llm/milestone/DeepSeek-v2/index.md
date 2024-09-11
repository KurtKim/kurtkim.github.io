+++
author = "Kurt"
title = "DeepSeek-V2"
date = "2024-05-10"
description = "A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
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

경제적인 학습과 효율적인 추론을 제공하는 Mixture-of-Experts (MoE) 언어 모델 DeepSeek-V2를 소개한다. 이 모델은 236B 파라미터를 가지며, 128K 토큰의 컨텍스트를 지원한다. 혁신적인 Multi-head Latent Attention (MLA)와 DeepSeekMoE 아키텍처를 채택하여 성능을 개선하고, 학습 비용을 42.5% 절감하며, KV 캐시를 93.3% 줄이고, 최대 생성 처리량을 5.76배 증가시킨다. 8.1T 토큰으로 사전 학습 후 Supervised Fine-Tuning (SFT)과 Reinforcement Learning (RL)을 진행하여 높은 성능을 보여준다.

---

## Introduction


---

## Reference

* [Paper](https://arxiv.org/pdf/2405.04434)
* [Github](https://github.com/deepseek-ai/DeepSeek-V2)