+++
author = "Kurt"
title = "Symbolic Music Generation with Diffusion Models"
date = "2024-09-30"
description = ""
categories = [
    "Paper Review"
]
tags = [
    "Audio",
    "Music Generation",
]
draft = true
+++

## Abstract

점수 기반 생성 모델과 확산 확률적 모델은 연속 도메인에서 고품질 샘플을 생성하는 데 성공적이지만, 이산 기호 음악 데이터에는 제한적이었다. 이 연구에서는 변분 오토인코더의 연속 잠재 공간에서 이산 도메인을 매개변수화하여 확산 모델을 학습하는 방법을 제시한다. 이 방법은 비자기회귀적이며, 반복적인 정제 단계를 통해 병렬 생성이 가능하고, 기존의 자기회귀 언어 모델보다 우수한 생성 성능을 보여준다.

---

# Introduction



---

## Reference

* [Paper](https://arxiv.org/pdf/2103.16091)
* [GitHub](https://github.com/magenta/symbolic-music-diffusion)