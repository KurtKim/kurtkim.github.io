+++
author = "Kurt"
title = "Musika!"
date = "2024-09-20"
description = "Fast Infinite Waveform Music Generation"
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

Musika는 수백 시간의 음악을 단일 GPU에서 학습하고, 소비자 CPU에서 임의 길이의 음악을 실시간보다 빠르게 생성할 수 있는 시스템이다. adversarial  오토인코더로 스펙트로그램의 압축 표현을 학습한 후, 이 표현을 GAN으로 특정 음악 도메인에 맞게 학습한다. 잠재 좌표 시스템과 글로벌 컨텍스트 벡터를 사용하여 음악을 병렬로 생성하고 스타일을 유지한다. 정량적 평가와 사용자 제어 옵션을 제공하며, 코드와 사전 훈련된 가중치는 github.com/marcoppasini/musika에서 제공된다.

---



---

## Reference

* [Paper](https://arxiv.org/pdf/2208.08706)
* [GitHub](https://github.com/marcoppasini/musika)