+++
author = "Kurt"
title = "EnCodec"
date = "2024-02-09"
description = "High Fidelity Neural Audio Compression"
categories = [
    "Paper Review"
]
tags = [
    "Audio",
    "Codec"
]
draft = true
+++

## Abstract

neural network를 활용한 state-of-the-art real-time, high-ﬁdelity, audio codec을 소개한다. 이는 아티팩트를 효율적으로 줄이고 고품질 샘플을 생성하는 스트리밍 encoder-decoder 구조이다. 학습을 안정화하기 위해 loss balancer mechanism을 도입하였으며, lightweight Transformer 모델을 사용하여 얻은 표현을 최대 40%까지 더 압축하는 방법을 연구하였다. 이 모델은 말하기, 소음이 많은 반향성 말하기, 음악 등 다양한 오디오 도메인에서 우수한 성능을 보여주었다.

---

## Introduction


---

## Reference

* [Paper](https://arxiv.org/pdf/2210.13438.pdf)
* [Github](https://github.com/facebookresearch/encodec)