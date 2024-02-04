+++
author = "Kurt"
title = "SoundStream"
date = "2024-02-6"
description = "An End-to-End Neural Audio Codec"
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

SoundStream이라는 새로운 neural audio codec은 음성, 음악, 일반 오디오를 효율적으로 압축할 수 있다. 이 codec은 fully convolutional encoder/decoder network와 residual vector quantizer로 구성되어 있으며, 학습 과정은 최근의 text-to-speech와 speech enhancement 기술을 활용한다. 이 모델은 3 kbps에서 18 kbps까지 다양한 비트레이트에서 작동할 수 있으며, 실시간 스마트폰 CPU에서 스트림 가능한 추론을 지원한다. 3 kbps의 SoundStream은 12 kbps의 Opus를 뛰어넘고, 9.6 kbps의 EVS에 근접한다. 추가적으로, 이 codec은 추가적인 지연 없이 압축과 향상을 동시에 수행할 수 있어, 배경 소음 억제 등의 기능도 가능하다.

---

## Introduction

---

## Reference

* [Paper](https://arxiv.org/pdf/2107.03312.pdf)
* [Github](https://github.com/wesbz/SoundStream)