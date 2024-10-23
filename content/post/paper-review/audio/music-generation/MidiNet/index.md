+++
author = "Kurt"
title = "MidiNet"
date = "2024-09-30"
description = "A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation"
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

대부분의 음악 생성 신경망 모델은 순환 신경망(RNN)을 사용하지만, DeepMind의 WaveNet 모델은 합성곱 신경망(CNN)도 현실적인 음악 파형을 생성할 수 있음을 보여준다. 이에 따라 CNN을 사용해 상징적 도메인에서 멜로디를 한 마디씩 생성하는 방법을 조사하였다. 판별기를 포함한 생성적 적대 신경망(GAN)을 통해 멜로디 분포를 학습하며, 코드 진행이나 이전 마디의 멜로디를 조건으로 사용할 수 있는 새로운 조건부 메커니즘을 제안한다. 결과적으로 생성된 MidiNet 모델은 여러 MIDI 채널을 지원하며, 사용자 연구를 통해 MidiNet과 Google의 MelodyRNN 모델이 생성한 8마디 멜로디를 비교한 결과, MidiNet의 멜로디가 더 흥미롭고 쾌적하다는 평가를 받았다.

---

# Introduction


---

## Reference

* [Paper](https://arxiv.org/pdf/2208.08706)
* [GitHub](https://github.com/RichardYang40148/MidiNet)