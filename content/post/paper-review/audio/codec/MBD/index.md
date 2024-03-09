+++
author = "Kurt"
title = "MBD"
date = "2024-03-10"
description = "From Discrete Tokens to High-Fidelity Audio Using Multi-Band Diffusion"
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

이 연구에서는 저비트율의 이산 표현에서 다양한 오디오 모달리티(예: 음성, 음악, 환경 소리)를 생성하는 고해상도 Multi-Band Diffusion 기반 프레임워크를 제안한다. 이 방법은 기존의 생성 모델이 완벽하지 않은 조건에서 audible artifact를 생성하는 문제를 해결하려고 한다. 제안된 접근법은 동일한 비트율에서 지각 품질 면에서 state-of-the-art의 생성 기술을 능가한다고 주장한다.

---

## Introduction

neural-based vocoder는 최신 neural network 아키텍처를 활용해 자연스러운 음색과 억양의 고품질 음성을 생성하는 데 뛰어난 성과를 보여주었다.

Self-Supervised Learning(SSL)은 음성 데이터에 적용되어 감정과 억양 등의 맥락적 정보를 담은 표현을 생성하며, 이로부터 파형 오디오를 만드는 것이 새로운 연구 주제가 되었다. 이 과정은 SSL을 사용해 오디오 표현을 학습하고, 그 후 GAN 접근법으로 음성을 디코딩하는 두 단계로 이루어진다. 이 방법들은 뛰어난 성능을 보이지만, 불안정하고 학습하기 어렵다는 단점이 있다.

압축 모델은 복원 손실을 활용하여 데이터의 의미 있는 표현을 학습하는 SSL 모델로 볼 수 있다. 이 모델들은 오디오 표현과 합성을 동시에 학습하는 과정에서 다양한 오디오 도메인을 모델링할 수 있다. 모델은 스펙트로그램 매칭, 특성 매칭, 그리고 다양한 적대적 손실 등 복잡하게 조합된 목표를 통해 최적화된다. 하지만 매우 낮은 비트율에서는 메탈릭한 목소리나 왜곡 같은 눈에 띄는 아티팩트가 추가될 수 있다.

모델 최적화 이후 학습된 표현은 다양한 오디오 작업에 활용될 수 있다. Kreuk et al. 은 텍스트를 통한 오디오 생성을, Wang et al. 은 zero-shot 텍스트-음성 변환을 제안하였다. 또한, Agostinelli et al. 은 텍스트-음악 생성에, Hsu et al. 은 조용한 비디오에서 음성 생성에 이 표현을 적용하였다.

이 연구에서는 MULTI-BAND DIFFUSION(MBD)이라는 새로운 diffusion 기반 방법을 제시하였다. 이 방법은 이산 압축 표현에서 음성, 음악, 환경 소리 등의 고품질 오디오 샘플을 생성할 수 있다. 이 방법은 다양한 작업과 오디오 도메인에 적용 가능하며, 전통적인 GAN 기반 decoder를 대체할 수 있다. 결과적으로, 이 방법은 평가된 기준선을 크게 웃돌아 성능을 보였다.

![](images/figure1.png)

**Our Contributions:** 오디오 합성을 위한 새로운 diffusion 기반 모델을 제안한다. 이 모델은 각각의 주파수 대역을 독립적으로 처리하는 diffusion 모델, 주파수 이퀄라이저, 그리고 풍부한 고조파를 가진 오디오 데이터를 위한 파워 노이즈 스케줄러를 포함한다. 이 방법은 객관적 지표와 인간 연구를 통해 최첨단 GAN과 diffusion 기반 접근법에 비해 더 효율적임을 입증하였다.

---

## Related work

neural audio synthesis은 웨이브넷과 같은 autoregressive 모델로 시작되었으나, 이는 학습이 느리고 어려운 단점이 있다. 음성 합성 분야에서는, mel-spectrogram에 기반한 다양한 방식, 특히 GAN 기반의 HiFi-GAN 같은 모델이 탐색되었다. 최근에는 HiFi-GAN을 사용해, HuBERT, VQ-VAE, CPC 같은 self-supervise 방법으로 학습한 저 비트율 표현과 함께 기본 주파수와 스피커 정보를 조합하여, 스피커와 기본 주파수로부터 독립적인 제어 가능한 음성을 생성하는 연구가 이루어졌다.

diffusion-based vocoder는 이미지 생성에서의 diffusion 성공에 영감을 받아 개발되었다. Diffwave는 기본 diffusion 방정식을 오디오에 적용하며, PriorGrad는 조건부 mel-spectrogram의 에너지를 사용해 사전 노이즈 분포를 조정하는 Diffwave의 확장이다. Wavegrad는 연속적인 노이즈 수준에 조건을 사용한다. Takahashi et al. 은 노래하는 목소리의 복잡한 분포를 다루며, 계층적 모델로 고품질 오디오를 생성한다. 최근 연구는 diffusion을 사용해 고해상도 오디오를 생성하지만, 아직 오디오 모달리티 범위가 좁다.

대부분의 diffusion 모델은 복잡한 데이터를 샘플링하기 위해 업샘플링을 사용하지만, 이 과정은 병렬 처리가 불가능하다. 최근, SimpleDiffusion이라는 연구에서는 단일 모델을 이용해 복잡한 diffusion 과정을 단순화하면서도, 낮은 주파수에 집중하여 고품질의 결과를 얻는 방법을 제안하였다. 그러나 이 아이디어는 아직 오디오 처리 분야에는 적용되지 않았다.

이 연구는 SoundStream과 EnCodec 같은 adversarial neural audio codec에 대한 대체 방안을 제시한다. 이는 다양한 손실 조합으로 학습된 encoder, quantizer, decoder 구조를 갖추고 있지만, diffusion 기반 decoder는 더 높은 품질의 오디오 생성을 주관적 평가를 통해 달성한다.

---

## Method

### Background



---

## Reference

* [Paper](https://arxiv.org/pdf/2308.02560.pdf)
* [Github](https://github.com/facebookresearch/audiocraft/blob/main/docs/MBD.md)