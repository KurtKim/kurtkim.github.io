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
#draft = true
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

Ho et al. (2020)의 연구에 따르면, Markov chain을 사용한 diffusion 과정에서 깨끗한 데이터 $x_0$에 점진적으로 Gaussian noise를 추가해, 결국 standard Gaussian noise에 가까운 noise가 섞인 데이터 $x_T$를 생성한다. 이 과정의 확률이 다음과 같이 정의된다:

$$ q(x_{0:\gamma} | x_0) = \Pi_{t=1}^T q (x_t | x_{t-1}) $$

$q(x_t | x_{t-1})$는 가우시안 분포를 따르며, $\beta_t$는 noise schedule을 나타낸다. 이를 통해 Markov chain의 어떤 단계도 효율적으로 샘플링할 수 있다.

$$ x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon $$

$\bar{\alpha}_t$는 잡음 수준을 나타내고, DDPM은 잡음이 섞인 데이터 $x_T$에서 깨끗한 데이터 $x_0$로 복원하는 것을 목표로 한다.

$$ p(x_{\gamma : 0}) = p(x_{\gamma}) \Pi_{t=1}^T p_{\theta} (x_{t-1} | x_t) $$

$p_\theta(x_t | x_{t+1})$는 diffusion chain을 역으로 하는 학습된 분포이고, $p(x_T)$는 학습되지 않은 사전 분포이다. 이상적인 잡음 조건에서 사전 분포는 $N(0, I)$로 근사할 수 있다.

Ho et al. (2020)에 따르면, $p_\theta(x_{t-1} | x_t)$ 분포는 $N(\mu_\theta(x_t, t), \sigma_t I)$로 나타낼 수 있으며, $\mu_\theta$는 reparameterize가 가능하다.

$$ \mu_{\theta} (x_t, t) = {{1}\over{\sqrt{1 - \beta_t}}} \big( x_t - {{\beta}\over{\sqrt{1 - \bar{\alpha}_t}}} \epsilon_{\theta} (x_t, t)  \big) $$

이 reparametrization를 통해 신경망 $\epsilon_\theta$는 오염된 데이터 $x_t$에서 잡음을 예측하도록 학습된다. Ho et al. (2020)의 방법에 따라, $x_t$ 샘플링 후 L2 손실을 최적화하여 신경망을 학습할 수 있다.

$$ L = \mathbb{E}_{x_0 \sim d(x_0), \epsilon \sim \mathcal{N}(0,I), t \sim \mathcal{U}\lbrace 1, \ldots, T \rbrace} ( \Vert \epsilon - \epsilon\theta\left(\sqrt{x_0} + \sqrt{1-t}\right) \Vert^2 ) $$

이러한 모델을 사용하면, 다음 방정식을 사용하여 diffusion 과정을 반복적으로 역전할 수 있다:

$$ x_{t-1} = {{1}\over{\sqrt{1 - \beta_t}}} \big( x_i - {{\beta_t}\over{\sqrt{1 - \bar{\alpha}_t}}} \epsilon\_{\theta} (x_t, t) \big) + \sqrt{\sigma_t} \epsilon $$

여기서 $\sigma$는 $\tilde{\beta}t = (1 - \bar{\alpha}{t-1})/(1 - \bar{\alpha}_t) \beta_t$와 $\beta_t$ 사이에서 결정해야 하는 parameter이며, 이 실험에서는 $\sigma_t = \beta_t$로 설정한다.

### Multi-Band Diffusion

Multi-Band Diffusion 방법은 Frequency Eq. Processor, Scheduler Tuning, Band-Specific Training의 세 가지 핵심 요소로 구성된다.

**Frequency Eq. Processor** diffusion 과정 이론은 모든 종류의 분포에서 샘플링을 가능하게 하지만, waveform 도메인의 다양한 오디오 모달리티를 위한 diffusion 네트워크 학습은 아직 해결되지 않은 문제이다. 다른 주파수 밴드에서 에너지 레벨의 균형이 효율적인 샘플링에 중요하다고 가정한다.

![](images/figure2.png)

white Gaussian noise는 모든 주파수에서 동등한 에너지를 가지지만, 자연 소리는(예: 음악, 연설) 다른 분포를 보이며, 특히 높은 주파수에서 더 많은 에너지를 가진다. 이로 인해 diffusion 과정에서 고주파수 내용이 저주파수보다 먼저 사라지고, 역 과정에서 고주파수에 더 큰 영향을 받게 된다.

이 문제를 해결하기 위해 멜 스케일을 기반으로 한 밴드 패스 필터를 사용하여 깨끗한 신호 $x_0$를 여러 주파수 밴드로 나누고, 각 밴드 $b_i$의 에너지를 정규화합니다.

$$ \hat{b}_i = b_i \cdot \big( {{\sigma_i^{\epsilon}\over{\sigma_i^d}}} \big)^p  $$

$\sigma_i^{\epsilon}$과 $\sigma_i^d$는 standard Gaussian noise와 데이터셋 신호의 밴드 $i$ 에너지를 나타내며, 매개변수 $ρ$로 에너지 수준 조정을 제어한다($ρ=0$은 조정 없음, $ρ=1$은 완전 일치). 고주파수 밴드의 instability를 피하기 위해, 음악 도메인에서 $\sigma_i^d$를 계산한다.

**Scheduler Tuning.**

**Band-Specific Training.**





---

## Reference

* [Paper](https://arxiv.org/pdf/2308.02560.pdf)
* [Github](https://github.com/facebookresearch/audiocraft/blob/main/docs/MBD.md)