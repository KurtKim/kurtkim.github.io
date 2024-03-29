+++
author = "Kurt"
title = "MBD"
date = "2024-03-13"
description = "From Discrete Tokens to High-Fidelity Audio Using Multi-Band Diffusion"
categories = [
    "Paper Review"
]
tags = [
    "Audio",
    "Codec"
]
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

**Scheduler Tuning.** 노이즈 스케줄은 diffusion 모델의 품질을 결정하는 핵심 hyperparameter이다.

raw waveform 생성에는 주로 linear 또는 cosine 스케줄이 사용되지만, 고샘플링 레이트에서는 성능이 떨어진다는 것을 발견하였다. 따라서, 이 연구에서는 더 급진적인 p-power 스케줄 사용을 제안한다.

$$ \beta_t = \big( \sqrt[p]{\beta_0} + {{t}\over{T}} ( \sqrt[p]{\beta_T} - \sqrt[p]{\beta_0} ) \big) $$

![](images/figure3.png)

학습 중 주입되는 노이즈의 분산($\beta_0$과 $\beta_T$)은 중요한 hyperparameter이다. 생성 시 노이즈 스케줄을 학습 후에 결정할 수 있음에도 불구하고, 실제로 학습 노이즈 스케줄은 diffusion 모델에 있어 중요한 역할을 한다. 이 스케줄은 학습 예제에 주입되는 노이즈 레벨을 결정하며, 제안된 파워 스케줄을 사용하면 대부분의 예제에 매우 적은 양의 노이즈를 주입하게 된다.

diffusion 과정의 마지막 단계에서, 모델이 추정하는 노이즈가 실제 데이터보다 못한 경우가 종종 발생한다. 이는 학습의 제한된 정밀도 때문이라고 추정된다. 이 문제를 해결하기 위해, 해당 시간 단계를 건너뛰는 것과 같은 효과를 내기 위해 모델을 정체 함수로 대체하고, 이 현상을 방지하기 위해 $\beta_t$ 값을 조정하여 $\sqrt{1-\alpha_t}$ 값을 충분히 크게 한다.

**Band-Specific Training.** audio diffusion 모델은 낮은 주파수를 먼저 생성하고, 역 과정의 마지막에서 고주파수를 처리한다. 오디오 데이터는 시간과 주파수에 걸쳐 복잡하게 얽혀 있어, 전대역 오디오 데이터를 사용한 학습은 고주파수 생성 시 항상 정확한 낮은 주파수를 제공한다. 하지만, 이 방식은 생성 초기의 오류를 역 과정에서 증폭시키는 문제를 가지고 있다.

각 주파수 대역을 독립적으로 학습시키는 멀티밴드 확산 방식을 제안하였다. 이 접근법은 샘플의 지각 품질을 크게 향상시켰으며, 모델 채널에 따른 주파수 대역 분할은 같은 결과를 내지 못했다. 이는 학습 시 이전에 생성된 내용을 모델에 제공하지 않음으로써 샘플링 오류 누적을 방지할 수 있다는 우리의 가설을 확인시켜 준다.

---

## Experimental Setup

### Model & Hyperparameters

**Overview.** 이 접근법은 EnCodec decoder를 대체하며, 필요에 따라 품질과 속도 사이에서 원본과 diffusion decoder를 자유롭게 전환할 수 있는 유연성을 제공한다.

**Architecture.** Chen et al., Kong et al., Lee et al.의 연구에 이어 Ronneberger et al.이 제안한 대칭형 U-net 네트워크를 사용하고, Défossez et al.의 두 residual block과 stride 4의 downsampling/upsampling block을 적용하였다. input audio conditioning과 timestep $t$는 네트워크 병목에 통합되고, 고차원 데이터 확산시 병목 부근에 계산 자원을 집중하는 것이 좋다고 Hoogeboom et al.이 권장한다. 이에 따라 growth rate를 4로 설정했으며, 모델의 크기는 1GB이다.

**Input Conditioning.** 공개된 24kHz EnCodec 모델의 latent representation을 사용하며, 이는 학습 동안 고정된다. 임베딩 시퀀스는 UNet 병목 차원에 맞게 linear interpolation으로 upsample 된다. 실험에는 1.5kbps, 3kbps, 6kbps 비트레이트에 해당하는 EnCodec 코드북 1, 2, 4를 사용한 재구성이 포함되며, 여러 코드북 사용 시 임베딩은 코드북들의 평균으로 계산된다.

**Schedule.** 제안된 power schedule로 diffusion 모델을 학습시켰다. 이때 파워 $p=7.5$, 초기 $\beta_0=1.0e−5$, 최종 $\beta_T=2.9e−2$를 사용하였다. 생성 시에는 20단계, 학습 시에는 1000단계를 사용하는 것이 모델의 다양성 증가와 다양한 노이즈 수준에서의 학습 가능성 때문에 유익하다는 것을 발견했다. 실험에서는 가장 간단한 시간 단계 하위 샘플링 방식 $S = \lbrace i * {{1000}\over{N}}, i \in \lbrace 0, 1, ..., N \rbrace \rbrace$ 을 사용, 여기서 $N$은 샘플링 단계 수(기본값 20)이다.

**Frequency EQ processor.** 실험에서 $ρ = 0.4$ 값을 가진 8개 멜 스케일 주파수 밴드를 활용하며, 내부 음악 데이터셋을 통해 해당 밴드 값들을 계산한다.

**Band Splitting.** 별개의 diffusion 과정을 사용하며, julius로 멜 스케일 기반 4개 주파수 밴드를 균등 분할한다. 이 밴드들은 프로세서와 무관하며, 모든 모델은 같은 hyperparameter, schedule, conditioning input EnCodec 토큰을 공유한다.

**Training.** Adam optimizer, batch size 128, learning rate 1e-4로 모델 학습. 16GB Nvidia V100 4개로 한 모델 학습에 2일 소요된다.

**Computational cost and model size.** diffusion 모델 샘플링의 비용은 생성을 위한 모델 패스 수에 의해 발생한다.

### Datasets

다양한 도메인에서 학습을 진행한다. Common Voice 7.0(9096시간)과 DNS 챌린지 4(2425시간)로 음성 데이터를, MTG-Jamendo(919시간)로 음악 데이터를, FSD50K(108시간)와 AudioSet(4989시간)으로 환경 소리 데이터를 사용한다. AudioSet은 연구 재현을 위해서만 사용되며, 평가에는 내부 음악 데이터셋 샘플을 활용한다.

### Evaluation Metrics

**Human evaluation.** 인간 연구에 MUSHRA 프로토콜을 적용해, 숨겨진 참조와 낮은 앵커를 사용한다. 크라우드 소싱을 통해 모집된 평가자들은 제공된 샘플의 품질을 1에서 100 사이로 평가하였다. 테스트 세트의 각 카테고리에서 무작위로 선정된 5초 길이의 50개 샘플에 대해 샘플 당 최소 10개의 평가를 받았다. 잡음이 많은 평가와 이상치를 제거하기 위해, 참조 녹음을 20% 이상의 경우에 90 미만으로, 낮은 앵커 녹음을 50% 이상의 경우에 80 이상으로 평가한 평가자들을 제외하였다.

**Objective metrics.** 두 가지 자동 평가 방법을 사용한다. 첫 번째는 ViSQOL 메트릭이고, 두 번째는 복원된 신호의 멜-스펙트로그램 충실도를 새로운 메트릭으로 측정하는 방법이다. 이를 위해, 참조 파형 신호와 복원된 신호를 정규화하고, $M$ 멜과 $H$ 홉 길이를 사용해 멜-스펙트로그램을 계산한다.

$$ z = mel \big[ {{x}\over{\epsilon + \sqrt{ \langle x^2 \rangle}}} \big], \ \text{and} \ \hat{z} = mel \big[ {{\hat{x}}\over{\epsilon + \sqrt{ \langle x^2 \rangle}}} \big] $$

멜-스펙트로그램의 왜곡을 분석하기 위해, 우리는 신호 대 잡음비(SNR)를 각 시간 단계와 주파수 빈에서 계산한다. 계산 시 -25dB와 +25dB 사이로 SNR 값을 제한하여 수치적 불안정성과 기준 멜-스펙트로그램의 거의 0에 가까운 값들로 인한 과도한 영향을 방지한다. 이는 신경망의 계산과 학습의 제한된 정밀도로 인해 완전히 0의 에너지 수준을 출력하는 것이 어렵다는 점을 고려한 것이다.

$$ s = clamp [10 \cdot (log 10 (z) − log 10 (δ))., \ −25dB, +25dB] $$

시간 단계별로 평균을 내고 멜 스케일 밴드를 3등분하여 저, 중, 고주파수의 멜-SNR(L, M, H)을 산출한다. 모든 밴드의 평균은 Mel-SNR-A로 보고된다. 24kHz에서는 512샘플 프레임에 대해 STFT를 사용, 홉 길이는 128, 멜 밴드는 80개이다.

---

## Results

### Multi modalities model

압축 작업에서 EnCodec과 비교해 diffusion 방식의 성능을 검토합니다. EnCodec encoder로 오디오 샘플에서 토큰을 추출하고, Multi-Band Diffusion과 원본 decoder로 디코딩한다.

![](images/table1.png)

DNS에서 깨끗한 음성, 손상된 음성, Jamendo와 내부 음악 데이터셋에서 각각 음악 샘플 50개씩, 총 4가지 부분집합에 대해 주관적 평가를 실시하였다. 모든 음성 샘플은 DNS 챌린지의 방 임펄스 응답을 이용해 확률 0.2로 울림효과를 부여받는다. 6kbps, 3kbps, 1.5kbps 비트율에서 세 가지 주관적 연구 결과를 제공하며, 평가는 상대적으로 이루어져 연구 간 비교는 불가하다. 6kbps에서 저품질 앵커와 지상 진실 샘플로 Opus를 포함시켰고, EnCodec과의 비교는 모델 크기가 결과에 제한적이지 않음을 명시한다.

Multi-Band Diffusion 방법은 음성 압축에서 EnCodec보다 최대 30% 더 우수한 성능을 보이고, 음악 데이터에서는 EnCodec과 비슷한 수준이다. 전체적으로, 모든 비트율에서 EnCodec보다 우수하다. GAN 기반 방법이 금속성 소리를 만들 수 있지만, diffusion 방법은 더 자연스러운 고주파 내용을 제공한다.

같은 데이터로 훈련된 HifiGAN과 PriorGrad와 이 연구의 모델을 비교한다. 이때 각 모델의 원본 논문에 제시된 hyperparameter를 사용하였다.

![](images/table2.png)

후반부에서는 EnCodec을 사용하지 않는 다른 종단간 오디오 코덱들과 비교를 추가한다. 이 중에는 24kHz에서 6kbps로 운영되는 DAC의 사전 학습된 모델이 포함된다. EnCodec + Multi-Band Diffusion이 다른 양자화 방식을 사용하는 DAC와 비슷한 수준이라고 보여준다. Multi-Band Diffusion을 DAC의 오디오 토큰으로 학습시키면 오디오 품질이 더 향상될 것으로 예상된다.

### Ablations

![](images/table3.png)

이 연구에서는 ViQOL 점수와 Mel-SNR을 사용하여 다양한 모달리티에서 150개 샘플의 복원 성능을 객관적으로 비교하였다. 연구 결과, EnCodec 방법이 객관적 지표에서는 우수한 성능을 보였지만, 주관적 평가에서는 다소 낮은 성능을 보여주었다. 반면, diffusion 기반 방법은 더 자연스러운 오디오 생성을 가능하게 하며, 생성적 작업에 있어서 선호되는 방법으로 주장된다. 이러한 차이는 각각의 방법이 콘텐츠 복원을 위해 특화되어 있기 때문에 발생한다.

본 논문에서 제시한 한 요소를 제외하고 모델을 평가하는 소거 연구를 통해 기여도의 영향을 평가하였다.

![](images/table4.png)

연구 결과, 단계 수를 20까지 늘리면 출력 품질이 향상되지만 그 이상은 효과가 줄어든다. 단일 모델 대비 네 모델 사용이 오디오 품질과 모든 측정 지표에서 우수함을 보여주었다. 또한, 주파수 밴드 재조정으로 ViSQOL 점수가 0.2 향상되었으며, 스케줄 방식은 기존 방식보다 성능이 0.2~0.4 개선되었다. 제안한 데이터 처리 기술로도 ViSQOL 점수가 0.2 증가했으며, 이는 주로 고주파수에 영향을 미쳤다.

### Text to audio

이 모델은 조건 없이 오디오를 생성할 수 없으나, 생성 언어 모델과 결합 시 품질이 크게 향상된다.

**Text to Speech.** 최근 텍스트에서 음성으로 변환하는 TTS 분야에서 오디오 코덱에 언어 모델을 적용하는 연구가 주목받고 있다. 이 분야에서 VALL-E와 SPEAR-TSS와 같은 모델이 좋은 성과를 보여주었다. Multi-Band Diffusion 토큰 decoder를 사용하여 오디오 품질을 더욱 향상시킬 수 있다고 주장한다. 이를 검증하기 위해 공개적으로 이용 가능한 Bark 3 모델을 사용하였고, 이 모델은 텍스트를 오디오 토큰으로 변환한 뒤, 이를 다시 처리하여 최종 오디오를 생성한다. 실험 결과, 사전 학습된 Bark 모델을 사용했을 때 음성 프롬프트의 5% 미만, 노래 목소리 프롬프트의 약 30%에서 언어 모델이 목소리를 생성하지 못하는 경우가 있었다.

![](images/table5.png)

**Text to Music.** 최근 오디오 토큰의 언어 모델링을 통한 음악 생성 분야에서 MusicLM과 MusicGen 같은 프로젝트를 통해 큰 진전이 이루어졌다. MusicGen의 압축 모델로 생성된 토큰을 기반으로 한 확산 모델을 학습시켜 디코딩 방식의 유연성을 입증했으며, 이 모델은 32kHz 샘플링 레이트와 16개 멜 스케일 밴드의 표준 편차를 가진 데이터셋에서 학습되었다.

표준 MusicGen 대비 MUSHRA 점수를 +4 향상시켰으며, diffusion decoder로 생성된 아티팩트가 적다. 특히, 복잡한 음악 요소가 있는 경우, Multi-Band Diffusion 출력이 원본보다 훨씬 명확함을 확인하였다.

---

## Discussion

diffusion 기반 디코딩 방법은 기존 decoder 대비 오디오 품질을 크게 향상시키지만, 더 많은 계산력을 요구하고 처리 속도가 느리다. 이 방법으로 더 자연스러운 오디오와 적은 아티팩트를 생성하지만, 실시간 성능이 중요한 경우에는 적합하지 않을 수 있다.

**Ethical concerns.** 생성 AI가 아님에도 Wang et al. (2023) 같은 기술과 결합하여 목소리의 진정성을 높일 수 있으나, 진짜 같은 딥페이크와 보이스 피싱 같은 오용의 위험이 있다. 학습 데이터의 질과 양에 의존하는 이 방법은 광범위한 시나리오에서 최적화되기 위해 큰 데이터셋으로 세심히 학습되었지만, 데이터셋의 불균형으로 인한 소수 집단에 대한 편향 가능성을 인정한다.

---

## Reference

* [Paper](https://arxiv.org/pdf/2308.02560.pdf)
* [Github](https://github.com/facebookresearch/audiocraft/blob/main/docs/MBD.md)