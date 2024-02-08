+++
author = "Kurt"
title = "EnCodec"
date = "2024-02-07"
description = "High Fidelity Neural Audio Compression"
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

neural network를 활용한 state-of-the-art real-time, high-ﬁdelity, audio codec을 소개한다. 이는 아티팩트를 효율적으로 줄이고 고품질 샘플을 생성하는 스트리밍 encoder-decoder 구조이다. 학습을 안정화하기 위해 loss balancer mechanism을 도입하였으며, lightweight Transformer 모델을 사용하여 얻은 표현을 최대 40%까지 더 압축하는 방법을 연구하였다. 이 모델은 말하기, 소음이 많은 반향성 말하기, 음악 등 다양한 오디오 도메인에서 우수한 성능을 보여주었다.

---

## Introduction

2021년에 스트리밍 오디오와 비디오가 인터넷 트래픽의 82%를 차지했고, 이런 트렌드는 오디오 압축의 중요성을 강조한다. 손실 압축은 샘플의 비트레이트와 왜곡을 최소화하는 것을 목표로 한다. 오디오 codec은 중복성을 제거하고 컴팩트한 비트 스트림을 생성하기 위해 encoder와 decoder를 결합한다. neural network를 활용한 encoder-decoder 메커니즘은 오디오 신호에 중점을 둔 연구의 일환으로서 탐구되어 왔다.

lossy neural compression 모델에서는 두 가지 문제가 발생한다. 첫 번째는 학습 세트를 과적합하지 않고, 아티팩트가 많은 오디오를 생성하지 않도록 다양한 신호를 표현해야 하는 것이다. 이를 위해 다양한 학습 세트와 perceptual 손실로 작용하는 discriminator network를 사용하였다. 두 번째 문제는 계산 시간과 크기를 모두 고려하여 효율적으로 압축하는 것이다.

실시간으로 단일 CPU 코어에서 작동하는 모델에 제한을 두며, neural encoder의 출력에 대한 residual vector quantization를 사용하여 효율적으로 압축한다. 이에 대한 여러 방법이 이전의 연구에서 제안되었다.

end-to-end neural compression 모델 설계가 encoder-decoder 아키텍처, quantization 방법, perceptual 손실 등을 포함한 선택의 집합이라고 주장한다. 이 모델의 평가는 객관적인 방법과 인간의 인식에 의존하는 방법 두 가지를 사용하였고, 이를 통해 이 모델이 음성과 음악 압축에서 state-of-the-art를 달성하였음을 확인하였다.

---

## Related Work

**Speech and Audio Synthesis.** 최근의 neural audio generation 기술 발전은 컴퓨터가 효율적으로 자연스러운 오디오를 생성하도록 하였다. autoregressive 모델인 WaveNet이 초기 성공을 거뒀지만, 추론 속도가 느렸다. 여러 다른 방법이 탐색되었지만, 특히 Generative Adversarial Network (GAN) 기반의 방법이 주목 받았다. 이들은 다양한 adversarial network를 결합하여 더 빠른 속도로 autoregressive 모델의 품질을 달성하였다. 이 연구는 이러한 adversarial 손실을 활용하고 확장하여 오디오 생성 중의 아티팩트를 줄이는 데 초점을 맞추고 있다.

**Audio Codec.** 낮은 비트레이트의 음성과 오디오 codec에 대한 연구가 오랫동안 이루어졌지만, 품질은 제한적이었다. excitation signal을 모델링하는 것은 여전히 어려운 과제로 남아 있다. 현재 state-of-the-art인 전통적인 오디오 codec은 Opus와 Enhanced Voice Service (EVS)로, 다양한 비트레이트, 샘플링 레이트, 실시간 압축을 지원하며 높은 코딩 효율성을 보여준다.

최근에 제안된 neural based audio codec은 놀라운 결과를 보여주었다. 대부분의 방법들은 latent space를 quantizing한 후 decoder에 입력하는 방식을 사용하였다. 여러 연구들에서 다양한 접근법이 시도되었으며, 가장 관련성이 높은 연구로는 SoundStream 모델이 있다. 이 모델은 Residual Vector Quantization layer를 포함하는 fully convolutional encoder-decoder 아키텍처를 제안하였고, reconstruction 손실과 adversarial perceptual 손실 모두를 사용하여 최적화하였다.

**Audio Discretization.** 최근에는 discrete 값으로 오디오와 음성을 표현하는 방법이 다양한 작업에 적용되었다. raw 오디오의 discrete 표현을 학습하기 위한 계층적 VQ-VAE 기반 모델은 고품질 음악 생성을 가능하게 했고, 음성에 대한 self-supervised 학습 방법이 conditional 및 unconditional 음성 생성에 사용되었다. 이러한 방법은 음성 재합성, 음성 감정 변환, 대화 시스템, 음성-음성 번역 등의 분야에도 적용되었다.

---

## Model

오디오 신호의 기간이 $d$라면, 이 신호는 $x \in [−1, 1]^{C_a \times T}$ 시퀀스로 표현될 수 있다. 여기서 $C_a$는 오디오 채널의 수이고, $T = d \cdot f_{sr}$는 주어진 샘플 비율 $f_{sr}$에서의 오디오 샘플 수이다.

![](images/figure1.png)

EnCodec 모델은 오디오 신호를 처리하는 세 가지 주요 요소로 구성된다. 첫째, encoder 네트워크 $E$는 오디오를 latent representation $z$로 변환한다. 둘째, quantization layer $Q$는 vector quantization 를 이용해 압축된 표현 $z_q$를 생성한다. 셋째, decoder 네트워크 $G$는 compressed latent representation $z_q$을 원래의 시간 도메인 신호 $x$로 재구성한다. 이 시스템은 시간과 주파수 도메인에서의 reconstruction 손실 최소화를 목표로 학습되며, 이 과정에는 다른 해상도에서 작동하는 판별자의 discriminator 손실이 포함된다.

### Encoder & Decoder Architecture

EnCodec 모델은 streaming과 convolutional-based encoder-decoder 구조로, latent representation에 순차적 모델링을 적용한다. 이 구조는 다양한 오디오 작업에서 뛰어난 성과를 보였으며, source separation, enhancement, neural vocoder, audio codec, artiﬁcial bandwidth extension 등에 활용되었다. 이 모델은 24 kHz와 48 kHz 오디오에 동일하게 적용된다.

**Encoder-Decoder.** EnCodec의 encoder 모델 $E$는 1D convolution과 여러 convolution block으로 구성된다. 각 block은 residual unit과 strided convolution으로 이루어진 down-sampling layer를 포함하며, down-sampling이 있을 때마다 채널 수가 두 배씩 증가한다. 이어서 시퀀스 모델링을 위한 LSTM 계층과 1D convolution layer가 뒤따른다. 이 모델은 low-latency streamable과 high ﬁdelity non-streamable에 따라 두 가지 변형으로 사용된다. encoder는 24 kHz에서 초당 75개, 48 kHz에서는 초당 150개의 latent step을 출력하며, decoder는 이를 받아 최종 오디오를 생성한다.

**Non-streamable.** non-streamable 설정에서는 각 convolution에 대해 총 패딩 $K - S$를 사용하고, 입력을 1초 청크로 분할한다. 10ms의 오버랩을 통해 클릭을 방지하고, 각 청크를 모델에 공급하기 전에 normalization한다. decoder의 출력에 inverse operation을 적용하고, 스케일 전송에 대한 negligible bandwidth overhead를 최소화한다. layer normalization를 사용하여 상대적인 스케일 정보를 유지한다.

**Streamable.** streamable 설정에서는 모든 패딩을 첫 번째 시간 단계 전에 배치한다. 스트라이드가 있는 transposed convolution을 사용하여, 처음 $s$ 시간 단계를 출력하고, 다음 프레임이 준비되면 나머지를 완성하거나, 스트림 끝에서 버린다. 이 패딩 방식 덕분에 모델은 첫 320 샘플을 받자마자 320 샘플을 출력할 수 있다. 또한, streamable 설정에 부적합한 layer normalization 대신 weight normalization를 사용한다. 이렇게 normalization을 유지함으로써 목표 지표에서 약간의 향상을 얻었다.

### Residual Vector Quantization

encoder의 출력을 quantize 하기 위해 Residual Vector Quantization (RVQ)을 사용한다. Vector quantization는 입력 벡터를 코드북의 가장 가까운 항목에 투영하는 것이며, RVQ는 이를 개선하여 quantization 후의 residual을 계산하고 추가로 quantizing 한다.

Dhariwal et al. 과 Zeghidour et al. 이 설명한 학습 절차를 따르며, 각 입력에 대한 코드북 항목을 exponential moving average을 사용해 업데이트한다. 사용되지 않는 항목은 현재 batch에서 샘플링된 후보로 대체된다. encoder의 기울기를 계산하기 위해 straight-through-estimator를 사용하고, quantizer의 입력과 출력 사이의 MSE로 구성된 commitment 손실을 전체 학습 손실에 추가한다.

학습 시간에 residual step의 수를 조절하여, 단일 모델이 multiple bandwidth 목표를 지원할 수 있다. 모든 모델은 최대 32개(48 kHz 모델은 16개)의 코드북을 사용하며, 각 코드북은 1024개의 항목을 가진다. variable bandwidth 학습 시, 4의 배수로 코드북의 수를 무작위로 선택한다. 이렇게 하여, encoder에서 나오는 continuous latent represention을 discrete set of index로 변환하고, 이를 decoder로 들어가기 전에 다시 벡터로 변환한다.

### Language Modeling and Entropy Coding

실시간보다 빠른 compression/decompression을 목표로, small Transformer 기반 언어 모델을 학습시킨다. 모델은 5개 layer, 8개 head, 200개 channel, feed-forward block의 800 dimension을 가진다. 학습 시, 대역폭과 해당 코드북의 수를 선택하고, 시간 단계별로 discrete representation을 continuous representation으로 변환한다. Transformer의 출력은 linear layer에 공급되어 각 코드북에 대한 estimated distribution의 logit을 제공한다. 이 방법은 코드북간의 잠재적 정보를 무시하면서도 추론을 가속화합니다. 모델은 5초 시퀀스에서 훈련됩니다.

**Entropy Encoding.** 언어 모델로부터 얻은 추정 확률을 활용하기 위해, range based arithmetic coder를 사용한다. 다른 아키텍처나 ﬂoating point approximation으로 인해 동일한 모델의 평가가 다르게 나올 수 있어 디코딩 오류가 발생할 수 있다. 특히, batch 평가와 real-life streaming 평가 사이에는 큰 차이가 있을 수 있다. 따라서 추정 확률을 $10^{-6}$의 정밀도로 반올림하며, 총 범위 너비를 $2^{24}$로, 최소 범위 너비를 $2$로 설정한다. 처리 시간에 미치는 영향에 대해서는 추후 논의하고자 한다.

### Training objective

reconstruction 손실, perceptual 손실 (via discriminators), 그리고 RVQ commitment 손실을 결합한 학습 목표를 상세히 설명한다.

**Reconstruction Loss.** reconstruction loss term은 시간 도메인과 주파수 도메인의 손실 항으로 이루어진다. 시간 도메인에서는 목표 오디오와 압축 오디오 사이의 L1 거리를 최소화하고, 주파수 도메인에서는 mel-spectrogram에서의 L1과 L2 손실의 선형 조합을 사용한다.

$$ l_f(x, \hat{x}) = {{1}\over{|\alpha| \cdot |s|}} \sum_{\alpha_i \in \alpha} \sum_{i \in e} \parallel S_i(X) - S_i(\hat{x}) \parallel_1 + \alpha \parallel S_i(X) - S_i(\hat{x}) \parallel_2 $$

$S_i$는 window size가 $2^i$ 이고 hop length가 $2^i/4$인 normalized STFT를 사용한 64-bins mel-spectrogram이다. $e = 5, ..., 11$은 스케일의 집합을 나타내고, $\alpha$$는 L1과 L2 항 사이의 균형을 맞추는 스칼라 계수의 집합이다. 단, 이 논문에서는 $\alpha_i = 1$을 선택하였다.

![](images/figure2.png)

**Discriminative Loss.** 생성된 샘플의 품질을 향상시키기 위해, multi-scale STFT-based (MS-STFT) discriminator에 기반한 perceptual 손실 항을 도입하였다. 이 discriminator는 audio signal의 다양한 구조를 포착하도록 설계되었으며, 복소수 값을 가진 multi-scale STFT에서 작동하는 동일 구조의 네트워크로 구성되어 있다. 각 하위 네트워크는 2D convolutional layer, 팽창율이 증가하는 2D convolution, 그리고 주파수 축에서 스트라이드를 가지고 있다. 이 discriminator는 STFT window length가 다양한 5개의 스케일을 사용하며, 오디오의 샘플링 레이트에 따라 window size를 조정한다. LeakyReLU 활성화 함수와 weight normalization을 사용한다.

generator에 대한 adversarial 손실은 discriminator의 수(K)에 따라 구성되며, 이는 $l_g(\hat{x}) = {{1}\over{K}} \sum_k max(0, 1 − D_k(\hat{x}))$ 공식으로 표현된다. 또한, 이전 neural vocoder 연구와 같이, generator에 대한 상대적 특징 매칭 손실을 추가적으로 포함한다.

$$ l_{feat}(x, \hat{x}) = {{1}\over{KL}} \sum_{k = 1}^K \sum_{l = 1}^L {{\parallel D_k^l(x) - D_k^l(\hat{x}) \parallel_1}\over{mean(\parallel D_k^l(x) \parallel_1)}} $$

평균은 모든 차원에서 계산되며, discriminator들은 hinge 손실 adversarial 손실 함수를 최소화하는 것을 목표로 한다. discriminator가 decoder를 쉽게 압도하는 경향이 있으므로, 24 kHz에서는 2/3의 확률로, 48 kHz에서는 0.5의 확률로 discriminator의 가중치를 업데이트한다.

**Multi-bandwidth training.**

---

## Reference

* [Paper](https://arxiv.org/pdf/2210.13438.pdf)
* [Github](https://github.com/facebookresearch/encodec)