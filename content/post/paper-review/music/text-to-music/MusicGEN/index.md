+++
author = "Kurt"
title = "MusicGEN"
date = "2024-01-30"
description = "Simple and Controllable Music Generation"
categories = [
    "Paper Review"
]
tags = [
    "Music",
    "Text-to-Music",
]
draft = true
+++

## Abstract

conditional music generation을 위한 "MusicGEN"이라는 언어 모델을 개발하였다. 이 모델은 여러 스트림의 압축된 이산 음악 표현을 다루며, 효율적인 토큰 교차 패턴과 single-stage transformer를 사용해 여러 모델을 계층적으로 구성하거나 업샘플링할 필요가 없다. 이 방법을 통해 텍스트 설명이나 멜로디 특징에 따라 높은 품질의 음악 샘플을 생성할 수 있음을 입증하였다. 실증적 평가를 통해 제안된 접근법이 기존 벤치마크보다 우수하다는 것을 보여주었다.

---

## Introduction

text-to-music은 텍스트 설명을 바탕으로 음악을 생성하는 작업이다. 이 과정은 long range sequence를 모델링하고 full frequency spectrum을 사용해야 하므로 어렵다. 또한, 다양한 악기의 하모니와 멜로디를 포함하는 음악은 복잡한 구조를 가지며, 이로 인해 음악 생성 과정에서는 멜로디 오류를 범할 여지가 거의 없다. 키, 악기, 멜로디, 장르 등 다양한 요소를 제어할 수 있는 능력은 음악 창작자에게 필수적이다.

self-supervised audio representation, sequential modeling, audio synthesis 등의 최근 연구 진보가 새로운 모델 개발을 가능하게 한다. 최근 연구들은 오디오 신호를 같은 신호를 표현하는 여러 이산 토큰의 스트림으로 나타내는 것을 제안하였는데, 이를 통해 고품질의 오디오 생성과 효과적인 오디오 모델링이 가능해졌다. 그러나 이는 여러 parallel dependent stream을 동시에 모델링해야한다는 비용을 수반한다.

Kharitonov et al. 과 Kreuk et al. 은 음성 토큰의 다중 스트림을 병렬로 모델링하는 지연 접근법을 제안하였다. Agostinelli et al. 은 음악 세그먼트를 다양한 세부성의 이산 토큰 시퀀스로 표현하고 이를 autoregressive 모델로 모델링하는 방식을 제안하였다. Donahue et al. 은 비슷한 접근법을 가요 생성 작업에 적용했고, Wang et al. 은 문제를 두 단계로 해결하는 방법을 제안하였다: 첫 번째 토큰 스트림만 모델링한 후, non-autoregressive 방식으로 나머지 스트림을 모델링한다.

이 연구에서는 텍스트 설명에 따른 고품질 음악을 생성하는 "MusicGEN"이라는 단순하고 조절 가능한 모델을 소개한다. 이 모델은 음향 토큰의 병렬 스트림을 모델링하는 프레임워크를 제안하며, 스테레오 오디오 생성을 추가 비용 없이 확장할 수 있다. 또한, 비지도 멜로디 조건 설정을 통해 생성된 샘플의 제어력을 향상시키고, 주어진 조화와 멜로디 구조에 맞는 음악을 생성할 수 있다. MusicGEN은 평가에서 100점 만점에 84.8점의 높은 점수를 받았으며, 이는 최고 기준선의 80.5점보다 우수한 성능을 보여준다. 마지막으로, 인간 평가에 따르면 MusicGEN은 주어진 조화 구조에 잘 맞는 멜로디를 가진 고품질 샘플을 생성하며, 텍스트 설명을 충실히 따른다.

**Our contribution:** 32 kHz에서 고품질 음악을 생성하는 간단하고 효율적인 모델, MusicGEN을 제안한다. 이 모델은 효율적인 코드북 교차 전략을 통해 일관된 음악을 생성하며, 텍스트와 멜로디 조건에 모두 부합하는 단일 모델을 제공한다. 생성된 오디오는 제공된 멜로디와 일치하고 텍스트 조건 정보에 충실하다. 또한, 주요 설계 선택에 대한 광범위한 객관적 평가와 인간 평가를 제공한다.

---

## Method

---

## Reference

* [Paper](https://arxiv.org/pdf/2306.05284.pdf)
* [Github](https://github.com/facebookresearch/audiocraft)