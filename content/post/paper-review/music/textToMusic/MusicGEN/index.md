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

---

## Reference

* [Paper](https://arxiv.org/pdf/2306.05284.pdf)
* [Github](https://github.com/facebookresearch/audiocraft)