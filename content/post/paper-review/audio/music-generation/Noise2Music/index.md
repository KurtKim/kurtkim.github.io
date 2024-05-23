+++
author = "Kurt"
title = "Noise2Music"
date = "2024-05-20"
description = "Text-conditioned Music Generation with Diffusion Models"
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

Noise2Music은 텍스트 프롬프트로부터 30초짜리 고품질 음악 클립을 생성하는 diffusion 모델 시리즈이다. 텍스트에 기반한 중간 표현을 생성하는 생성 모델과 고해상도 오디오를 생성하는 캐스케이더 모델을 이용해, 장르, 템포, 악기 등의 텍스트 프롬프트 요소를 반영하는 음악을 만든다. 중간 표현으로는 스펙트로그램과 낮은 해상도 오디오가 사용된다. 이 과정에서 사전 학습된 대규모 언어 모델이 학습 세트 오디오의 텍스트를 생성하고 텍스트 프롬프트의 임베딩을 추출하는 데 중요한 역할을 한다.

---

## Introduction



---

## Reference

* [Paper](https://arxiv.org/pdf/2302.03917)
* [Demo](https://google-research.github.io/noise2music/)