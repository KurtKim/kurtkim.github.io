+++
author = "Kurt"
title = "VampNet"
date = "2024-06-10"
description = "Music Generation via Masked Acoustic Token Modeling"
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

VampNet은 음악 합성, 압축, 인페인팅, 변주를 위한 masked acoustic token 모델링 기법이다. 다양한 마스킹 방법을 사용하여 일관된 고해상도 음악을 생성하며, bidirectional transformer 아키텍처를 통해 non-autoregressive 방식으로 작동한다. VampNet은 음악의 스타일, 장르, 악기 사용 등을 유지하면서 음악 생성, 압축, 인페인팅 등 다양한 작업에 적용 가능하다. 이는 VampNet을 음악 공동 창작의 강력한 도구로 만든다. 관련 코드와 오디오 샘플은 온라인에서 제공된다.

---

## INTRODUCTION

discrete acoustic token 모델링의 최근 발전은 음성 및 음악 생성에 중요한 진전을 가져왔다. 또한, non-autoregressive parallel iterative decoding 방법이 이미지 합성을 위해 개발되어, 과거와 미래 정보를 모두 고려하는 작업에 더 효율적이고 빠른 추론을 제공한다.

이 작업에서는 parallel iterative decoding과 acoustic token 모델링을 결합하여 음악 오디오 합성에 적용한 VampNet을 소개한다. 이는 parallel iterative decoding을 신경망 기반 음악 생성에 처음으로 확장한 사례이다. VampNet은 선택적으로 마스킹된 음악 토큰 시퀀스를 사용해 공백을 채우도록 유도할 수 있으며, 고품질 오디오 압축부터 원본 음악의 스타일과 장르를 유지하면서도 음색과 리듬이 변형된 변주까지 다양한 출력을 제공한다.

auto-regressive 음악 모델은 접두사 기반 음악 생성만 가능하지만, 이 연구의 접근 방식은 프롬프트를 어디에나 둘 수 있다. 주기적, 압축, 비트 기반 마스킹 등 다양한 프롬프트 디자인을 탐구하였다. 모델이 루프와 변주 생성에 잘 반응하여 VampNet이라 명명하였다. 코드를 오픈 소스로 공개하며 오디오 샘플 청취를 권장한다.

---

## BACKGROUND



---

## Reference

* [Paper](https://arxiv.org/pdf/2302.03917)
* [Demo](https://google-research.github.io/noise2music/)