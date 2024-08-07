+++
author = "Kurt"
title = "Museformer"
date = "2024-08-10"
description = "Transformer with Fine- and Coarse-Grained Attention for Music Generation"
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

symbolic 음악 생성은 자동으로 악보를 만드는 것을 목표로 하며, 최근 Transformer 모델이 사용되지만 긴 음악 시퀀스 처리와 반복 구조 생성에 한계가 있다. 이 논문에서는 Museformer를 제안한다. Museformer는 ﬁne-attention과 coarse-grained attention 메커니즘을 결합하여 음악 구조와 맥락 정보를 효과적으로 캡처한다. 이 방법은 full-attention 메커니즘보다 계산 효율성이 높아 3배 더 긴 음악 시퀀스를 모델링할 수 있으며, 높은 품질과 구조적 완성도를 제공한다.

---

## Introduction


---

## Reference

* [Paper](https://arxiv.org/pdf/2210.10349)
* [GitHub](https://github.com/microsoft/muzic)