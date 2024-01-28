+++
author = "Kurt"
title = "OPT"
date = "2024-01-29"
description = "Open Pre-trained Transformer Language Models"
categories = [
    "Paper Review"
]
tags = [
    "NLP",
    "LLM",
]
draft = true
+++

## Abstract

대규모 언어 모델들은 높은 계산 비용 때문에 복제하기 어렵다. 이를 해결하기 위해, Open Pre-trained Transformers (OPT)를 제시한다. 이는 125M에서 175B의 parameter 범위를 가진 사전 학습된 transformer 모델들을 포함하며, 이들은 완전하게 그리고 책임감 있게 관심 있는 연구자들과 공유될 것이다. OPT-175B는 GPT-3와 비교할 수 있으나, 개발하는 데 필요한 탄소 발자국은 1/7밖에 되지 않는다.

--- 

## Introduction

대규모 텍스트 컬렉션에 학습된 거대 언어 모델은 텍스트 생성 및 zero-shot, few-shot 학습 등 놀라운 기능을 보여준다. 그러나 현재로서는 완전한 모델 접근이 풍부한 자원을 가진 몇몇 연구소에만 제한되어 있다. 이 제한된 접근은 대형 언어 모델이 어떻게 그리고 왜 작동하는지 연구하는 능력을 제한하고, 견고성, 편향, 독성 등의 문제를 개선하는 데 있어 진전을 방해하고 있다.

125M에서 175B parameter 범위의 decoder 기반 사전 학습된 transformer인 Open Pretrained Transformers (OPT)를 소개하고 있다. OPT 모델은 GPT-3 계열 모델의 성능과 크기를 대략 맞추도록 학습되었으며, 최신 데이터 수집 및 효율적 학습 방법을 적용하였다. 이 모델은 대규모 연구를 가능하게 하고, 거대 언어 모델의 영향력을 연구하는 다양한 의견을 수렴하기 위해 개발되었다. risk, harm, bias, toxicity 등의 정의는 연구 커뮤니티 전체가 공동으로 명시해야 하며, 이는 모델들이 연구에 사용 가능할 때만 가능하다.

125M부터 66B parameter 사이의 모든 모델을 공개하며, 요청에 따라 OPT-175B에 대한 연구 접근 권한을 제공한다. 학계 연구자, 정부 및 학계의 조직, 산업 연구소에 접근 권한이 부여된다. 모델 생성 로그북과 OPT-175B를 992개의 80GB A100 GPU에서 학습시키는 데 사용된 코드베이스인 metaseq도 공개된다. 이를 통해, 우리는 GPT-3의 탄소 발자국의 1/7만큼의 에너지를 사용해 OPT-175B를 개발할 수 있었다. 이는 큰 성과이지만, 이렇게 큰 모델을 만드는 에너지 비용은 중요하며, 이를 계속 복제하면 LLM들의 컴퓨팅 발자국이 계속 증가할 것이다.

전체 AI 커뮤니티가 책임있는 AI와 LLM 사용에 대한 명확한 지침을 개발하기 위해 협력해야한다고 생각한다. 더 넓은 AI 커뮤니티가 이 모델에 접근하고 재현 가능한 연구를 수행하여 전체 필드를 발전시키는 것이 필요하다. OPT-175B와 작은 규모의 기준선 출시를 통해, 이러한 기술의 윤리적 고려사항에 대한 다양한 의견을 더욱 들을 수 있을 것을 기대한다.

---

## Method

### Models




---

## Reference

* [Paper](https://arxiv.org/pdf/2205.01068.pdf)
* [Github](https://github.com/facebookresearch/metaseq)