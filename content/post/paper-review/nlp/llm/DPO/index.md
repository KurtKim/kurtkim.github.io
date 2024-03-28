+++
author = "Kurt"
title = "Direct Preference Optimization"
date = "2024-03-29"
description = "Your Language Model is Secretly a Reward Model"
categories = [
    "Paper Review"
]
tags = [
    "NLP",
    "LLM",
]
+++

## Abstract

이 논문에서는 대규모 unsupervised 언어 모델(LM)의 행동을 인간의 선호도에 맞추어 정밀하게 제어하는 새로운 방법, Direct Preference Optimization(DPO)를 소개한다. 기존의 복잡하고 불안정한 reinforcement learning from human feedback(RLHF) 대신, DPO는 보상 모델의 새로운 parameterization를 통해 단순한 분류 손실만으로 LM을 미세 조정할 수 있게 하여, 안정적이고 계산적으로 가벼운 방법을 제공한다. 실험 결과, DPO는 기존 방법들을 뛰어넘거나 일치하는 수준으로 인간의 선호도에 부합하게 LM을 미세 조정하며, 특히 생성물의 감정 제어와 요약, 단일 턴 대화의 응답 품질에서 우수한 성능을 보여준다.

---

## Introduction

대규모 비지도 언어 모델들은 다양한 인간 데이터에 기반하여 학습되며, 모든 학습 내용이 바람직하지 않을 수 있다. 예를 들어, AI 코딩 어시스턴트에게는 고품질 코딩 능력을, 언어 모델에게는 특정 오해를 사실로 주장하지 않는 능력을 부여하고 싶을 수 있다. 이러한 모델의 원하는 반응과 행동을 선택하는 것은 AI 시스템을 안전하고 효과적으로 제어하는 데 중요하다. 기존에는 강화 학습을 사용했지만, 단순한 binary cross-entropy 목표를 통해 선호도 학습 파이프라인을 단순화할 수 있음을 보여준다.

기존 방법은 인간의 선호를 반영한 선별된 데이터를 사용해 언어 모델에 원하는 행동을 주입한다. 이 과정은 대규모 비지도 사전 학습 후에 이루어지며, 가장 성공적인 방법은 reinforcement learning from human (or AI) feedback (RLHF/RLAIF)이다. RLHF는 인간 선호에 기반한 보상 모델을 사용해 언어 모델이 고품질 응답을 생성하도록 최적화하지만, 이 방법은 감독 학습보다 복잡하고 상당한 계산 비용이 든다.

이 논문은 명시적인 보상 모델링이나 강화 학습 없이도 언어 모델을 인간의 선호에 맞게 최적화할 수 있는 Direct Preference Optimization (DPO) 알고리즘을 제안한다. DPO는 기존 RLHF 알고리즘과 유사한 목표를 달성하지만, 구현과 학습이 더 간단하고 직관적이다. 이 방법은 선호하는 응답을 강화하고 모델 퇴화를 방지하기 위해 동적 중요도 가중치를 사용한다. 또한, DPO는 이론적 선호 모델을 바탕으로 정책 직접에 대한 선호 손실을 정의하며, 인간 선호 데이터를 기반으로 간단한 이진 교차 엔트로피 목적을 사용하여 최적의 정책을 도출할 수 있다.

선호도 기반 언어 모델 훈련을 위한 간단한 Direct Preference Optimization (DPO) 알고리즘을 제안하였다. 실험 결과, DPO는 감정 조절, 요약, 대화 등의 작업에서 최대 6B parameter 언어 모델을 사용하여 기존의 PPO 기반 RLHF 방법만큼 효과적임을 입증하였다.

---

## Related Work

self-supervised 언어 모델은 instruction-tuning을 통해 downstream 작업 성능과 사용자 의도 일치를 크게 향상시킬 수 있다. relative human judgment을 사용하여 LLM을 미세 조정하는 것이 전문가 시연보다 수월하며, 이는 번역, 요약, 스토리텔링, 지시사항 따르기 등의 숙련도를 향상시킨다. 이 과정에서는 먼저 선호도 데이터셋과 호환되는 보상 함수를 최적화한 후, REINFORCE, PPO 같은 강화 학습 알고리즘으로 언어 모델을 미세 조정한다. 또한, 인간 피드백을 활용한 LLM은 추가적인 합성 선호도 데이터 생성에 사용된다. 이는 강화 학습과 인간 선호도 학습을 융합한 연구 분야를 대표하며, RL 없이 상대적 선호도를 최적화하는 이론적 접근을 제공한다.

언어 맥락을 벗어난 선호도 기반 학습은 bandit과 강화 학습 환경에서 모두 연구되었다. contextual dueling bandit(CDB)은 보상 대신 행동의 선호도를 사용하며, 폰 노이만 승자 개념으로 최적 정책을 대체한다. preference-based RL(PbRL)은 알려지지 않은 점수 함수에서 생성된 이진 선호도로부터 학습하며, 대부분 잠재적인 점수 함수를 추정하고 최적화하는 과정을 거친다. 이 연구에서는 선호도를 만족시키는 정책을 직접 최적화하는 새로운 단일 단계 학습 접근 방식을 제안한다.

---

## Preliminaries


---

## Reference

* [Paper](https://arxiv.org/pdf/2305.13048.pdf)
* [GitHub](https://github.com/BlinkDL/RWKV-LM)