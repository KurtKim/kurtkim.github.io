+++
author = "Kurt"
title = "Codex"
date = "2023-12-30"
description = "Evaluating Large Language Models Trained on Code"
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

GitHub에서 공개적으로 사용 가능한 코드에 미세 조정된 GPT 언어 모델인 Codex를 소개하고 그 Python 코드 작성 능력을 연구한다. 이 모델은 GitHub Copilot를 구동하며, 새로운 평가 세트인 HumanEval에서 문제의 28.8%를 해결합니다. 또한, 모델에서 반복 샘플링은 어려운 프롬프트에 대한 해결책을 만드는 데 효과적인 전략이라는 것을 발견하였다. 모델의 한계를 조사하였고, 강력한 코드 생성 기술의 배포가 안전, 보안, 경제 등에 미치는 잠재적인 영향을 논의한다.

---

## Introduction

스케일러블한 시퀀스 예측 모델들은 자연어 처리, 컴퓨터 비전, 오디오 및 음성 처리, 생물학, 다중 모달리티 등 다양한 분야에서 생성 및 표현 학습의 일반적인 방법으로 사용되고 있다. 최근에는 언어 모델들이 대규모 데이터셋에서 코드를 활용하고 이를 통해 학습된 프로그래밍 능력을 바탕으로 프로그램 합성이라는 도전적인 문제를 해결하는데 기여하고 있다. 또한, masked language modeling과 span prediction과 같은 인기 있는 언어 모델링 방법들이 프로그래밍 학습을 위해 적용되고 있다.

초기 GPT-3 연구에서는 Python docstrings로부터 간단한 프로그램을 생성할 수 있다는 사실을 발견하였다. 이는 GPT-3가 명시적으로 코드 생성을 위해 학습되지 않았음에도 불구하고 가능했다. 이러한 성공과 공개적으로 사용 가능한 코드의 풍부함을 바탕으로, Codex라는 특화된 GPT 모델이 다양한 코딩 작업에서 탁월하게 수행될 수 있을 것이라고 가정하였다. 이 논문은 GitHub Copilot과 OpenAI API에 사용된 초기 Codex 모델들에 대해 설명하고 있다.

이 연구에서는 docstrings에서 Python 함수를 생성하는 작업에 집중하고, 이를 유닛 테스트를 통해 자동으로 평가한다. 이를 위해 언어 이해, 알고리즘, 간단한 수학을 평가하는 164개의 프로그래밍 문제 데이터셋을 만들었다.

모델로부터 여러 샘플을 생성해 유닛 테스트를 통과하는지 확인한다. 12B parameter의 Codex는 단일 샘플로 28.8%의 문제를 해결하며, 300M parameter의 Codex는 13.2%를 해결한다. 반면, 6B parameter의 GPT-J는 동일한 데이터셋에서 11.4%를 달성하며, 모든 GPT 모델은 거의 0%에 가깝다. docstrings에서 함수를 합성하는 작업을 개선하기 위해, 이 연구에서는 Codex를 독립적으로, 올바르게 구현된 함수들에 대해 미세조정하였고, 결과적으로 생성된 Codex-S 모델은 문제들 중 37.7%를 단일 샘플로 해결한다.

실제 프로그래밍 작업은 접근 방식의 반복과 버그 수정을 포함하는데, 이는 모델로부터 여러 샘플을 생성하고 모든 유닛 테스트를 통과하는 샘플을 선택하는 것으로 모사할 수 있다. 100개의 샘플 내에서, Codex-S는 문제들 중 77.5%에 대해 적어도 하나의 올바른 함수를 생성할 수 있다. 실제로, mean log-probability가 가장 높은 샘플이 문제들 중 44.5%에서 유닛 테스트를 통과하였다.

---

## Evaluation Framework

pass@k 메트릭을 정의하고 그 장점을 설명하며, 모델 평가를 위해 만든 "HumanEval"이라는 수기로 작성된 문제 데이터셋에 대해 설명한다. 마지막으로, 모델이 생성한 코드를 안전하게 실행하기 위해 사용한 샌드박스 환경에 대해 이야기한다.

### Functional Correctness

---

## Reference

* [Paper](https://arxiv.org/pdf/2107.03374.pdf)
* [Github](https://github.com/openai/human-eval)