+++
author = "Kurt"
title = "GPT-1"
date = "2023-12-02"
description = "Improving Language Understanding by Generative Pre-Training"
categories = [
    "Paper Review"
]
tags = [
    "LLM",
    "Milestone",
]
+++

## Abstract

자연어 이해는 텍스트의 함축, 질문에 대한 답변, 의미의 유사성 평가, 문서 분류 등 다양한 작업으로 구성되어 있다. 레이블이 지정된 데이터가 부족한 상황에서, 이 논문은 레이블이 없는 텍스트 데이터에 대해 언어 모델을 (생성적) 사전학습(generative pre-training)하고, 이를 특정 작업에 미세조정(fine-tuning)하는 방식을 제안한다. 이 방법은 모델 아키텍처에 최소한의 변경만을 요구하면서도 효과적인 전이를 달성하였고, 다양한 자연어 이해 벤치마크에서 우수한 성능을 보여주었다. 이 모델은 각 작업에 특별히 설계된 모델을 능가하며, 12개의 작업 중 9개에서 최고 성능을 달성하였다.

---

## Introduction

자연어 처리(NLP)에서 지도 학습의 의존성을 줄이는 것은 중요한데, 이는 대부분의 딥러닝 방법이 수동 레이블링된 대량의 데이터가 필요하기 때문이다. 이런 상황에서 레이블이 없는 데이터에서 언어 정보를 추출할 수 있는 모델은 유용한 대안이 될 수 있으며, 비지도 학습을 통해 학습하는 것이 더 나은 결과를 얻는 경우도 있다. 이를 입증하는 가장 강력한 예는 사전 학습된 단어 임베딩이며, 이는 다양한 NLP 작업에서 성능 향상을 위해 널리 사용되고 있다.

레이블이 없는 텍스트에서 단어 수준을 넘어서는 정보를 활용하는 것은 어려운 도전 과제이며, 이유는 다음과 같다. 첫째, 텍스트 표현을 학습하고 다른 곳에 유용하게 전이하는 최적화 목표가 무엇인지 확실하지 않다. 둘째, 학습된 표현을 어떤 작업에 가장 효과적으로 적용할 방법이 아직 확립되지 않았다. 이런 불확실성이 효과적인 준지도 학습 방법을 개발하는 것을 어렵게 한다.

이 연구는 언어 이해 작업에 비지도 사전 학습(unsupervised pre-training)과 지도 미세 조정(supervised fine-tuning)을 결합하는 준지도 학습을 제안한다. 목표는 적은 조정으로 다양한 작업에 적용 가능한 표현을 학습하는 것이다. 레이블이 없는 대량의 텍스트와 수동으로 레이블링된 훈련 예제를 사용하며, 학습은 두 단계로 진행된다. 먼저, 레이블이 없는 데이터로 모델의 초기 파라미터를 학습하고, 그 다음으로 지도 학습을 통해 이 파라미터를 목표 작업에 맞게 조정한다.

이 연구에서는 다양한 작업에서 뛰어난 성능을 보인 Transformer모델을 사용한다. 이 모델은 텍스트의 장기적인 의존성을 처리하는 더 구조화된 메모리를 제공하므로 강한 전이 성능을 보여준다. 전이 단계에서는 작업 특정 입력 조정을 사용하여 텍스트 입력을 연속 토큰 시퀀스로 처리하며, 이 방식은 사전 학습된 모델의 구조를 최소한으로 변경하면서 효과적으로 미세 조정할 수 있음을 실험적으로 입증한다.

이 연구는 자연어 추론, 질문 응답, 의미 유사성, 텍스트 분류 등 네 가지 언어 이해 작업에서 모델을 평가하였다. 제시된 모델은 각 작업에 특화된 모델들보다 더 우수한 성능을 보여주었고, 12개 작업 중 9개에서 최고 성능을 보여주었다.

---

## Related Work

### Semi-supervised learning for NLP

이 연구는 자연어에 대한 준지도 학습 범주에 속하며, 이는 시퀀스 라벨링이나 텍스트 분류와 같은 작업에 적용된다. 초기에는 레이블 없는 데이터를 사용해 단어나 구문 수준의 통계를 계산하였지만, 최근에는 레이블이 없는 말뭉치에서 훈련된 단어 임베딩을 활용하여 작업 성능을 향상시키는 방향으로 연구가 진행되고있다. 그러나 이 논문의 목표는 단어 수준 이상의 의미를 포착하는 것이며, 이를 위해 구문이나 문장 수준의 임베딩을 활용하여 텍스트를 벡터 표현으로 인코딩하는 방식을 채택하였다.

### Unsupervised pre-training

비지도 사전 학습은 좋은 초기화 지점을 찾는 것을 목표로 하며, 이미지 분류, 음성 인식, 엔티티 구분, 기계 번역 등 다양한 작업에서 DNN의 훈련을 돕는데 사용되고있다.

이 연구는 언어 모델링 목표를 사용하여 신경망을 사전 학습하고, 지도 학습으로 목표 작업에서 미세 조정하는 방식을 따른다. 이 방법은 LSTM을 사용하는 이전의 방법들이 제한적인 예측 능력을 가지는 반면, Transformer는 더 넓은 범위의 언어 구조를 포착할 수 있게 한다. GPT 모델은 자연어 추론, 패러프레이즈 감지, 스토리 완성 등 다양한 작업에서 효과를 보여주었으며, 다른 모델이 새로운 파라미터를 많이 필요로 하는 반면, GPT 모델은 아키텍처에 최소한의 변경만 필요로 한다.

### Auxiliary training objectives

보조적인 비지도 학습 목표 추가는 준지도 학습의 변형 형태로, 다양한 NLP 작업을 통해 의미 역할 라벨링을 개선하는데 사용되었다. 최근에는 이러한 보조 목표를 목표 작업에 추가하여 시퀀스 라벨링 작업에서 성능을 향상시켰다. 이 연구에서도 비지도 사전 훈련이 이미 목표 작업과 관련된 다양한 언어적 요소를 학습한다는 것을 보여준다.

---

## Framework

학습은 큰 말뭉치에서 대용량 언어 모델을 학습하는 단계와 레이블이 달린 데이터를 활용해 모델을 목표 작업에 맞게 미세 조정하는 단계로 이루어진다.

### Unsupervised pre-training

비지도 토큰 말뭉치 $U = \lbrace u_1, ... , u_n \rbrace $ 가 주어질때, 다음 Likelihood를 최대화하도록 표준언어모델링 목적함수를 사용한다:

$$ L_1(U) = \sum_{i} \log{P} (u_i | u_{i-k}, ... , u_{i-1}, \theta) $$

$k$는 context window의 크기이며, 조건부 확률 $P$는 parameter $\theta$를 가진 신경망을 사용하여 모델링된다. 이 parameter들은 stochastic gradient descent를 사용하여 학습된다.

GPT 모델은 언어모델로 multi-layer Transformer decoder를 사용하며, 이 모델은 입력 컨텍스트 토큰에 대해 multi-headed self-attention을 적용한 후, position-wise feedforward layer를 적용하여 목표 토큰에 대한 출력 분포를 생성한다:

$$ h_0 = UW_e + W_p $$
$$ h_l = \text{transformer_block}(h_{l-1}) \forall i \in [1, n] $$
$$ P(u) = \text{softmax}(h_n W^T_e) $$

$U = (u_{i-k}, ... , u_{i-1}) $ 는 토큰의 컨텍스트 벡터이고, $n$은 layer의 수, $W_e$ 는 토큰 임베딩 행렬, $W_p$ 는 위치 임베딩 행렬이다.

### Supervised ﬁne-tuning

모델을 학습한 후, parameter를 목표 작업에 맞게 조정한다. 레이블이 지정된 데이터셋 $C$ 는 입력 토큰 $x^1, ... , x^m $ 과 레이블 $y$로 구성된다. 입력은 사전 훈련된 모델을 통과하여 최종 transformer block의 활성값인 $h^m_l$ 을 얻으며, 이는 parameter $W_y$ 와 함께 선형 출력층으로 전달되어 $y$ 를 예측한다:

$$ P(y|x^1, ... , x^m) = \text{softmax}(h^m_l W_y) $$

이는 다음을 최대화 한다.

$$ L_2(C) = \sum_{(x,y)} \log{P(y|x^1, ... , x^m)} $$

추가로 미세 조정을 위한 보조 목표로 언어 모델링을 포함시키는 것은 지도 모델의 일반화를 향상시키고, 수렴을 가속화하는데 도움이 된다. 구체적으로, weight $\lambda$에 대해 다음을 최적화한다:

$$ L_3(C) = L_2(C) + \lambda L_1(C) $$

미세 조정 과정에서 추가 매개변수는 $W_y$ 와 구분자 토큰의 임베딩뿐이다.

### Task-speciﬁc input transformations

텍스트 분류같은 일부 작업들은 모델을 직접 미세 조정할 수 있지만, 질문 답변이나 텍스트 함의 같은 작업들은 구조화된 입력을 필요로 하는데, 이러한 입력에 대해 사전 학습된 모델은 별도의 수정 없이도 처리할 수 있다. 대신, 이런한 입력을 모델이 처리할 수 있는 순서가 있는 시퀀스로 변환한다. 이 접근법은 작업 간에 아키텍처를 크게 변경할 필요를 없애준다. 또한, 모든 변형에는 무작위로 초기화된 시작과 종료 토큰을 포함한다. 

![](images/figure1.png)

#### Textual entailment

텍스트 함의에서는, 전제 $p$와 가설 $h$를 구분자 `$`로 연결한다.

#### Similarity

유사성 경우, 비교되는 두 문장의 순서는 정해져 있지 않으므로, 텍스트 두 개를 다른 순서로 이어붙여 각각을 독립적으로 처리하여 두 시퀀스 표현 $h^m_l$을 생성한다.

#### Question Answering and Commonsense Reasoning

컨텍스트 문서 $z$, 질문 $q$, 가능한 답변들 $\lbrace a_k \rbrace$을 받는다. 각 가능한 답변을 문맥 문서와 질문에 연결하고, 구분자 토큰을 추가해 시퀀스 $[z; q;$ `$`; $a_k]$ 를 만든다. 이 시퀀스들은 독립적으로 처리되고, softmax 계층을 통해 정규화되어 답변들에 대한 출력 분포를 생성한다.

---

## Experiments

### Setup

#### Unsupervised pre-training

언어 모델 학습에 BooksCorpus 데이터셋을 사용한다. 이는 다양한 장르의 7천개가 넘는 미발행 책들을 포함하며, 연속적인 긴 텍스트를 통해 모델이 long term depency를 학습할 수 있다. ELMo에서 사용된 1B Word Benchmark 데이터셋은 문장들이 서로 섞여 있어 long term depency를 학습하기 어렵다.

![](images/table1.png)

#### Model speciﬁcations

|Hyperparameter|Descrption|
|:---:|:---:|
|layer|12-layer decoder-only transformer with masked self-attention heads|
|state dimension|decoder: 768, attention heads: 12, position-wise FFN: 3072|
|optimizer|Adam|
|learning rate|max: 2.5e-4, schedule: cosine annealing, warm-up step: 2,000|
|schedule|100 epochs|
|batch size|64 random sample $\times$ 512 token/sample|
|weight initialization|$N(0, 0.02)$|
|subword segmentation|BPE (40,000 merges)|
|dropout|0.1|
|regularization|L2($w=0.01$)|
|activation function|Gaussian Error Linear Unit(GELU)|
|position embedding|learned positoin embeddings|
|pre-processing|cleaning: ftfy, tokenizer : spaCy|

#### Fine-tuning details

명시되지 않은 것들은 사전학습에 사용된 hyperparameter를 재사용했다.

|Hyperparameter|Descrption|
|:---:|:---:|
|dropout|0.1|
|Learning rate|max: 6.25e-5, warm-up: 0.2% of training|
|batch size|32|
|epochs|3|
|auxiliary objective weight($\lambda$)|0.5|

### Supervised ﬁne-tuning

자연어 추론, 질문 응답, 의미론적 유사성, 텍스트 분류등의 평가를 진행하였고, 그 중 일부는 GLUE benchmark에 포함되어 있다.

#### Natural Language Inference

자연어 추론(NLI) 작업, 즉 텍스트 함의를 인식하는 것은 문장 쌍을 읽고, 그들 사이의 관계를 함의, 모순 또는 중립 중 하나로 판단하는 것으로, 이미지 캡션(SNLI), 텍스트 변환된 연설, 대중 소설, 정부 보고서(MNLI), 위키백과 기사(QNLI), 과학 시험(SciTail) 또는 뉴스 기사(RTE)를 포함한 다양한 출처의 다섯 개의 데이터셋을 사용해서 평가하였다. 

![](images/table2.png)

다섯 가지 데이터셋 중 네 가지에서 좋은 성능을 보여주었으며, MNLI에서 1.5%, SciTail에서 5%, QNLI에서 5.8%, SNLI에서 0.6%의 성능 향상을 보였다. 이는 GPT 모델이 여러 문장을 더 잘 이해하고, 언어적 모호성의 측면을 처리할 수 있다는 것을 보여준다.

#### Question answering and commonsense reasoning

질문 응답 작업은 한 문장이나 여러 문장을 이해하는 능력을 평가한다. 중고등학교 시험의 영어 지문과 질문이 포함된 RACE 데이터셋을 사용한 평가에서 좋은 성능을 보여주었다. 또한, 여러 문장의 이야기 중에서 올바른 결말을 고르는 Story Cloze 평가에서도 GPT 모델은 이전 최고 성능을 크게 능가하였다. 이 결과는 GPT 모델이 넓은 범위에 걸친 문맥 정보를 잘 처리할 수 있음을 보여준다.

![](images/table3.png)

#### Semantic Similarity

의미론적 유사성(또는 패러프레이즈 감지) 작업은 두 문장이 의미적으로 동일한지 여부를 판단한다. 뉴스 출처에서 수집된 Microsoft Paraphrase(MRPC), Quora Question Pairs(QQP), 그리고 Semantic Textual Similarity benchmark(STS-B) 데이터셋을 사용한다. 이 중 STSB와 QQP에서 좋은 성늘을 보여주었다.

![](images/table4.png)

#### Classiﬁcation

텍스트 분류로 사용한 데이터셋은 문법적으로 맞는지를 판단하는 Corpus of Linguistic Acceptability(CoLA)와 단순 이진분류 평가인 Stanford Sentiment Treebank(SST-2)을 사용하였다. CoLA에서 35.0 에서 45.4점으로, SST-2에서 68.9 에서 72.8점으로 상승하였으며, GLUE benchmark에서도 72.8점으로 이전 최고 성능을 크게 능가하였다.

GPT모델은 평가한 12개의 데이터셋 중 9개에서 state-of-the-art를 달성하였다. 그리고 STS-B(약 5.7k)와 같은 작은 데이터셋부터 가장 큰 SNLI(약 550k)와 같은 크기의 다양한 데이터셋에서 잘 작동함을 보여준다.

---

## Analysis

### Impact of number of layers transferred

unsupervised pre-training에서 supervised target task로 transfer하는 layer 개수의 영향을 분석했다. MultiNLI와 RACE에서 성능을 관찰했고 transferring embeddings이 성능을 향상시킨다는 것과 각 transformer layer가 최대 9%까지 성능을 향상시킨다는 결과를 얻었다. 이는 pre-trained model의 각 layer가 target task를 푸는 데 유용한 기능을 포함함을 의미한다.

![](images/figure2.png)

### Zero-shot Behaviors

Trasformer를 사용한 language model이 pre-training에 효과적인 이유에 대한 가설로, Generative model이 학습하는 target tasks가 language modeling의 성능을 향상에 도움을 준다고 생각했고, 이를 검증하기 위해 pre-training 업데이트 횟수에 따른 target tasks의 성능을 fine-tuning없이 측정하였다.

실험 결과 pre-training 업데이트 횟수에 따라 안정적 & 지속적으로 관련 taget task의 성능이 증가하는 것을 확인할 수 있었으며 이는 generative pre-training이 관련 task의 학습에 도움을 준다는 것을 의미한다. 반면, LSTM의 경우에는 업데이트 횟수에 따라 일관되게 안정적으로 증가하지 않고 분산을 가지면서 증가하는데, 이는 LSTM 보다 더 구조화된 transformer의 attentional memory가 transfer learning에 도움을 준다는 것을 의미한다.

### Ablation studies

세 가지 ablation study를 통해 다음의 결과를 얻었다. 첫째, 미세조정 시 보조 목적함수의 도움이 큰 데이터셋에서는 두드러지지만 작은 데이터셋에서는 그렇지 않다는 것을 확인하였다. 둘째, LSTM과 Transformer를 비교한 결과, LSTM은 오직 MRPC 데이터셋에서만 Transformer를 능가하는 것을 확인하였다. 마지막으로, 사전학습 없이 지도학습을 진행한 Transformer는 모든 작업에서 성능이 저하되었다. 

![](images/table5.png)

---

## Conclusion

생성적 사전 학습과 미세조정을 사용한 모델을 통해 강력한 자연어 이해를 구현하였다. GPT 모델은 연속된 텍스트로 이루어진 다양한 말뭉치로 사전학습된 모델은 일반 지식(world knowledge)과 long term depency 처리하는 능력을 가질 수 있었다. 이를 통해, 우리는 지도학습 없이도 특정 작업의 성능을 향상시키는 것이 가능하다는 것을 보여주었으며, 특히 Trasformer 모델과 long term depency가 있는 텍스트 데이터셋이 이 접근법에서 잘 작동함을 확인하였다.

---

## Reference

* [Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
* [Github](https://github.com/openai/finetune-transformer-lm)