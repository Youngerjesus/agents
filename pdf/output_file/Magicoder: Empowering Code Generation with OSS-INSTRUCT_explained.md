
## Abstract

이 논문은 Magicoder라는 코드 생성에 특화된 대형 언어 모델(LLMs) 시리즈를 소개합니다. Magicoder는 오픈 소스 코드 스니펫을 활용하여 다양한 명령 데이터를 생성하는 OSS-INSTRUCT라는 새로운 접근 방식을 사용하여 훈련된 모델입니다. 이러한 모델은 75,000개의 합성 명령 데이터를 기반으로 학습되었으며, 모델의 매개변수가 70억 개 이하임에도 불구하고 최상위 코드 모델들과의 성능 격차를 크게 줄였습니다. 우리의 주요 목표는 매직코더가 더욱 현실적이고 제어 가능한 데이터를 생산할 수 있도록 오픈 소스 참조 자료를 활용하여 LLM들이 생성한 합성 데이터의 내재적 편향을 완화하는 것입니다. 

OSS-INSTRUCT는 Evol-Instruct 같은 다른 데이터 생성 방법과 결합되어 확장된 MagicoderS를 구축할 수 있게 합니다. Magicoder 및 MagicoderS는 코드 기준 확인에서 유사하거나 더 큰 크기의 최첨단 코드 모델들을 능가합니다. 특히, CODELLAMA 기반의 Magicoder -CL-7B 모델은 HumanEval $^{+}$ 테스트에서 ChatGPT를 넘어서는 성능을 보여주었습니다(pass $@1$에서 66.5 vs. 65.9). 전반적으로, OSS-INSTRUCT는 풍부한 오픈 소스 참조를 사용하여 다양한 합성 명령 데이터를 생성하는 새로운 방향을 제시합니다.

모델이 닫힌 소스(GPT-3.5 Turbo 및 GPT-4) 지배의 중에도, SELF-INSTRUCT는 LLMS의 명령 수행 능력을 향상시키기 위해 사용되었습니다. 예를 들어, "Code Alpaca"는 ChatGPT를 활용하여 21개의 시드 태스크를 바탕으로 자동 생성된 20,000개의 코드 명령을 포함하고 있습니다. 코딩 능력을 더욱 향상시키기 위해 Code Evol-Instruct가 시드 코드 명령의 복잡성을 증가시키기 위한 다양한 방법을 사용하여, 오픈 소스 모델 중 SOTA(최첨단) 성과를 달성했습니다.

---

## 1. Introduction

이 논문은 코드 생성, 즉 프로그램 합성(program synthesis) 문제에 대한 새로운 접근법을 제안합니다. 전통적으로, 코드 생성은 추상화 기반 합성을 포함한 다양한 상징적 접근법이 사용되어왔으나, 이 방법들은 사전 정의된 태스크나 제한된 휴리스틱에 크게 의존하는 한계를 가지고 있습니다. 예를 들어, Code Alpaca와 같은 방법은 21개의 시드 태스크만을 사용하고, Code Evol-Instruct는 5개의 휴리스틱만 사용하여 데이터셋을 발전시킵니다. 이러한 방식들은 LLM의 시스템 편향이나 태스크 편향을 그대로 수용할 가능성이 큽니다.

이에 따라, 이 논문은 오픈 소스 코드 스니펫에서 직접 학습하여 다양한 창의적 코드 지시문을 작성할 수 있는 OSS-INSTRUCT를 제안합니다. OSS-INSTRUCT를 통해, LLM은 오픈 소스 코드를 활용하여 새로운 코딩 문제를 자동으로 생성하고, 결과적으로 75K개의 합성 데이터를 생성하여 CODELLAMA-PYTHON-7B 모델을 튜닝해서 Magicoder-CL을 만듭니다. 이 방법은 기존 데이터 생성 방법과 독립적이면서도 상호보완적으로 작용할 수 있어 모델의 코딩 능력을 크게 향상시킵니다.

Magicoder 및 MagicoderS는 다양한 코딩 태스크에서 뛰어난 성능을 보였으며, 특히 HumanEval 및 MBPP, MultiPL-E, DS-1000과 같은 벤치마크에서 매우 높은 성과를 보였습니다. Magicoder-CL은 기존의 WizardCoder-CL-7B 등의 모델을 능가했으며, 특히 MagicoderS-CL은 ChatGPT를 HumanEval에서 근접하게 따라잡고 HumanEval+에서는 능가했습니다.

이어지는 연구로, DeepSeek-Coder 시리즈에 대한 언급도 있으나, 구체적인 기술적 세부정보가 부족하여 간단히 논의됐습니다. OSS-INSTRUCT를 DeepSeek-Coder에 적용하여 Magicoder-DS 및 MagicoderS-DS를 생성했으며, 더 효율적인 데이터 사용으로 높은 성능을 기록했습니다.

OSS-INSTRUCT의 디자인은 오픈 소스 프로젝트로부터 코드 조각을 직접 가져오는 대신, 이를 기반으로 명령 데이터를 생성함으로써 더 나은 성능을 달성하는 것을 실험적으로 보였습니다. 

결론적으로, 이 논문은 다양한 LLM의 성능을 크게 향상시킬 수 있는 OSS-INSTRUCT라는 혁신적인 접근법을 소개했고, Magicoder 시리즈를 만들어 그 효과를 검증했습니다. 모든 자료를 오픈 소스로 제공하여 향후 연구를 촉진하고자 했습니다.

---

## 2. OSS-INSTRUCT: Instruction Tuning from Open Source

이 섹션에서는 OSS-INSTRUCT 접근법에 대해 자세히 설명합니다. 그림 1에서 볼 수 있듯이, OSS-INSTRUCT는 대형 언어 모델(예: ChatGPT)을 활용하여 오픈 소스에서 수집한 시드 코드 스니펫(예: GitHub)을 기반으로 코딩 문제와 그 해결책을 생성하도록 합니다. 이 시드 코드 스니펫은 생성 과정에 제어 가능성을 제공하며, LLM이 실세계 프로그래밍 시나리오를 반영할 수 있도록 다양한 코딩 문제를 생성하도록 장려합니다. 이를 통해 더욱 현실적인 코딩 명령이 가능한 환경을 조성하며, 다양하고 창의적인 코딩 문제를 만들어냅니다.

---

## 2.1. Generating Coding Problems

OSS-INSTRUCT는 오픈 소스에서 쉽게 수집할 수 있는 시드 코드 스니펫을 기반으로 작동합니다. 이 연구에서는 StarCoder가 학습된 데이터셋인 The Stack(Kocetkov et al., 2022)의 필터링된 버전인 starcoderdata를 시드 코퍼스로 직접 채택했습니다. starcoderdata는 광범위하게 사용되며, 다양한 프로그래밍 언어로 작성된 고품질 코드 스니펫이 대량으로 포함되어 있을 뿐만 아니라, 데이터 정화 작업이 완료된 자료를 제공합니다(Li et al., 2023; Allal et al., 2023). 

코퍼스의 각 코드 문서에서 1-15개의 연속된 줄을 무작위로 추출하여 모델이 이를 기반으로 영감을 얻어 코딩 문제를 생성하도록 합니다. 총 80,000개의 초기 시드 스니펫이 수집되었으며, 파이썬에서 40,000개, C++, 자바, TypeScript, Shell, C#, Rust, PHP, 그리고 Swift 각각에서 5,000개씩 확보했습니다. 이후, 각 수집된 시드 코드 스니펫은 부록 A.1에 제시된 프롬프트 템플릿에 적용되며, 이 템플릿을 입력으로 받아 교사 모델이 코딩 문제와 그 해결책을 출력합니다.

---

## 2.2. Data Cleaning and Decontamination

데이터 정제 및 오염 제거 과정에서는 동일하거나 동일한 시드 코드 스니펫을 공유하는 샘플을 제외하여 데이터 정제를 수행합니다. 생성된 데이터에 다른 종류의 잡음(예: 불완전한 해결책)이 존재할 수 있지만, Honovich et al. (2023)의 연구에 영감을 받아 그러한 데이터는 여전히 LLM이 학습하기에 유용한 정보를 포함하고 있다고 판단하여 제거하지 않았습니다. 실험에 관한 더 자세한 내용은 부록 C.3에서 확인할 수 있습니다.

마지막으로, StarCoder Li et al. (2023)와 동일한 논리를 적용하여 훈련 데이터를 오염 제거합니다. 여기에는 HumanEval(Chen et al., 2021)과 MBPP(Austin et al., 2021)의 도큐멘트 문자열이나 해결책, APPS(Hendrycks et al., 2021)의 도큐멘트 문자열, DS1000(Lai et al., 2022)의 프롬프트, GSM8K(Cobbe et al., 2021)의 질문을 포함하는 코딩 문제를 제거하는 것이 포함됩니다. 우리의 분석에서는 이 오염 제거 절차가 추가적으로 9개의 샘플을 필터링할 뿐임을 보여줍니다. 시드 코퍼스인 starcoderdata가 이미 철저한 데이터 오염 제거 과정을 거쳤기 때문에, 이러한 관찰은 OSS-INSTRUCT가 시드 외의 추가적인 데이터 누출을 일으킬 가능성이 낮음을 시사합니다. 최종적으로 OSS-INSTRUCT 데이터셋은 약 75,000개의 항목을 포함하고 있으며, 데이터셋에 대한 개요는 부록 A.3에서 확인할 수 있습니다.

---

## 2.3. Qualitative Examples of OSS-INSTRUCT

이 섹션에서는 OSS-INSTRUCT가 시드 코드 스니펫으로부터 LLM이 어떻게 새로운 코딩 문제와 해결책을 창출할 수 있는지에 대한 질적 예시를 보여줍니다. 

예를 들어, 쉘 스크립트 예시에서는 단 한 줄의 쉘 스크립트로 LLM이 파이썬 코딩 문제를 생성하는 과정을 보여줍니다. 라이브러리 import 예시에서는 LLM이 몇 개의 import 문만으로 현실적인 머신러닝 문제를 만들어 내는 경우를 설명합니다. 또한, 'class' 시그니처 사례는 'SpringBootApplication'과 같은 주석 및 'bank'라는 키워드가 포함된 불완전한 클래스 정의에서 영감을 얻어 완전한 은행 시스템을 구현해야 하는 문제를 생성하는 능력을 보여줍니다. 

전반적으로, OSS-INSTRUCT는 다양한 코드 구조와 의미를 통해 알고리즘 도전 과제, 현실 문제, 단일 함수 코드 생성, 라이브러리 기반 프로그램 완료, 전체 프로그램 개발, 심지어 전체 애플리케이션 구축 같은 다양한 코딩 태스크를 창출할 수 있도록 LLM에 영감을 제공합니다.

그림 3은 HumanEval과 OSS-INSTRUCT를 포함한 여러 방법으로 생성된 합성 데이터의 코사인 유사도를 보여줍니다. 

HumanEval과의 유사성을 연구하기 위해, 우리는 75,000개의 데이터셋 샘플 각각을 164개의 HumanEval(Chen et al., 2021) 샘플과 쌍을 이루고, TF-IDF(SPARCK JONES, 1972) 임베딩을 사용하여 코사인 유사도를 계산했습니다. 각 OSS-INSTRUCT 샘플은 가장 높은 유사도 점수를 가진 HumanEval 샘플과 연결됩니다. 우리는 또한 SELF-INSTRUCT를 적용한 20,000개 데이터셋 Code Alpaca와 110,000개 코딩 지시문을 포함한 Evol-Instruct의 오픈 소스 재현본인 evol-codealpaca-v1과 비교 분석합니다. 모든 데이터셋은 §2.2에서 언급한 방식으로 오염 제거 절차를 거쳤습니다.

그림 3의 결과에 따르면, OSS-INSTRUCT는 연구된 모든 데이터 생성 기법들 중에서 가장 낮은 평균 유사도를 보이는 반면, SELF-INSTRUCT는 가장 높은 평균 유사도를 보였습니다. 이는 OSS-INSTRUCT의 개선이 단순히 동일한 분포에서 데이터를 포함하는 것에 기인하는 것은 아님을 시사합니다.

---

## 3. Evaluation

이 논문에서는 평가를 위해 CODELLAMA-PYTHON-7B 및 DeepSeekCoder-Base 6.7B를 기반 대형 언어 모델(LLMs)로 선택합니다. Magicoder 시리즈를 도출하기 위해, 먼저 OSS-INSTRUCT를 통해 생성된 75,000개의 합성 데이터로 모델을 미세 조정합니다. 그런 다음, 약 110,000개의 샘플을 포함하는 오픈 소스 Evol-Instruct 구현인 evol-codealpaca-v1 데이터셋을 사용하여 Magicoder의 추가 미세 조정을 통해 MagicoderS를 얻습니다. 구현 세부 사항과 추가 평가 결과는 부록 B와 C에 나와 있습니다. 또한, 부록 D에서는 명령 조정의 효과성을 반영하는 흥미로운 사용 사례를 제시하고, 부록 E에서는 Magicoder의 복잡한 프로그램 생성 능력을 시연합니다.

---

## 3.1. Python Text-to-Code Generation

이 섹션에서는 파이썬 코드 생성에 대한 평가를 설명합니다. HumanEval(Chen et al., 2021)과 MBPP(Austin et al., 2021)는 코드 생성의 대표적 벤치마크로, 각각의 태스크는 문서 문자열(docstring)을 프롬프트로 제공하며 LLM들이 생성된 코드의 정확성을 테스트 케이스로 확인합니다. 이러한 테스트가 부족할 수 있기 때문에, 우리는 더 엄격한 평가를 위해 EvalPlus 프레임워크(Liu et al., 2023b)를 활용하여 HumanEval+와 MBPP+를 사용했으며, 각각 80배, 35배 더 많은 테스트를 제공합니다. 이전 연구(Liu et al., 2023b; Chen et al., 2023)를 따른 평가에서는 태스크와 LLM에 대해 탐욕적 디코딩(greedy decoding)을 사용하여 하나의 샘플을 생성하고, pass $@1$ 메트릭 비교를 중심으로 진행했습니다.

표 1은 HumanEval(+)과 MBPP(+)에서 다른 LLM들의 pass $@1$ 결과를 보여줍니다. 이 결과에서 Magicoder-CL이 기본 CODELLAMA-PYTHON-7B보다 상당히 개선되었음을 먼저 확인할 수 있으며, CODELLAMA-PYTHON-34B와 WizardCoder-CL-34B를 제외한 모든 오픈 소스 모델보다 뛰어난 성능을 보입니다. 특히, Magicoder-CL은 WizardCoder-SC-15B를 능가하고, HumanEval 및 HumanEval+에서 CODELLAMA-PYTHON-34B에 비해 상당한 개선을 보여줍니다. Magicoder-CL은 또한 EvolInstruct 방법으로 훈련되면서 추가적인 성능 향상을 보였습니다. MagicoderS-CL은 HumanEval+에서 ChatGPT와 다른 모든 오픈 소스 모델을 능가합니다. 비록 HumanEval에서 WizardCoder-CL-34B와 ChatGPT보다 약간 낮은 점수를 기록했지만, 보다 엄격한 HumanEval+ 데이터셋에서는 두 모델을 모두 능가하여, MagicoderS-CL이 더욱 견고한 코드를 생성할 수 있음을 시사합니다.

---

## 3.2. Multilingual Code Generation

이 섹션에서는 파이썬 외에도 Java, JavaScript, C++, PHP, Swift, Rust의 6가지 널리 사용되는 프로그래밍 언어에 대해 MultiPL-E 벤치마크(Cassano et al., 2022)를 사용하여 광범위한 평가를 수행합니다. WizardCoder 논문(Luo et al., 2023b)에서 제공된 결과를 참고하고, bigcode-evaluation-harness(Ben Allal et al., 2022)를 통해 일관되게 모델을 평가했습니다. 이 프레임워크에서 지원되지 않는 ChatGPT 및 GPT-4와 같은 독점 모델은 분석에 포함하지 않았습니다. 또한, WizardCoder-CL-7B를 우리의 환경에서 사용하는 동안 상당한 추론 지연이 발생했기 때문에 이 모델은 분석에서 제외했습니다.

결과는 Magicoder-CL이 모든 연구된 프로그래밍 언어에서 기본 모델인 CODELLAMA-PYTHON-7B보다 크게 개선되었음을 나타냅니다. 더 나아가, Magicoder-CL은 SOTA 15B 모델인 WizardCoder-SC보다 절반의 프로그래밍 언어에서 더 나은 결과를 얻었습니다. Magicoder-CL은 모든 프로그래밍 언어에서 Magicoder-CL에 비해 추가적인 개선을 보여주었으며, 7B의 매개변수만으로도 WizardCoder-CL-34B와 유사한 성능을 달성했습니다. 주목할 점은, Magicoder-CL이 매우 제한된 다국어 데이터를 사용해 훈련되었음에도 불구하고, 유사하거나 더 큰 크기의 다른 LLM보다 뛰어난 성능을 보였다는 것입니다. 또, 하네스는 기본 모델을 위한 코드 완성 포맷으로 모델을 평가함에도 불구하고, Magicoder는 명령 조정만으로도 상당한 개선을 보여주었습니다. 이는 LLM이 데이터의 포맷을 넘어 지식을 학습할 수 있음을 시사합니다.

---

## 3.3. Code Generation for Data Science

이 섹션에서는 DS-1000 데이터셋(Lai et al., 2022)을 사용하여 데이터 과학을 위한 코드 생성 평가를 설명합니다. DS-1000은 파이썬의 7개의 인기 있는 데이터 과학 라이브러리를 활용하여 1,000개의 고유한 데이터 과학 코딩 문제를 포함하고 있으며, LLM의 현실적이고 실용적인 사용 사례를 평가합니다. 각 문제에 대한 단위 테스트를 제공하여 문제를 검증합니다. DS-1000에는 코드 완성과 삽입 모드가 있지만, 여기서는 CODELLAMA-PYTHON이 채우기(infilling)를 지원하지 않기 때문에 코드 완성만을 평가했습니다.

표 3에서는 최근의 INCODER(Fried et al., 2023), CodeGen(Nijkamp et al., 2023), Code-Cushman-001(Microsoft, 2023a), StarCoder(Li et al., 2023), CODELLAMA-PYTHON(Roziere et al., 2023), WizardCoder(Luo et al., 2023b)를 포함한 평가 결과를 보여줍니다. 표에서 볼 수 있듯이, Magicoder-CL-7B는 이미 평가된 모든 기준 모델을 초과했으며, 여기에는 최신의 WizardCoder-CL-7B 및 WizardCoder-SC-15B도 포함됩니다. MagicoderS-CL-7B는 WizardCoder-SC-15B에 비해 8.3퍼센트 포인트의 절대 개선을 도입하여 그 한계를 더욱 뛰어넘었습니다.

---

## 3.4. Comparison with DeepSeek-Coder

DeepSeek-Coder(Guo et al., 2024)는 우리가 작업을 진행하는 동안 동시에 발표된 모델 시리즈로, 뛰어난 코딩 성능을 보여줍니다. 이 섹션에서는 해당 모델의 데이터 및 명령 조정 세부 정보가 작성 시점에 공개되지 않았기 때문에 간단히 논의합니다. 우리는 CODELLAMA-PYTHON-7B에 적용한 것과 동일한 미세 조정 전략을 DeepSeek-Coder-Base-6.7B에도 적용하여 Magicoder-DS를 생성했습니다. 

표 4는 표 1에서와 비슷한 경향을 보여주며, OSS-INSTRUCT를 적용한 후 기본 모델이 크게 개선될 수 있음을 나타냅니다. 특히, Magicoder-DS 변형은 훨씬 적은($\times8$) 학습 토큰을 사용하면서도 모든 벤치마크에서 DeepSeekCoder-Instruct-6.7B를 능가하며, 이러한 데이터셋에서도 DeepSeekCoder-Instruct-33B와 근접한 성능을 보였습니다.

---

## 4. Ablations of Data Source

It seems like the section content for "4. Ablations of Data Source" is missing in your request. If you can provide the specific content or main points from that section, I'll be able to help you explain it based on our previous discussion of the paper.

---

## 4.1. Impact of the Language Distribution

이 섹션에서는 훈련 데이터에 나타나는 프로그래밍 언어의 분포와 해당 언어의 다운스트림 성능 간의 상관 관계를 이해하기 위해 추가적인 ablation 연구를 수행합니다. 우리는 75K 훈련 데이터를 약 43K의 파이썬 전용 데이터와 32K의 비파이썬 데이터로 분류합니다. 이는 생성된 데이터에 ‘python’이 포함되어 있는지를 기준으로 하며, 시드 코드 스니펫을 기반으로 분류하지는 않습니다. 이는 OSS-INSTRUCT를 수행하는 LLM이 시드와 다른 언어로 코드를 생성할 수 있기 때문입니다.

표 5에서는, 동일한 훈련 하이퍼파라미터를 사용하여 다른 데이터 분할에 대해 2회차 동안 기본 모델(CODELLAMA-PYTHON-7B)을 일관되게 미세 조정한 평가 결과를 보여줍니다. 표에서 볼 수 있듯이, 파이썬 또는 비파이썬 데이터로만 훈련하면 각 모델의 파이썬 또는 비파이썬 태스크에서의 성능이 크게 향상됨을 알 수 있습니다. 흥미롭게도, 다른 프로그래밍 언어에서의 명령 조정은 여전히 전체적인 코딩 성능을 향상시키며, 이는 분포 외의 언어도 포함합니다. 예를 들어, 비파이썬 데이터로만 훈련했을 때 Magicoder-CL은 파이썬 전용 평가에서 기본 모델에 비해 10.4 퍼센트 포인트의 개선을 이뤘습니다. 이는 LLM이 다양한 프로그래밍 언어 간의 상관 관계를 설정하고 더 깊은 코드 의미론의 전이 학습을 수행할 수 있음을 시사합니다.

마지막으로, 두 소스의 데이터를 결합할 경우 파이썬 평가 성능에서 더 큰 향상을 관찰했으며, 다국어 성능에서는 약간의 감소를 보였습니다. 이는 명령 조정 동안 파이썬 데이터의 우세한 비율(약 57%)에 기인하는 것으로 해석됩니다.

---

## 4.2. OSS-INSTRUCT vs. Direct Finetuning

이 섹션에서는 오픈 소스 코드 스니펫으로부터 영감을 얻는 OSS-INSTRUCT의 접근방식과, 이 코드들을 직접 미세 조정에 사용하는 접근방식의 차이를 탐구합니다. 

한 가지 자연스러운 의문은 이 오픈 소스 코드를 직접 미세 조정에 사용하는 것이 왜 고려되지 않았는가입니다. 이를 해소하기 위해 우리는 CodeSearchNet(Husain et al., 2020)을 따라, 75K OSS-INSTRUCT 데이터셋을 구성하는 데 사용된 같은 시드 문서 코퍼스로부터 의미론적으로 관련 있는 주석-함수 쌍을 추출했습니다. 우리는 함수 서명과 주석으로부터 함수 본문을 예측하도록 모델을 훈련시켰으며, 75K 시드 스니펫과 겹치는 주석-함수 쌍을 우선시하여 약 11K 개의 데이터 포인트를 얻었습니다. 우리의 75K 샘플에 맞추기 위해, 나머지 64K 샘플은 전체 75K 시드 문서 코퍼스를 활용하여 수집되었습니다. 

CODELLAMA-PYTHON-7B 기본 모델을 2회차 동안 이 쌍 데이터로 미세 조정했으며, 결과는 표 6에 나타나 있습니다. 표를 보면, 75K 쌍 데이터를 미세 조정하는 것이 기본 모델을 오히려 악화시키는 반면, OSS-INSTRUCT는 상당한 성능 향상을 가져옵니다. 우리는 이러한 성능 저하가 쌍 데이터가 HumanEval이나 MultiPL-E 문제와 매우 유사한 형식을 가지고 있음에도 불구하고 근본적으로 존재하는 상당한 잡음과 불일치 때문이라고 추측합니다. 이는 코드 명령 조정에서 데이터의 형식이 아닌 사실성이 중요하다는 것을 보여줍니다. 또한, 이는 서로 느슨하게 관련된 코드 조각을 의미론적으로 일관된 명령 조정 데이터로 변환할 수 있는 OSS-INSTRUCT의 우수성을 나타냅니다. 

결과적으로, OSS-INSTRUCT를 사용한 모델이 주석-함수 쌍으로 직접 미세 조정한 모델보다 HumanEval+ 및 MultiPL-E에서 훨씬 더 높은 성능을 보였습니다.

---

## 4.3. OSS-INSTRUCT with A Less Powerful Teacher

이 섹션에서는 OSS-INSTRUCT의 효과가 단순히 교사 모델의 증류에 있는지 여부를 탐색합니다. 두 가지 주요 이유를 제시합니다. 첫째, 기본 모델이 이미 포괄적인 코드 데이터로 사전 훈련되어 있기 때문에, 증류 과정은 모델의 내부 능력을 활성화시켜 코딩 작업에서의 성능을 향상시키게 됩니다. 둘째, OSS-INSTRUCT는 시드 코드 스니펫을 사용하여 일회성 문제-해결 쌍을 생성합니다. 이러한 시드 스니펫은 귀중한 문맥을 제공하므로, 시드 정보가 없는 기존의 교사 모델보다 더 나은 솔루션을 생성할 수 있습니다. 이렇게 향상된 솔루션은 보다 효과적인 학생 모델을 훈련하는 데 사용될 수 있습니다.

이 점들을 검증하기 위해, 우리는 최신의 범용 오픈 소스 LLM인 Mixtral-∇8x7B-Instruct-v0.1 (Jiang et al., 2024)을 사용하여 20K OSS-INSTRUCT 데이터를 생성하는 추가 실험을 수행했습니다.

표 7은 Magicoder-CL-Mixtral-7B가 기본 CODELLAMA-PYTHON 모델에 비해 크게 개선될 뿐만 아니라, HumanEval+와 MBPP+에서 Mixtral-∇8x7B-Instruct-v0.1(즉, 교사 모델)보다도 더 나은 성능을 보임을 나타냅니다. 이러한 결과는 OSS-INSTRUCT가 단순히 교사 모델을 증류하는 것이 아니라, 기본 모델의 자체 능력을 자극하고 시드 코드 스니펫에 내포된 정보를 효과적으로 활용하고 있음을 시사합니다.

---

## 5. Related Work

이 섹션에서는 논문 내의 관련 연구들을 다루고 있습니다.

**기초 코드 모델들:** 대량의 코드라인을 학습시킨 대형 언어 모델(LLMs)은 코드 생성(Chen et al., 2021; Austin et al., 2021), 프로그램 수리(Xia & Zhang, 2022; Wei et al., 2023 등), 소프트웨어 테스트(Xia et al., 2023a 등) 등 소프트웨어 공학의 다양한 작업에서 뛰어난 성능을 보여주고 있습니다. 특히, CodeGen(Nijkamp et al., 2023), CodeT5(Wang et al., 2021), StarCoder(Li et al., 2023), CoDELLAMA(Roziere et al., 2023) 등과 같은 주요 기초 모델들은 방대한 수의 코드베이스를 처음부터 학습하며, 일반적인 코드 생성 및 이해 능력을 구축해왔습니다. 최근의 DeepSeek-Coder(Guo et al., 2024)와 StarCoder2(Lozhkov et al., 2024) 같은 코딩 LLM들은 리포지토리 레벨에서의 사전 학습 데이터를 구성하여 모델의 문맥 이해 능력을 향상시킵니다. 또한 이러한 기본 모델들은 특정 도메인 관련 코딩 작업 해결에 맞춰 미세 조정되거나(Luo et al., 2023b) 프롬프트(Cohen et al., 2023)를 통해 세부 조정됩니다.

**합성 데이터를 통한 명령 조정:** 명령 조정은 사전 훈련된 LLM의 성능을 명령과 그에 대한 응답을 포함한 데이터로 미세 조정하여 향상시키는 것을 목표로 합니다(Wei et al., 2022). 고품질 명령 데이터를 얻는 것은 종종 힘든 작업이기 때문에, 연구자들은 합성 명령 데이터를 생성하는 방법 개발에 집중하고 있습니다. Wang et al. (2023a)은 SELF-INSTRUCT를 도입하여, GPT-3 (Brown et al., 2020) 같은 기초 LLM이 정교하게 설계된 프롬프트로 합성 명령-응답 쌍을 생성하도록 했습니다. 이후 동일한 LLM을 이 합성 데이터로 명령 조정하여 자체 생성된 지식을 증류합니다. 이 기술은 다른 LLM에서도 합성 데이터를 생성하기 위해 확장되었습니다. 예를 들어, Alpaca(Taori et al., 2023)와 Code Alpaca(Chaudhary, 2023)는 ChatGPT에서 생성된 명령을 사용하여 LLAMA와 같은 모델을 미세 조정합니다. SELF-INSTRUCT를 개선하기 위해, WizardLM(Xu et al., 2023)과 WizardCoder(Luo et al., 2023a)는 ChatGPT를 휴리스틱 프롬프트로 안내하여 합성 데이터를 더 복잡하고 다양하게 만드는 Evol-Instruct와 Code Evol-Instruct를 제안했습니다. Gunasekar et al. (2023)는 최근에 교과서 수준의 합성 데이터만으로도 모델이 뛰어난 코딩 및 추론 능력을 얻을 수 있음을 보여주었습니다. 우리의 제안인 OSS-INSTRUCT는 모든 기존 방법과는 동떨어져 있으며, 현실 세계의 코드 스니펫에서 영감을 얻어 코딩 작업의 제어 가능성, 품질 및 창의성을 높이는 것을 목표로 합니다.

**코드 평가를 위한 LLMs:** 대부분의 코드 벤치마크는 자연어 설명에서 단일 함수 프로그램을 생성하는 LLM을 평가합니다. 여기에는 HumanEval(Chen et al., 2021), MBPP(Austin et al., 2021), APPS(Hendrycks et al., 2021), CodeContests(Li et al., 2022) 등이 포함됩니다. 몇 가지 수작업 테스트를 사용하여 LLM이 생성한 솔루션의 기능적 정확성을 평가하지만, 충분하지 않은 테스트는 잘못된 부정 결과를 초래할 수 있습니다. 이로 인해 EvalPlus 프레임워크(Liu et al., 2023b)는 HumanEval+와 MBPP+를 만들기 위해 80배/35배 더 많은 테스트를 추가했습니다. 데이터셋 오염 문제를 해결하기 위해, 연구자들은 모델 훈련에 포함되지 않은 새로운 코딩 문제를 컴파일하는 LiveCodeBench(Jain et al., 2024)와 기존 벤치마크를 발전시켜 새로운 코딩 작업을 생성하는 EvoEval(Xia et al., 2024)을 제안했습니다. 한편, 데이터 과학을 위한 코드 생성(DS-1000(Lai et al., 2022)), 오픈 소스 이슈 처리(SWE-bench(Jimenez et al., 2023)), 리포지토리 수준 코드 생성(CROSSCODEEVAL(Ding et al., 2023) 및 RepoEval(Zhang et al., 2023))을 평가하는 포괄적인 벤치마크도 있습니다.

---
