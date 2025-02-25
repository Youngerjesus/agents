
## ABSTRACT

**요약 설명:**  
이 논문은 대규모 언어 모델을 특정 작업과 사용자 선호에 맞추기 위한 핵심 기술인 **지시 튜닝(Instruction Tuning)**의 데이터 선택 전략을 탐구합니다. 기존 연구는 적절한 데이터 선택이 적은 양으로도 우수한 성능을 낼 수 있음을 보였지만, "좋은 데이터"의 기준과 자동 선택 방법은 명확히 규명되지 않았습니다.  

**핵심 내용:**  
1. **데이터 측정 3가지 차원**:  
   - **복잡성(Complexity)**: 데이터의 난이도와 정보 밀도  
   - **품질(Quality)**: 정확성과 유용성  
   - **다양성(Diversity)**: 주제와 형식의 다양성  
   기존 방법을 분석하고, 이를 개선한 새로운 측정 기법을 제안합니다.  

2. **DEITA (Data-Efficient Instruction Tuning for Alignment)**:  
   - LLaMA와 Mistral 모델을 기반으로, **자동 선택된 6,000개의 SFT 데이터**만으로 미세 조정한 모델 시리즈입니다.  
   - 기존 오픈소스 모델 대비 **10배 이상 적은 데이터**로 동등하거나 더 나은 성능을 달성했습니다.  
   - DPO(Direct Preference Optimization) 추가 학습 시 **MT-Bench 7.55점**, **AlpacaEval 90.06%**의 우수한 결과를 보였습니다.  

3. **의의**:  
   - 데이터 선택의 과학적 접근법과 효율적인 정렬 도구를 제시하여, **적은 데이터로 고성능 모델을 구축**하는 방법을 개척했습니다.  
   - 선택된 데이터셋과 모델을 공개해 향후 연구에 기여할 것으로 기대됩니다.  

**결론**: 이 연구는 데이터 품질과 선택 전략이 모델 성능에 미치는 영향을 체계화하고, 효율적인 모델 정렬을 위한 실용적인 프레임워크를 제시했습니다.

---

## 1 INTRODUCTION

**1. 서론 (INTRODUCTION) 설명:**

이 섹션은 대규모 언어 모델(LLM)을 인간의 선호도에 맞추는 **정렬(Alignment)** 과정의 중요성과 방법론, 특히 **데이터 효율성**을 높이기 위한 접근법을 다룹니다. 핵심 내용은 다음과 같습니다:

---

### **배경 및 문제 제기**
1. **LLM 정렬의 필요성**  
   - LLM이 인간의 지시를 정확히 이해하고 유용한 응답을 생성하려면 **인간 선호도와의 정렬**이 필수적입니다.  
   - 주요 정렬 기법으로는 **지시 튜닝(Instruction Tuning/SFT)**과 **인간 피드백 강화학습(RLHF)**이 사용됩니다.  
     - **지시 튜닝**: 사전 학습된 모델을 주석이 달린 지시 데이터로 미세 조정합니다.  
     - **RLHF**: 모델의 응답에 대한 인간 피드백을 바탕으로 강화학습을 적용합니다.  
   - 최근 연구는 지시 튜닝만으로도 RLHF 수준의 성능을 달성할 수 있음을 보였습니다.

2. **데이터 효율성의 중요성**  
   - 기존 과제별 미세 조정은 대량의 데이터가 필요했지만, **지시 튜닝은 모델의 사전 학습된 지식을 적은 데이터로 조정**합니다.  
   - 1,000개 수준의 고품질 데이터만으로도 효과적인 정렬이 가능하다는 연구 결과가 있습니다.  
   - **문제점**: 현재 데이터 선택은 경험적 자동화(예: ChatGPT에서 추출)나 수작업에 의존하며, 체계적인 기준이 부족합니다.  

---

### **해결 방안: 데이터 측정 3가지 차원**
연구진은 "좋은 데이터"를 정의하기 위해 다음 세 가지 측면을 제시합니다:  
1. **복잡성(Complexity)**: 데이터의 난이도와 정보 밀도  
2. **품질(Quality)**: 응답의 정확성과 유용성  
3. **다양성(Diversity)**: 주제와 형식의 다양성  

**측정 방법**  
- **EVOL COMPLEXITY/QUALITY**:  
  - 단일 데이터 포인트를 변형해 복잡성/품질이 다른 예시를 생성합니다.  
  - ChatGPT로 변형된 예시를 순위 매기고 점수화한 후, 이를 학습해 **자동 점수 판별기**를 개발합니다.  
- **다양성 측정**: 모델 임베딩 간 거리를 계산해 데이터의 다양성을 평가합니다.  

이를 바탕으로 대량 데이터 풀에서 **효율적인 데이터 샘플을 자동 선별**하는 전략을 수립합니다(그림 1 참조).

---

### **DEITA 모델과 성과**
- **DEITA (Data-Efficient Instruction Tuning for Alignment)**:  
  - LLaMA와 Mistral 모델을 기반으로, **자동 선별된 6,000개 SFT 데이터**로 미세 조정된 모델군입니다.  
  - **주요 성과**:  
    - 기존 모델(Zephyr, Vicuna 등) 대비 **10배 이상 적은 데이터**로 동등/우수한 성능 달성.  
    - DEITA-Mistral-7B: 6K SFT 데이터로 MT-bench 7.22점, AlpacaEval 80.78%.  
    - DPO(Direct Preference Optimization) 추가 시 **MT-bench 7.55점, AlpacaEval 90.06%** 기록.  
  - 데이터셋과 모델을 공개해 향후 연구에 기여합니다.  

---

### **의의**  
- 데이터 선택의 과학적 기준을 제시하고, **적은 데이터로 고성능 모델 정렬**을 가능하게 함.  
- 자동화된 데이터 선별 프레임워크를 통해 LLM 개발의 리소스 효율성을 개선.

---

## 2 WHAT MAKES GOOD DATA FOR ALIGNMENT?

**2. "정렬을 위한 좋은 데이터의 조건은 무엇인가?" 섹션 설명**  
이 섹션에서는 **지시 튜닝(Instruction Tuning)**에 효과적인 데이터의 특성을 체계적으로 분석합니다. 연구진은 "좋은 데이터"의 기준을 규명하기 위해 다음 단계로 접근합니다:

---

### **2.1 데이터 선택 문제의 정의**  
- **핵심 질문**:  
  "적은 양의 데이터로도 대규모 언어 모델(LLM)을 인간의 의도에 효과적으로 정렬하려면, 어떤 데이터를 선택해야 하는가?"  
- **문제 배경**:  
  기존 연구는 데이터 품질과 양의 균형이 모델 성능에 미치는 영향에 대한 명확한 기준이 부족했습니다.  
  - *예시*: 수작업 선별이나 ChatGPT 추출과 같은 경험적 방법은 과학적 근거가 미흡합니다.

---

### **2.2 실험 설계**  
- **목표**:  
  데이터의 **복잡성(Complexity)**, **품질(Quality)**, **다양성(Diversity)**이 모델 정렬에 미치는 영향을 정량적으로 평가합니다.  
- **방법**:  
  1. 다양한 데이터 측정 지표(기존 방법 + 새로운 기법)를 도입합니다.  
  2. 각 지표와 모델 성능 간의 상관관계를 분석합니다.  
  3. 실제 지시 튜닝 실험을 통해 이론적 가설을 검증합니다.

---

### **2.3-2.5 데이터 측정 지표 탐구**  
#### **1. 복잡성(§2.3)**  
- **정의**: 데이터의 **난이도**와 **정보 밀도**  
- **측정 방법**:  
  - **EVOL COMPLEXITY**: 단일 데이터를 변형해 복잡성 수준이 다른 예시 생성 → ChatGPT로 순위 매기기 → 복잡성 점수 판별기 학습.  
  - *예시*: "고양이 설명" (단순) vs. "양자역학의 기본 원리 설명" (복잡).  

#### **2. 품질(§2.4)**  
- **정의**: 응답의 **정확성**과 **유용성**  
- **측정 방법**:  
  - **EVOL QUALITY**: 데이터 품질을 인위적으로 조절한 변형 예시 생성 → ChatGPT로 품질 순위 평가 → 품질 점수 판별기 개발.  
  - *예시*: 명확한 답변 (고품질) vs. 모호하거나 오류 있는 답변 (저품질).  

#### **3. 다양성(§2.5)**  
- **정의**: 데이터의 **주제**와 **형식**적 다양성  
- **측정 방법**:  
  - 모델 임베딩(Embedding) 간 **코사인 유사도** 계산 → 유사도가 낮을수록 다양성이 높음.  
  - *예시*: 다양한 분야(과학, 문학, 역사)와 질문 유형(설명, 요약, 추론)을 포함하는 데이터.  

---

### **의의**  
- **3가지 차원의 측정 지표**를 통해 데이터 선택의 과학적 근거 마련.  
- **EVOL 기법**을 통해 복잡성/품질을 자동으로 평가하는 방법 제시.  
- 후속 실험(§3)에서 이 지표들을 활용해 **고효율 데이터 선택 전략**을 수립함으로써, 적은 데이터로도 우수한 모델 성능을 입증합니다.

---

## 2.1 THE DATA SELECTION PROBLEM

**2.1 데이터 선택 문제 (THE DATA SELECTION PROBLEM) 설명**  
이 섹션은 **지시 튜닝(Instruction Tuning)**을 위한 최적의 데이터를 선별하는 문제를 체계적으로 정의하고, 이를 해결하기 위한 프레임워크를 제시합니다. 핵심 개념은 다음과 같습니다:

---

### **문제 정의**  
- **목표**: 대규모 데이터 풀에서 **제한된 데이터 예산(Data Budget, *m*)**으로 최고의 정렬 성능(*Q*)을 달성하는 데이터 부분집합(*S_π^(m)*)을 선택하는 것.  
- **데이터 풀 구성**:  
  - *X = {x₁, x₂, ..., xₙ}*: 지시-응답 쌍(instruction-response pair)으로 이루어진 대규모 데이터 집합.  
  - 각 *xᵢ*는 하나의 데이터 샘플을 의미합니다.  

---

### **선택 전략(π)과 최적화**  
- **선택 전략(π)**: 데이터 품질을 평가하는 **측정 지표(metric)**를 기반으로 데이터를 선별하는 방법.  
- **최적 전략(π*)**: 주어진 데이터 예산 *m*으로 최대 정렬 성능 *Q*를 달성하는 전략.  
  - 수식:  
    $$  
    \pi^{*} = \arg\max_{\pi} Q(S_{\pi}^{(m)})  
    $$  
  - *해석*: 가능한 모든 전략(π) 중에서 *m*개의 데이터로 학습했을 때 가장 높은 성능(*Q*)을 내는 전략을 선택합니다.  

---

### **핵심 개념**  
1. **데이터 예산(Data Budget, *m*)**:  
   - 사용할 데이터 샘플 수. 모델 학습에 소요되는 계산 자원과 비례합니다.  
   - *예시*: 6,000개의 데이터만으로도 우수한 성능을 내는 것이 목표일 때, *m=6,000*.  

2. **정렬 성능(*Q*)**:  
   - 지시 튜닝 후 모델이 보이는 성능 지표(예: MT-Bench, AlpacaEval 점수).  

3. **측정 지표의 역할**:  
   - 데이터 샘플의 **복잡성, 품질, 다양성** 등을 정량화하여, 어떤 데이터가 *Q*를 높이는지 판단하는 기준으로 사용됩니다.  

---

### **연구의 방향**  
- 이론적 프레임워크를 바탕으로, 다양한 측정 지표와 선택 전략을 실험적으로 탐구합니다.  
  - *예시 실험 과정*:  
    1. 특정 지표(예: 복잡성 점수)로 데이터를 정렬합니다.  
    2. 상위 *m*개 데이터를 선택해 지시 튜닝을 수행합니다.  
    3. 결과 성능(*Q*)을 비교하여 해당 지표의 유효성을 검증합니다.  

---

### **의의**  
- 데이터 선택 문제를 **수학적 최적화 문제**로 명확히 정의함으로써,  
  - 체계적인 데이터 선별 전략 수립의 기반을 마련했습니다.  
  - 후속 섹션(§2.3-2.5)에서 제안된 복잡성, 품질, 다양성 지표가 *Q*와 어떻게 연관되는지 분석할 수 있는 토대를 제공합니다.  
- 이 프레임워크는 **DEITA** 모델 개발 시 데이터 효율성을 극대화하는 데 직접적으로 활용됩니다.

---

## 2.2 EXPERIMENTAL SETUP

**2.2 실험 설계 (EXPERIMENTAL SETUP) 설명**  
이 섹션에서는 데이터 측정 지표의 효과를 검증하기 위한 **체계적인 실험 절차**를 소개합니다. 핵심은 **3단계 프레임워크**를 통해 각 지표의 유효성을 평가하는 것입니다:

---

### **실험 프레임워크**  
1. **데이터 선별**: 특정 측정 지표(예: 복잡성)를 기반으로 데이터 풀에서 *m=6,000개* 샘플을 선택합니다.  
2. **모델 미세 조정**: 선별된 데이터로 LLaMA-1 13B 모델을 지시 튜닝합니다.  
3. **성능 평가**: MT-Bench 점수를 통해 모델의 **지시 수행 능력**을 측정합니다.  

---

### **데이터 풀 구성**  
두 가지 유형의 데이터 풀을 구축해 다양한 실제 시나리오를 모의합니다:  

1. **X_sota (고품질 데이터 풀)**  
   - **목적**: 이미 우수한 데이터 풀에서 **효율성 극대화** 가능성 탐구.  
   - **구성**: WizardLM, UltraChat 등 SOTA 모델의 학습 데이터를 통합 (총 300K 샘플).  
   - **특징**: 복잡성, 다양성, 품질이 모두 높음.  

2. **X_base (저품질 데이터 풀)**  
   - **목적**: 실제 환경에서 흔히 접하는 **저품질/중복 데이터** 대응 전략 검증.  
   - **구성**: Alpaca, Dolly 등 기본 데이터셋 통합 (총 100K 샘플).  
   - **특징**: 응답 길이 짧고, 주제/형식 다양성 낮음.  

---

### **학습 및 평가 설정**  
- **모델**: LLaMA-1 13B (고정된 하이퍼파라미터 사용, 부록 A 참조).  
- **데이터 예산**: *m=6,000* 샘플.  
- **평가 지표**: **MT-Bench** (다중 회차 대화 평가, GPT-4가 응답 점수화).  

---

### **복잡성 측정 지표 비교 결과**  
표 2는 다양한 복잡성 지표로 선별된 데이터로 학습한 모델의 MT-Bench 점수를 보여줍니다.  

#### **주요 결과**  
1. **EVOL COMPLEXITY의 우수성**:  
   - X_sota: **6.27점** (최고 성능).  
   - X_base: **5.57점** (기존 방법 대비 큰 격차).  
   - **강점**: 데이터 품질에 관계없이 **강건한 성능** 발휘.  

2. **기존 방법의 한계**  
   - **Instruction Length (지시 길이)**: 긴 지시문 ≠ 고복잡성 (X_base 4.00점).  
   - **Perplexity (응답 혼란도)**: 낮은 성능 (X_sota 4.06점) → 짧은 응답 샘플 선별 경향성 문제.  
   - **Direct Scoring/Instruction Node**: ChatGPT 주석 비용高 → 50K 샘플 제한 시 성능 하락.  

3. **EVOL COMPLEXITY의 혁신성**  
   - **진화 기반 평가**: 단일 데이터를 점진적 복잡성 변형 → ChatGPT로 세밀한 점수화.  
   - **자동 점수 판별기**: 소량 시드 데이터 학습 → 대규모 데이터에 확장 적용 가능.  

---

### **결론적 시사점**  
- **데이터 복잡성 ≠ 단순 길이/혼란도**: 질적 진화와 세밀한 평가가 필수적.  
- **EVOL 기법의 효용성**: 비용 효율적이며, 다양한 데이터 환경에서 일관된 성능 확보.  
- **실용적 프레임워크**: 제한된 자원으로 고성능 모델 구축 가능성 입증.

---

## 2.4 FROM THE QUALITY PERSPECTIVE – EVOL QUALITY

**2.4 품질 관점 – EVOL QUALITY**  
이 섹션은 지시 튜닝 데이터의 **품질(Quality)** 측정 방법을 탐구하며, 특히 **EVOL QUALITY** 기법을 제안합니다. 핵심 내용은 다음과 같습니다:

---

### **품질 측정의 중요성**  
- **목표**: 정확성, 상세성, 유용성이 높은 응답을 생성하는 모델 개발.  
- **문제 인식**:  
  - 저품질 데이터(모호한 답변, 오류 포함)는 모델 성능을 크게 저하시킵니다.  
  - 기존 방법(예: 응답 길이)은 품질을 정확히 반영하지 못함.  

---

### **품질 측정 방법 비교**  
1. **Random Selection (무작위 선택)**: 기준 없이 샘플링.  
2. **Response Length (응답 길이)**: 긴 응답 = 고품질 가정.  
3. **Direct Scoring (직접 점수화)**: ChatGPT로 응답 정확성 평가 (고비용).  
4. **EVOL QUALITY (제안 방법)**:  
   - **진화 기반 품질 향상**:  
     1. 원본 데이터 $(I_k^{(0)}, R_k^{(0)})$에 대해 ChatGPT가 응답을 **품질 향상 방향**으로 변형 (예: 세부 정보 추가, 창의성 강화).  
     2. 5회 변형을 거쳐 다양한 품질의 응답 집합 $\{R_k^{(0)}, ..., R_k^{(5)}\}$ 생성.  
   - **ChatGPT 순위/점수화**:  
     - 동일 지시문에 대한 변형 응답들을 비교하여 세밀한 품질 점수($q$) 부여.  
   - **자동 점수 판별기 학습**:  
     - LLaMA-1 7B 모델을 미세 조정해 품질 점수 예측.  

---

### **실험 결과 (표 3)**  
- **X_sota (고품질 풀)**:  
  - EVOL QUALITY **6.19점** (Random 5.84점 대비 우수).  
  - 고품질 풀에서는 응답 길이 영향 적음 (5.94점).  
- **X_base (저품질 풀)**:  
  - EVOL QUALITY **5.67점** (Random 4.93점 대비 큰 격차).  
  - 품질 지표가 저품질 풀에서 더 큰 성능 향상 기여.  

**결론**:  
- EVOL QUALITY는 **데이터 품질 편차가 큰 환경**에서 특히 효과적.  
- ChatGPT 주석 의존도 낮춰 비용 효율성 확보.  

---

**2.5 다양성 관점 – 임베딩 기반 접근법**  
이 섹션은 데이터 **다양성(Diversity)** 측정 및 선택 전략을 다룹니다.  

---

### **다양성의 중요성**  
- **목표**: 다양한 주제/형식의 요청 처리 능력 확보.  
- **문제 인식**: 실제 데이터는 중복성 높음 → 다양성 보장 필수.  

---

### **다양성 측정 방법 비교**  
1. **Random Selection**: 다양성 무시.  
2. **Instag Diversity (태그 기반)**:  
   - 데이터에 의미론적 태그 부여 → 태그 집합의 성장으로 다양성 측정.  
3. **Repr Filter (제안 방법, 임베딩 기반)**:  
   - **LLaMA-1 13B 임베딩** 활용:  
     1. 데이터 샘플을 벡터로 변환.  
     2. **코사인 거리** 계산 → 기존 선택 집합($S$)과의 유사도 평가.  
   - **반복적 선택 프로세스**:  
     - 임계값($\tau=0.9$) 미만 시 샘플 추가 → $S$의 다양성 유지.  
   - **복잡성/품질 점수 선정렬**: 우수한 샘플 우선 검토.  

---

### **실험 결과 (표 4)**  
- **X_sota**:  
  - Repr Filter **6.17점** (Instag Diversity 6.10점 대비 소폭 우위).  
- **X_base**:  
  - Repr Filter **4.68점** (Instag Diversity 4.46점 대비 성능 개선).  

**결론**:  
- 임베딩 기반 접근법이 **태그 기반 방법보다 강건함**.  
- 다양성 보장 시 무작위 선택 대비 성능 향상 필수적.  

---

### **종합 시사점**  
- **품질, 복잡성, 다양성**의 3축 데이터 측정이 모델 정렬 성능을 결정.  
- **EVOL 기법**과 **임베딩 필터링**을 결합해 자동화된 고효율 데이터 선택 가능 → DEITA 모델의 성공적 구현.

---

## 3 DEITA– DATA EFFICIENT INSTRUCTION TUNING FOR ALIGNMENT

**3. DEITA – 데이터 효율적 지시 튜닝을 위한 정렬**  
이 섹션에서는 복잡성(Complexity), 품질(Quality), 다양성(Diversity)의 **3가지 차원을 통합**하여 최적의 데이터를 선별하는 방법론 **DEITA**를 제안합니다. DEITA는 적은 양의 데이터로도 고성능 모델을 구축하는 것을 목표로 합니다.  

---

### **DEITA의 데이터 선택 전략**  
1. **3차원 통합 접근법**:  
   - **복잡성**: EVOL COMPLEXITY로 측정된 점수를 기반으로 고난이도 데이터 우선 선정.  
   - **품질**: EVOL QUALITY 점수를 통해 정확성과 유용성이 높은 응답을 가진 데이터 필터링.  
   - **다양성**: 임베딩 기반 **Repr Filter**를 적용해 중복성을 제거하고 주제/형식 다양성 보장.  

2. **선별 프로세스**:  
   - **단계적 필터링**:  
     1. **복잡성 + 품질 점수**로 상위 데이터 추출.  
     2. **다양성 필터링**: 임베딩 유사도 분석을 통해 중복 데이터 제거.  
   - **우선순위 정렬**: 복잡성과 품질이 높은 데이터를 우선 검토한 후, 다양성을 확보하기 위해 반복적 선택 수행.  

---

### **DEITA 모델 학습**  
- **기반 모델**: LLaMA 및 Mistral 모델을 사전 학습된 모델로 사용.  
- **학습 데이터**: 상기 전략으로 선별된 **6,000개의 SFT 데이터**만 활용.  
- **성능**:  
  - **DEITA-Mistral-7B**: 6K SFT 데이터로 **MT-Bench 7.22점**, **AlpacaEval 80.78%** 달성.  
  - **DPO 추가 학습 시**: 6K SFT + 10K DPO 데이터로 **MT-Bench 7.55점**, **AlpacaEval 90.06%** 기록.  
- **효율성**: 기존 모델 대비 **10배 이상 적은 데이터**로 동등/우수한 성능 구현.  

---

### **의의 및 기여**  
1. **과학적 데이터 선별**:  
   - 복잡성, 품질, 다양성의 체계적 측정을 통해 경험적 방법의 한계 극복.  
2. **리소스 효율성**:  
   - 대량 데이터 수집/주석 비용 절감 및 학습 시간 단축.  
3. **오픈소스 공개**:  
   - 선별된 데이터셋과 모델을 공개해 연구 재현성 및 후속 연구 지원.  

---

### **결론**  
DEITA는 **데이터 효율성**과 **모델 성능** 간의 균형을 최적화하는 프레임워크로, LLM 정렬에 필요한 데이터 선택의 과학적 기준을 제시했습니다. 이 접근법은 향후 모델 개발 시 자원 제약 문제를 해결하는 데 기여할 것으로 기대됩니다.

---

## 3.1 METHOD

**3.1 방법론 (METHOD) 설명**  
(참고: 원문에 내용이 누락된 것으로 보이지만, 논문의 흐름과 이전 섹션들을 바탕으로 **추론된 설명**을 제공합니다.)

---

### **DEITA 방법론의 핵심 단계**  
DEITA는 복잡성(Complexity), 품질(Quality), 다양성(Diversity)의 **3가지 차원을 종합**하여 데이터를 선택하고, 이를 바탕으로 고효율 지시 튜닝을 수행합니다. 구체적인 단계는 다음과 같습니다:

---

#### **1. 데이터 측정 및 점수화**  
1. **복잡성 점수 (Complexity Score)**:  
   - **EVOL COMPLEXITY** 기법을 사용해 각 데이터 샘플의 복잡성을 측정합니다.  
   - ChatGPT로 생성된 변형 예시의 순위와 점수를 학습해 자동 복잡성 판별기를 구축합니다.  

2. **품질 점수 (Quality Score)**:  
   - **EVOL QUALITY** 기법을 적용해 응답의 정확성과 유용성을 평가합니다.  
   - 동일한 지시문에 대한 다양한 품질의 응답을 비교하여 세밀한 점수화를 수행합니다.  

3. **다양성 점수 (Diversity Score)**:  
   - **임베딩 기반 Repr Filter**를 활용해 데이터 간 유사도를 계산합니다.  
   - LLaMA 모델의 임베딩 공간에서 코사인 거리를 측정하여 중복성을 제거합니다.  

---

#### **2. 데이터 선별 전략**  
1. **우선순위 정렬**:  
   - 복잡성과 품질 점수를 종합해 상위 데이터를 선별합니다.  
   - 예: 복잡성 점수 × 품질 점수의 가중합으로 순위 결정.  

2. **다양성 필터링**:  
   - 선정된 상위 데이터 집합에 대해 **임베딩 유사도 분석**을 수행합니다.  
   - 임계값(τ=0.9)을 초과하는 중복 샘플을 제거하며, 최종 6K 데이터 선택.  

---

#### **3. 모델 학습 및 평가**  
1. **모델 미세 조정**:  
   - 선별된 데이터로 LLaMA 또는 Mistral 모델을 지시 튜닝합니다.  
   - 고정된 하이퍼파라미터를 사용해 재현성 보장 (부록 A 참조).  

2. **성능 평가**:  
   - **MT-Bench**, **AlpacaEval**, **Open LLM Leaderboard**에서 성능을 측정합니다.  
   - GPT-4를 평가자로 활용해 응답의 유용성과 정확성을 점수화합니다.  

---

### **의의**  
- **통합 프레임워크**: 3가지 차원의 측정 지표를 조합해 데이터 효율성을 극대화합니다.  
- **계층적 선택**: 복잡성/품질 → 다양성 순의 단계적 필터링으로 최적화된 데이터셋 구축.  
- **확장성**: 다른 LLM 및 데이터셋에 적용 가능한 일반적인 방법론을 제시합니다.  

---

이 방법론은 **DEITA 모델**의 우수한 성능(§3 결과 섹션 참조)을 가능하게 한 핵심 기여로, 적은 데이터로도 고품질 정렬을 달성하는 과학적 접근법을 구현합니다.

---

## Algorithm 1 Score-First, Diversity-Aware Data Selection

**알고리즘 1: 점수 우선, 다양성 고려 데이터 선택 (Score-First, Diversity-Aware Data Selection)**

이 알고리즘은 **복잡성(Complexity)**과 **품질(Quality)** 점수를 기반으로 데이터를 우선순위화한 후, **다양성(Diversity)**을 보장하기 위해 중복 샘플을 제거하는 과정을 체계적으로 결합합니다.  
DEITA 모델의 데이터 효율적 학습을 위해 설계되었으며, 주요 단계는 다음과 같습니다:

---

### **알고리즘 단계 설명**  
1. **입력**:  
   - 데이터 풀 \( X \): 지시-응답 쌍으로 구성된 대규모 데이터 집합.  
   - 데이터 예산 \( m \): 선택할 샘플 수 (예: 6,000개).  

2. **출력**:  
   - 선별된 데이터 부분집합 \( S_{\pi_{\mathrm{DEITA}}}^{(m)} \).  

3. **초기화**:  
   - 빈 데이터셋 \( S \) 생성.  

4. **점수 계산 및 정렬**:  
   - 각 샘플의 **복잡성 점수(\( c \))**와 **품질 점수(\( q \))**를 곱해 종합 점수 \( s = c \times q \) 계산.  
   - \( s \)를 기준으로 데이터 풀 \( X \)를 내림차순 정렬 → \( X^* \).  

5. **반복적 선택 및 다양성 필터링**:  
   - 정렬된 \( X^* \)의 샘플을 순차적으로 검토:  
     - **단계 7-8**: 현재 샘플 \( x \)와 \( S \) 내 가장 가까운 샘플 간 **임베딩 거리** 계산.  
       - **LLaMA-1 13B** 모델로 임베딩 추출 → **코사인 거리** 측정.  
     - **단계 9-12**: 거리가 임계값 \( \tau=0.9 \) 미만이면 \( S \)에 추가 (중복으로 판단해 제외).  
     - **단계 14-16**: \( S \)의 크기가 \( m \)에 도달하면 종료.  

---

### **핵심 개념**  
1. **Evol Score (\( s = c \times q \))**:  
   - 복잡성과 품질을 동시에 반영한 종합 점수.  
   - 높은 \( s \)를 가진 샘플은 **난이도 높고 정확한 응답**을 가짐.  
   - **다중 회차 대화**에서는 각 회차별 점수를 합산해 전체 점수 계산.  

2. **Repr Filter (임베딩 기반 필터링)**:  
   - **LLaMA 임베딩**을 사용해 샘플 간 의미적 유사도 측정.  
   - 코사인 거리 \( d(x, S) \)가 임계값 \( \tau=0.9 \)보다 작으면 중복으로 판단 → 제외.  
   - **계층적 선택**: 우수한 샘플을 우선 검토한 후, 다양성 보장을 위해 필터링.  

---

### **DEITA 모델 학습**  
- **학습 데이터**: 알고리즘으로 선별된 \( m \)개 샘플 (예: 6K).  
- **기반 모델**: LLaMA-1-13B, LLaMA-2-13B, Mistral-7B 등.  
- **학습 세부사항**: 고정된 하이퍼파라미터 사용 (부록 A 참조).  

---

### **의의**  
- **효율성**: 복잡성/품질 점수 정렬 + 다양성 필터링으로 **적은 데이터로 고성능 달성**.  
- **과학적 데이터 선택**: 경험적 방법 대신 체계적 지표 기반 접근.  
- **확장성**: 다른 LLM 및 데이터셋에 적용 가능한 일반화된 프레임워크.  

이 알고리즘은 DEITA 모델이 **10배 이상 적은 데이터**로도 SOTA 성능을 내는 데 기여한 핵심 메커니즘입니다.

---

## 3.2 EXPERIMENTAL SETUP

**3.2 실험 설정 (EXPERIMENTAL SETUP) 설명**  
이 섹션에서는 DEITA 모델의 학습 및 평가를 위한 구체적인 실험 환경을 설명합니다. 주요 내용은 다음과 같습니다:

---

### **데이터 및 학습 설정**  
1. **데이터 예산**:  
   - **6K** 및 **10K** 샘플로 DEITA 모델을 각각 학습합니다.  
   - 데이터 풀: 고품질 데이터 집합인 **X_sota**에서 샘플 선별.  

2. **기반 모델**:  
   - **LLaMA-1-13B**, **LLaMA-2-13B**, **Mistral-7B**를 기본 모델로 사용합니다.  

3. **학습 방법**:  
   - **지시 튜닝(SFT)**에 집중: 데이터 선택 전략의 효과를 명확히 분석하기 위해 RLHF 대신 SFT만 적용.  
   - 추가 실험: 최고 성능 SFT 모델에 **DPO(Direct Preference Optimization)**를 적용해 성능 향상 확인.  
     - DPO 데이터: **UltraFeedback** 데이터셋에서 추출한 10K 비교 쌍 사용.  

---

### **평가 벤치마크**  
1. **MT-Bench**:  
   - 다중 회차 대화 평가 (글쓰기, 추론, 코딩 등).  
   - GPT-4가 응답 품질을 점수화.  

2. **AlpacaEval**:  
   - 지시 수행 능력 평가.  
   - 인간 평가자 또는 GPT-4가 응답의 유용성과 정확성을 평가.  

3. **Open LLM Leaderboard**:  
   - 4가지 분류 과제로 구성:  
     - **ARC**: 과학적 추론 능력.  
     - **HellaSwag**: 상식 추론.  
     - **MMLU**: 다학제적 지식 이해.  
     - **TruthfulQA**: 사실 기반 응답 정확성.  

4. **인간 평가**: 부록 D에서 추가 결과 제공.  

---

### **비교 대상 모델**  
1. **데이터 선택 기법 비교**:  
   - **LIMA**, **Alpagasus**, **TAGLM**과 DEITA의 성능 차이 분석.  

2. **오픈소스 SOTA 모델 비교**:  
   - **Vicuna**, **WizardLM**, **Mistral-Instruct**, **Zephyr** 등과 성능 경쟁력 평가.  

---

### **실험 목적**  
1. **데이터 선택 전략 검증**:  
   - 제안된 복잡성-품질-다양성 통합 전략이 기존 방법 대비 우수함을 입증.  
2. **모델 확장성 확인**:  
   - 다양한 기반 모델(LLaMA-1, LLaMA-2, Mistral)에서 DEITA의 일반화 가능성 검토.  
3. **DPO 효과 분석**:  
   - SFT 후 DPO 적용 시 최종 성능 향상 정도 측정.  

---

### **의의**  
- DEITA의 **데이터 효율성**과 **범용성**을 다각도로 입증하기 위한 체계적 실험 설계.  
- 오픈소스 LLM 생태계에서 **적은 데이터로 고성능 모델 구축** 가능성을 제시.

---

## 3.3 RESULTS

**3.3 실험 결과 (RESULTS) 설명**  
이 섹션은 DEITA 모델의 성능을 다양한 벤치마크와 기존 SOTA 모델과 비교한 결과를 제시합니다. 핵심 내용은 다음과 같습니다:

---

### **주요 비교 결과**  
#### **표 5: 데이터 선택 기법 간 성능 비교 (LLaMA-1-13B 기반)**  
- **DEITA-LLaMA1-13B (6K)**:  
  - **MT-Bench 6.46**, **AlpacaEval 77.08%**로 모든 경쟁 방법(Random, Alpagasus, LIMA, TAGLM)을 큰 격차로 능가.  
  - **LIMA (1K 데이터)** 대비 MT-Bench **+2.17점**, AlpacaEval **+35.1%p** 향상.  

#### **표 6: 다양한 기반 모델별 SOTA 모델 비교**  
1. **LLaMA-1-13B 기반**:  
   - **DEITA-10K**: MT-Bench **6.60**, AlpacaEval **78.01%**로 WizardLM(70K 데이터) 및 Vicuna(125K 데이터)보다 우수.  
2. **LLaMA-2-13B 기반**:  
   - **DEITA-10K**: MT-Bench **6.79**, AlpacaEval **81.09%**로 RLHF 적용된 LLaMA2-Chat(>100K SFT + 1M RLHF)와 동등.  
3. **Mistral-7B 기반**:  
   - **DEITA-6K + DPO**: MT-Bench **7.55**, AlpacaEval **90.06%**로 Zephyr(200K SFT + 60K DPO)과 비슷한 성능 달성.  
   - **Mistral-Instruct-v0.2** (비공개 데이터)에 근접한 성능.  

---

### **Open LLM Leaderboard 결과 (표 7)**  
- **DEITA-Mistral-7B (6K + DPO)**:  
  - **평균 69.86점**으로 Zephyr-beta(66.36점)를 능가.  
  - **TruthfulQA 67.14점**, **ARC 66.21점**에서 특히 뛰어남.  
- **DEITA-LLaMA1-13B (10K)**:  
  - LIMA, WizardLM, Vicuna 등 모든 LLaMA-1 기반 모델을 큰 격차로 제치고 최고 평균 점수(**64.27**) 기록.  

---

### **데이터 스케일링 분석 (그림 2)**  
- **DEITA의 효율성**:  
  - **3K 데이터**로 전체 300K 데이터 사용 시와 동등 성능 → **100배 데이터 효율성**.  
  - 데이터 증가 시 초기 성능 상승 후 감소 → "좋은 데이터" 비중 제한적임을 시사.  

---

### **MT-Bench 세부 능력 분석 (그림 3)**  
- **DEITA-Mistral**의 강점:  
  - **코딩**, **수학**, **추론** 과제에서 탁월한 성능 → MT-Bench 고득점 주원인.  
  - AlpacaEval은 기본 지시 수행에 집중되므로 상대적 격차 작음.  

---

### **핵심 시사점**  
1. **데이터 효율성**:  
   - **6K~10K 데이터**로 기존 70K~200K 데이터 기반 모델 대비 우수한 성능.  
2. **DPO의 시너지**:  
   - SFT 후 DPO 적용 시 성능 추가 향상 (MT-Bench **7.22 → 7.55**, AlpacaEval **80.78% → 90.06%**).  
3. **범용성**:  
   - LLaMA-1, LLaMA-2, Mistral 등 다양한 기반 모델에서 일관된 우수성 입증.  
4. **데이터 품질의 중요성**:  
   - 고품질 데이터 선택이 양적 확장보다 성능에 더 직결됨을 실험적으로 검증.  

---

### **결론**  
DEITA는 **데이터 선택의 과학적 접근법**을 통해 적은 자원으로도 SOTA 성능을 달성함으로써, LLM 정렬 분야에 새로운 패러다임을 제시했습니다. 특히 오픈소스 모델 생태계에서 리소스 효율적 고성능 모델 개발의 가능성을 입증했습니다.

---
