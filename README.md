# Keyword-based Music Recommendation System
> **Word2Vec과 하이브리드 필터링을 활용한 사용자 맞춤형 음악 추천 엔진**

본 프로젝트는 사용자가 입력한 키워드와 곡의 가사/장르 데이터를 분석하여 최적의 음악을 추천하는 시스템입니다. 
**Java Spring Boot**의 안정적인 서비스 운영 능력과 **Python Flask**의 데이터 분석 역량을 결합한 Polyglot 아키텍처로 설계되었습니다.

---

## Tech Stack

| 분류 | 기술 스택 |
| :--- | :--- |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Backend** | Java 17, Spring Boot, Spring Data MongoDB, Gradle |
| **Rec-Engine** | Python, Flask, Gensim (Word2Vec), Scikit-learn, Pandas, NumPy |
| **Database** | MongoDB (NoSQL) |

---

## System Architecture
시스템은 확장성과 언어별 강점을 극대화하기 위해 **3-Tier 구조**로 설계

* **Frontend**: 사용자 인터페이스 제공 및 피드백(평점) 수집
* **Main Backend (Java)**: 서비스 흐름 제어, 사용자 관리 및 MongoDB 연동 (CRUD)
* **Recommendation Engine (Python)**: 대규모 벡터 연산 및 머신러닝 모델 구동 (RESTful API 통신)

---

## Key Features & Algorithms

### 1. Hybrid Recommendation Logic
단순 추천을 넘어 정교한 개인화를 위해 세 가지 필터링 기법을 결합

#### 1-1 콘텐츠 기반 필터링 (CBF)
* **가사 임베딩**: Word2Vec을 활용해 가사를 300차원 벡터로 변환
* **메타데이터**: 장르 정보를 One-hot Encoding 후 가사 벡터와 결합
* **유사도**: Cosine Similarity 기반 상위 5개 곡 추출

#### 1-2 협업 필터링 (CF)
* 사용자-아이템 선호도 행렬을 통한 유사 사용자 취향 반영

#### 1-3 하이브리드 기반
* Cold Start 해결: 활동 데이터가 없는 신규 유저에게는 CBF 결과를 우선 제공
* 가중치 결합: 데이터 축적에 따라 CF 비중을 높여 정교한 개인화 추천으로 전환

### 2. 키워드 스코어링
* 사용자가 입력한 키워드(예: '비', '우울')와 곡 가사/장르 메타데이터 간의 연관성을 계산하여 최적화된 리스트 제공

---

##  Data Strategy & Performance

### Data Pipeline
실제 서비스 효용성을 높이기 위해 세 단계에 걸쳐 데이터셋을 고도화했습니다.
1. Spotify API: 초기 프로토타입 설계 (데이터 활용 정책 이슈로 전환)
2. Last.fm API: Word2Vec 모델 사전 학습 및 대량의 가사/태그 데이터 확보
3. Melon Data (Final): 한국 사용자 정서에 맞춘 국내 음원 데이터를 확보하여 최종 DB 구축

### Evaluation
추천 성능을 정량적으로 검증하기 위해 실제 사용자 피드백을 기반으로 평가를 진행했습니다.
* **Metric**: 'Precision@5', '10 Scale Score'

---
