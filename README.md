# AI Startup Investment Evaluation Agent
본 프로젝트는 AI 스타트업의 기술력, 시장성, 리스크 등을 다각적으로 평가하여 투자 가능성을 자동으로 분석하는 Agentic RAG 기반 실습 프로젝트입니다.

## Overview

- Objective: AI 스타트업에 대한 투자 관점에서 핵심 기술, 시장성, 리스크를 분석하여 투자 보고서 자동 생성
- Method: Agent Orchestration + Retrieval Augmented Generation (Agentic RAG)
- Tools: LangChain, LangGraph, Pinecone, OpenAI API, WeasyPrint

## Features

- PDF, 뉴스, STT 기반 정보 임베딩 및 검색 (Pinecone 기반)
- 평가 기준별 에이전트 (기업 요약, 시장성 평가, 리스크 분석, 투자 의사결정, 보고서 생성)
- 최종 투자 보고서 자동 생성 (투자 권고: 유망 / 보류 / 회피 등)


## Tech Stack 

| Category       | Details                                                |
|----------------|--------------------------------------------------------|
| Framework      | LangChain, LangGraph, Python                           |
| LLM            | GPT-3.5 / GPT-4o via OpenAI API                        |
| Embeddings     | OpenAIEmbeddings                                       |
| Vector Store   | Pinecone Vector DB                                     |
| Retrieval      | Pinecone Retriever (k=5 Top-K Search)                  |
| Orchestration  | LangGraph (Step-by-Step Agent Flow)                   |
| PDF Generator  | WeasyPrint (HTML → PDF via markdown2)                 |


## Agents
 
| Agent             | Description                          |
|--------------------|-------------------------------------|
| summary_agent      | 기업 핵심 요약 (기술, 제품, 개요)  |
| market_agent       | 시장성 및 성장 가능성 평가         |
| risk_agent         | 투자 리스크 분석                  |
| decision_agent     | 투자 적합성 종합 평가              |
| report_agent       | 최종 투자 보고서 생성              |

## Architecture

```
graph TD
    A[기업 문서 및 뉴스] -->|임베딩| V[Pinecone]
    V --> AG1[1. summary_agent (기업 요약)]
    AG1 --> AG2[2. market_agent (시장성 평가)]
    AG1 --> AG3[3. risk_agent (리스크 분석)]
    AG2 --> AG4[4. decision_agent (투자 평가)]
    AG3 --> AG4
    AG4 --> AG5[5. report_agent (투자 보고서)]
    AG5 --> R[최종 투자 보고서]
```

## Directory Structure

```
├── data/                  # 스타트업 PDF 문서, 뉴스, 영상 STT 등의 pdf 파일
├── agents/                # Agent 모듈 (summary, market, risk, decision, report)
├── prompts/               # 프롬프트 템플릿
├── outputs/               # 최종 투자 보고서 저장
├── app.py                 # 전체 파이프라인 실행 스크립트
└── README.md
```

## Sample Output



```

1. 기업 요약
 → 스타트업의 제품/서비스, 핵심 기술, 비즈니스 모델, 시장 타겟 등 주요 정보를 간결하게 요약

2. 시장성 평가
 → 시장 규모, 성장률(CAGR), 경쟁 강도, 진입 장벽, 산업 트렌드 적합성 등 다섯 가지 항목별로 정리하여 분석

3. 리스크 분석
 → 기술, 시장, 조직, 기타 리스크에 대해
 • 설명 + 위험도(낮음/중간/높음)로 구성된 평가

4. 최종 판단
 → 체크리스트 기반 정성 판단 + 가중치 기반 정량 평가(100점 만점)를 포함
 → 최종적으로 ‘투자 적합 / 보류 / 부적합’ 판단과 함께 명확한 이유 제시

```

## Contributors 

- 김다빈 : decision_agent 설계, Prompt Engineering, PDF Parsing
- 김민수 : risk_agent 설계, Prompt Engineering, Pinecone Integration
- 장수희 : market_agent 설계, Prompt Engineering, report_agent 설계, 문서화
- 하동헌 : summary_agent 설계, Prompt Engineering, STT 데이터 처리

## User Interface Overview
<img src="https://github.com/user-attachments/assets/d4d13821-a84c-4b98-a130-62e88be772a5"/>
