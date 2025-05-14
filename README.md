# AI Startup Investment Evaluation Agent
본 프로젝트는 AI 스타트업의 기술력, 시장성, 리스크 등을 다각적으로 평가하여 투자 가능성을 자동으로 분석하는 Agentic RAG 기반 실습 프로젝트입니다.

## Overview

- Objective: AI 스타트업에 대한 투자 관점에서 핵심 기술, 시장성, 리스크를 분석하여 투자 보고서 자동 생성
- Method: Agent Orchestration + Retrieval Augmented Generation (Agentic RAG)
- Tools: Pinecone, LangChain, OpenAI API

## Features

- PDF, 뉴스, STT 기반 정보 임베딩 및 검색 (Pinecone 기반)
- 평가 기준별 에이전트 (기업 요약, 시장성 평가, 리스크 분석, 투자 의사결정, 보고서 생성)
- 최종 투자 보고서 자동 생성 (투자 권고: 유망 / 보류 / 회피 등)

## Tech Stack 

| Category   | Details                                         |
|------------|-------------------------------------------------|
| Framework  | LangChain, Python                               |
| LLM        | GPT-3.5-turbo via OpenAI API                    |
| Retrieval  | Pinecone Vector Store                           |
| Architecture | Agent Orchestration (Step-by-Step Pipeline)  |

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

## Contributors 

- 김다빈 : decision_agent 설계, Prompt Engineering, PDF Parsing
- 김민수 : risk_agent 설계, Prompt Engineering, Pinecone Integration
- 장수희 : market_agent 설계, Prompt Engineering, report_agent 설계, 문서화
- 하동헌 : summary_agent 설계, Prompt Engineering, STT 데이터 처리