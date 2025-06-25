//전체 구조 설명

// pages/posts/architecture.tsx
import React from 'react';

export default function Architecture() {
  return (
    <main style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
      <h1>📐 AI Framework 구조 개요</h1>
      <p>
        본 프레임워크는 Python으로 선언된 모델을 기반으로 계산 그래프를 구성하고,
        이를 행렬 기반 표현으로 압축하여 CUDA에서 실행하는 구조입니다.
      </p>

      <h2>🧩 전체 구조 흐름</h2>
      <pre>
        [Python 모델 선언]
            ↓
        [forward_matrix() 호출 → 연산 정보 추출]
            ↓
        [GraphCompiler → E, W, b, shapes 저장]
            ↓
        [run_graph_cuda() → CUDA에서 그래프 해석 및 실행]
      </pre>

      <h2>🔧 핵심 컴포넌트</h2>
      <ul>
        <li><b>Node:</b> 연산 단위로서 계산 흐름을 구성</li>
        <li><b>GraphCompiler:</b> 연산을 E 행렬 등으로 압축</li>
        <li><b>CUDA 실행기:</b> GPU에서 단일 커널로 전체 실행</li>
      </ul>

      <h2>🎯 설계 목표</h2>
      <ul>
        <li>모델 구조를 완전하게 시각화하고 디버깅 가능</li>
        <li>단일 커널로 실행하여 속도 최적화</li>
        <li>실제 프레임워크 내부 구조를 체득하는 학습 목적</li>
      </ul>
    </main>
  );
}
