//전체 구조 설명

// pages/posts/architecture.tsx
import React from 'react'

export default function Architecture() {
  return (
    <div style={{ padding: '2rem' }}>
      <h1>AI Framework 구조 개요</h1>
      <p>프레임워크는 계산 그래프 → 컴파일 → CUDA 실행 흐름으로 구성됩니다.</p>

      <ul>
        <li>계산 노드 기반 자동 미분</li>
        <li>GraphCompiler를 통한 행렬 변환</li>
        <li>Pybind11, CUDA 실행 구조</li>
      </ul>
    </div>
  );
}
