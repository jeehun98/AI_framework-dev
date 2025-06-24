// pages/index.tsx

import Link from 'next/link';

export default function Home() {
  return (
    <main style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
      <h1>🧠 AI Framework from Scratch</h1>
      <p>
        이 프로젝트는 딥러닝 계산 그래프와 CUDA 연산을 직접 구현한
        경량 프레임워크입니다.
        <br />
        Python 모델을 행렬 기반 구조로 컴파일하고, CUDA 커널로 실행합니다.
      </p>

      <h2 style={{ marginTop: '2rem' }}>📚 문서 바로가기</h2>
      <ul>
        <li><Link href="/posts/architecture">📐 구조 개요 및 설계</Link></li>
        <li><Link href="/posts/layers/dense">🔧 Dense 레이어 구현</Link></li>
        <li><Link href="/posts/graph/node">🧱 계산 그래프(Node)</Link></li>
        <li><Link href="/posts/cuda/matmul">⚡ CUDA MatMul 커널</Link></li>
      </ul>

      <h2 style={{ marginTop: '2rem' }}>🔗 외부 링크</h2>
      <ul>
        <li><a href="https://github.com/사용자명/AI-Framework" target="_blank">🔗 GitHub 저장소</a></li>
        <li><a href="https://notion.site/작성한문서" target="_blank">📝 Notion 문서</a></li>
      </ul>
    </main>
  );
}
