// pages/docs/index.tsx
import Link from 'next/link';

export default function DocsIndex() {
  return (
    <>
      <h1 className="text-2xl font-bold mb-6">📚 문서 카테고리</h1>
      <ul className="list-disc pl-6 space-y-2 text-blue-700">
        <li><Link href="/docs/graph">계산 그래프 구조</Link></li>
        <li><Link href="/docs/layers">레이어 구조 및 작동 방식</Link></li>
        <li><Link href="/docs/backend">CUDA 및 백엔드 구조</Link></li>
      </ul>
    </>
  );
}
