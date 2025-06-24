//글 목록 페이지

// pages/posts/index.tsx
import Link from 'next/link'

export default function PostsIndex() {
  return (
    <div style={{ padding: '2rem' }}>
      <h1>📚 문서 목록</h1>
      <ul>
        <li><Link href="/posts/architecture">Framework 구조 개요</Link></li>
        <li><Link href="/posts/layers/dense">Dense Layer</Link></li>
        <li><Link href="/posts/graph/node">Node 구조</Link></li>
      </ul>
    </div>
  );
}
