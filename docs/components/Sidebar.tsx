import Link from 'next/link';

// components/Sidebar.tsx
const menu = [
  { title: '📚 문서 개요', href: '/docs' },
  {
    title: '🧮 계산 그래프',
    href: '/docs/graph',
  },
  {
    title: '🔧 레이어',
    href: '/docs/layers',
  },
  {
    title: '🚀 백엔드',
    href: '/docs/backend',
  },
];


export default function Sidebar() {
  return (
    <aside className="w-64 bg-gray-100 p-6 border-r h-full">
      <h2 className="font-bold text-xl mb-6">📘 문서 탐색</h2>
      <nav className="space-y-2">
        {menu.map(({ title, href }) => (
          <Link key={href} href={href} className="block text-gray-700 hover:font-semibold">
            {title}
          </Link>
        ))}
      </nav>
    </aside>
  );
}