export default function Footer() {
  return (
    <footer className="bg-inverse text-white py-10 mt-auto">
      <div className="container flex flex-col sm:flex-row sm:justify-between items-center gap-2">
        <p className="text-ink-5 text-sm">
          © {new Date().getFullYear()} Heulistic LLC. Turning ML Potential into Production Reality.
        </p>
        <a
          href="mailto:business@heulistic.com"
          className="text-ink-7 text-sm no-underline hover:text-white transition-colors"
        >
          business@heulistic.com
        </a>
      </div>
    </footer>
  )
}
