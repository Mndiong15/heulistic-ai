/**
 * Whether a file name matches an `accept` list of extensions (".jsonl,.yml").
 *
 * The `accept` attribute only filters the file picker; drag-and-drop ignores
 * it, so dropped files are checked with this instead.
 */
export function matchesAccept(fileName: string, accept: string): boolean {
  const extensions = accept
    .split(',')
    .map((e) => e.trim().toLowerCase())
    .filter(Boolean)
  if (extensions.length === 0) return true
  const name = fileName.toLowerCase()
  return extensions.some((ext) => name.endsWith(ext) && name.length > ext.length)
}
