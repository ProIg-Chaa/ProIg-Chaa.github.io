import { copyFile, mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const notesRoot = path.join(root, "notes");
const distNotesRoot = path.join(root, "dist", "notes");

function hashText(value) {
  let hash = 5381;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 33) ^ value.charCodeAt(index);
  }
  return (hash >>> 0).toString(36);
}

function slugify(value, fallback) {
  const slug = value
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/-{2,}/g, "-");

  return slug || `note-${hashText(fallback)}`;
}

async function findMarkdownFiles(dir) {
  const files = [];
  const entries = await import("node:fs/promises").then(({ readdir }) =>
    readdir(dir, { withFileTypes: true })
  );

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...await findMarkdownFiles(fullPath));
    } else if (/\.mdx?$/i.test(entry.name)) {
      files.push(fullPath);
    }
  }

  return files;
}

function noteSlug(file) {
  const relative = path.relative(notesRoot, file).split(path.sep);
  const category = relative.length > 1 ? relative[0] : "notes";
  const basename = path.basename(relative.at(-1), path.extname(relative.at(-1)));
  const normalizedCategory = slugify(category, category);
  return `${slugify(normalizedCategory, normalizedCategory)}-${slugify(basename, relative.join("/"))}`;
}

function cleanAssetRef(src) {
  const trimmed = src.trim().replace(/^<|>$/g, "").replace(/^["']|["']$/g, "");
  const withoutHash = trimmed.split("#")[0].split("?")[0];
  return withoutHash.replace(/\\/g, "/");
}

function isLocalAsset(src) {
  return src && !/^(?:[a-z]+:|\/|#)/i.test(src);
}

function imageRefs(markdown) {
  const refs = new Set();
  const htmlImage = /<img\b[^>]*\bsrc=["']([^"']+)["'][^>]*>/gi;
  const markdownImage = /!\[[^\]]*]\(([^)\s]+)(?:\s+["'][^"']*["'])?\)/g;

  for (const match of markdown.matchAll(htmlImage)) {
    refs.add(cleanAssetRef(match[1]));
  }

  for (const match of markdown.matchAll(markdownImage)) {
    refs.add(cleanAssetRef(match[1]));
  }

  return Array.from(refs).filter(isLocalAsset);
}

let copied = 0;
let missing = 0;

for (const file of await findMarkdownFiles(notesRoot)) {
  const markdown = await readFile(file, "utf8");
  const slug = noteSlug(file);

  for (const ref of imageRefs(markdown)) {
    const source = path.resolve(path.dirname(file), ref);
    const relativeTarget = ref.split("/").filter(Boolean).join(path.sep);
    const target = path.join(distNotesRoot, slug, relativeTarget);

    if (!source.startsWith(notesRoot)) {
      console.warn(`[note-assets] skipped outside notes/: ${ref}`);
      continue;
    }

    try {
      await mkdir(path.dirname(target), { recursive: true });
      await copyFile(source, target);
      copied += 1;
    } catch {
      console.warn(`[note-assets] missing asset for ${path.relative(root, file)}: ${ref}`);
      missing += 1;
    }
  }
}

console.log(`[note-assets] copied ${copied} asset(s)`);

if (missing > 0) {
  process.exitCode = 1;
}
