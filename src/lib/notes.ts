import { getCollection } from "astro:content";

export const categoryLabels = {
  lecture: "CS336 课程解读",
  transformerarch: "Transformer 架构"
};

export type CategoryKey = keyof typeof categoryLabels;

export type SiteNote = {
  entry: any;
  id: string;
  title: string;
  slug: string;
  category: string;
  categoryLabel: string;
  summary: string;
  updated: Date;
  url: string;
};

export function getCategoryEntries() {
  return Object.entries(categoryLabels).map(([key, label]) => ({
    key,
    label,
    url: `/notes/category/${key}/`
  }));
}

function hashText(value: string) {
  let hash = 5381;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 33) ^ value.charCodeAt(i);
  }
  return (hash >>> 0).toString(36);
}

function slugify(value: string, fallback: string) {
  const slug = value
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/-{2,}/g, "-");

  return slug || `note-${hashText(fallback)}`;
}

function stripExtension(value: string) {
  return value.replace(/\.(md|mdx)$/i, "");
}

function firstHeading(body: string) {
  const match = body.match(/^#\s+(.+)$/m);
  return match?.[1]?.trim();
}

function plainSummary(body: string) {
  const cleaned = body
    .replace(/^#.+$/gm, "")
    .replace(/\$\$[\s\S]*?\$\$/g, "")
    .replace(/```[\s\S]*?```/g, "")
    .replace(/!\[[^\]]*\]\([^)]+\)/g, "")
    .replace(/\[[^\]]+\]\([^)]+\)/g, "")
    .replace(/<?https?:\/\/\S+>?/g, "")
    .replace(/[*_`>#-]/g, "")
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean)
    .find((part) => part.length > 24);

  return cleaned ? `${cleaned.slice(0, 120)}${cleaned.length > 120 ? "..." : ""}` : "一篇关于 AI systems 与 Transformer 的学习笔记。";
}

export async function getAllNotes(): Promise<SiteNote[]> {
  const entries = await getCollection("notes", ({ data }) => data.draft !== true);

  return entries
    .map((entry) => {
      const id = entry.id;
      const parts = id.split("/");
      const category = parts.length > 1 ? parts[0] : "notes";
      const basename = stripExtension(parts.at(-1) ?? id);
      const title = entry.data.title ?? firstHeading(entry.body ?? "") ?? basename;
      const slug = `${slugify(category, category)}-${slugify(basename, id)}`;
      const updated = entry.data.updated ?? entry.data.date;

      return {
        entry,
        id,
        title,
        slug,
        category,
        categoryLabel: categoryLabels[category] ?? category,
        summary: entry.data.summary ?? plainSummary(entry.body ?? ""),
        updated,
        url: `/notes/${slug}/`
      };
    })
    .sort((a, b) => b.updated.getTime() - a.updated.getTime());
}

export async function getNotesByCategory(category: string) {
  const notes = await getAllNotes();
  return notes.filter((note) => note.category === category);
}

export async function getNoteBySlug(slug: string) {
  const notes = await getAllNotes();
  return notes.find((note) => note.slug === slug);
}

export function formatDate(date: Date) {
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit"
  }).format(date);
}
