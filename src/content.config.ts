import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const notes = defineCollection({
  loader: glob({ base: "./notes", pattern: "**/*.md" }),
  schema: z.object({
    title: z.string().optional(),
    date: z.coerce.date(),
    updated: z.coerce.date().optional(),
    summary: z.string().optional(),
    tags: z.array(z.string()).optional(),
    draft: z.boolean().optional()
  })
});

export const collections = { notes };
