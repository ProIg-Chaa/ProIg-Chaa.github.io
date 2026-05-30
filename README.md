# ProIg-Chaa.github.io

Chaa 的 AI 学习笔记主页，使用 Astro 构建并部署到 GitHub Pages。

## 本地开发

```bash
npm install
npm run dev
```

## 发布笔记

把 Markdown 放到 `notes/` 下的分类目录中，例如：

- `notes/lecture/`
- `notes/transformerArch/`

站点会在构建时自动扫描这些文件，提取一级标题作为文章标题，并生成安全的文章 URL。
Astro 会把分类 key 归一为小写，界面会继续显示成中文分类名。

## 部署

仓库名使用 `ProIg-Chaa.github.io`。推送到 `main` 后，GitHub Actions 会自动构建并部署到 `https://proig-chaa.github.io/`。
