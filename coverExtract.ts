import fs from "fs";
import path from "path";
import os from "os";
import { spawn } from "child_process";
import { PDFDocument } from "pdf-lib";
import { ffmpegBin, ffprobeBin } from "./ffmpegBin";

export type CoverResult = {
  jpegPath: string;
  width: number;
  height: number;
  source: "pdf-page" | "epub-cover" | "epub-image" | "epub-chapter";
};

export type EpubImageEntry = {
  href: string;
  mediaType: string;
  isCover: boolean;
  localPath: string;
};

function tmpDir(): string {
  const dir = path.join(os.tmpdir(), "aura-cover");
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

function uniqueTmp(ext: string): string {
  return path.join(
    tmpDir(),
    `cover-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}.${ext}`
  );
}

async function runFfmpeg(args: string[]): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const child = spawn(ffmpegBin(), ["-y", ...args], { stdio: ["ignore", "pipe", "pipe"] });
    let stderr = "";
    child.stderr?.on("data", (d) => {
      stderr += String(d);
    });
    child.on("error", (err) => reject(err));
    child.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`ffmpeg falhou (código ${code}): ${stderr.slice(-800)}`));
    });
  });
}

/** Convert any image ffmpeg can read into a JPEG. */
export async function convertImageToJpeg(
  inputPath: string,
  outputPath?: string,
  quality = 2
): Promise<string> {
  const out = outputPath || uniqueTmp("jpg");
  await runFfmpeg(["-i", inputPath, "-q:v", String(quality), out]);
  return out;
}

async function renderPdfPageToJpeg(
  pdfBytes: Buffer,
  pageNumber: number
): Promise<{ jpegPath: string; width: number; height: number }> {
  const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
  const { createCanvas } = await import("@napi-rs/canvas");

  const loadingTask = pdfjs.getDocument({
    data: new Uint8Array(pdfBytes),
    useSystemFonts: true,
  });
  const doc = await loadingTask.promise;
  const total = doc.numPages;
  if (pageNumber < 1 || pageNumber > total) {
    throw new Error(`Página da capa ${pageNumber} está fora do intervalo (1–${total}).`);
  }

  const page = await doc.getPage(pageNumber);
  const viewport = page.getViewport({ scale: 2 });
  const canvas = createCanvas(Math.ceil(viewport.width), Math.ceil(viewport.height));
  const context = canvas.getContext("2d");

  await page.render({
    canvasContext: context as unknown as CanvasRenderingContext2D,
    viewport,
    canvas: canvas as unknown as HTMLCanvasElement,
  }).promise;

  const jpegPath = uniqueTmp("jpg");
  const jpegBuffer = canvas.toBuffer("image/jpeg", 85);
  await fs.promises.writeFile(jpegPath, jpegBuffer);

  return {
    jpegPath,
    width: Math.ceil(viewport.width),
    height: Math.ceil(viewport.height),
  };
}

function resolveZipPath(baseDir: string, href: string): string {
  const cleaned = href.replace(/^\//, "").split("#")[0];
  return path.normalize(path.join(baseDir, cleaned));
}

type OpfParsed = {
  opfDir: string;
  coverHref: string | null;
  images: { href: string; mediaType: string; isCover: boolean }[];
};

function parseOpf(opfXml: string, opfDir: string): OpfParsed {
  const images: { href: string; mediaType: string; isCover: boolean }[] = [];
  let coverId: string | null = null;
  let coverHref: string | null = null;

  const metaCover = opfXml.match(
    /<meta[^>]*name=["']cover["'][^>]*content=["']([^"']+)["'][^>]*\/?>/i
  );
  if (metaCover) coverId = metaCover[1];

  const itemRe =
    /<item\b([^>]*)\/?>/gi;
  let m: RegExpExecArray | null;
  while ((m = itemRe.exec(opfXml))) {
    const attrs = m[1];
    const id = attrs.match(/\bid=["']([^"']+)["']/i)?.[1] || "";
    const href = attrs.match(/\bhref=["']([^"']+)["']/i)?.[1] || "";
    const mediaType = attrs.match(/\bmedia-type=["']([^"']+)["']/i)?.[1] || "";
    const properties = attrs.match(/\bproperties=["']([^"']+)["']/i)?.[1] || "";
    if (!href || !mediaType.startsWith("image/")) continue;
    const props = properties.split(/\s+/);
    const isCover =
      props.includes("cover-image") ||
      props.includes("cover-image") ||
      props.includes("cover-image") ||
      (coverId != null &&
        (id === coverId ||
          href === coverId ||
          path.basename(href) === coverId ||
          path.basename(href) === `${coverId}.jpg` ||
          path.basename(href) === `${coverId}.jpeg` ||
          path.basename(href) === `${coverId}.png`));
    if (isCover) coverHref = href;
    images.push({ href, mediaType, isCover });
  }

  if (!coverHref && images.length > 0) {
    const named = images.find(
      (i) => /cover|capa|front/i.test(i.href) || /cover|capa|front/i.test(path.basename(i.href))
    );
    if (named) {
      named.isCover = true;
      coverHref = named.href;
    }
  }

  return { opfDir, coverHref, images };
}

async function extractEpubZip(epubPath: string, destDir: string): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const child = spawn("unzip", ["-o", "-q", epubPath, "-d", destDir], {
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stderr = "";
    child.stderr?.on("data", (d) => {
      stderr += String(d);
    });
    child.on("error", (err) => reject(err));
    child.on("close", (code) => {
      // unzip returns 1 for warnings (e.g. backslash paths) — still ok if files exist
      if (code === 0 || code === 1) resolve();
      else reject(new Error(`Falha ao abrir EPUB (unzip ${code}): ${stderr.slice(-400)}`));
    });
  });
}

function findOpfPath(extractRoot: string): string {
  const containerPath = path.join(extractRoot, "META-INF", "container.xml");
  if (fs.existsSync(containerPath)) {
    const xml = fs.readFileSync(containerPath, "utf8");
    const fullPath = xml.match(/full-path=["']([^"']+)["']/i)?.[1];
    if (fullPath) return path.join(extractRoot, fullPath);
  }
  // Fallback: first .opf
  const walk = (dir: string): string | null => {
    for (const name of fs.readdirSync(dir)) {
      const p = path.join(dir, name);
      const st = fs.statSync(p);
      if (st.isDirectory()) {
        const found = walk(p);
        if (found) return found;
      } else if (name.toLowerCase().endsWith(".opf")) {
        return p;
      }
    }
    return null;
  };
  const found = walk(extractRoot);
  if (!found) throw new Error("Não foi possível localizar o arquivo OPF do EPUB.");
  return found;
}

export async function extractEpubImages(
  epubBytes: Buffer
): Promise<{ extractDir: string; coverJpegPath: string | null; artworks: string[] }> {
  const extractDir = path.join(tmpDir(), `epub-${Date.now().toString(36)}`);
  fs.mkdirSync(extractDir, { recursive: true });
  const epubPath = path.join(extractDir, "book.epub");
  await fs.promises.writeFile(epubPath, epubBytes);
  await extractEpubZip(epubPath, extractDir);

  const opfPath = findOpfPath(extractDir);
  const opfDir = path.dirname(opfPath);
  const opfXml = await fs.promises.readFile(opfPath, "utf8");
  const parsed = parseOpf(opfXml, opfDir);

  const artworks: string[] = [];
  const candidates: CoverCandidate[] = [];

  const coverRank = (img: { href: string; isCover: boolean }): number => {
    if (img.isCover || img.href === parsed.coverHref) return 0;
    if (/cover|capa|front|jacket/i.test(path.basename(img.href))) return 1;
    return 2;
  };

  const ordered = parsed.images.slice().sort((a, b) => coverRank(a) - coverRank(b));

  const tryConvert = async (img: (typeof parsed.images)[number]): Promise<void> => {
    const src = resolveZipPath(opfDir, img.href);
    if (!fs.existsSync(src)) return;
    try {
      const jpeg = await convertImageToJpeg(src);
      const dims = await probeImageSize(jpeg);
      if (dims && (dims.width < 64 || dims.height < 64)) {
        await fs.promises.unlink(jpeg).catch(() => undefined);
        return;
      }
      artworks.push(jpeg);
      candidates.push({
        jpeg,
        href: img.href,
        isCover: Boolean(img.isCover || img.href === parsed.coverHref),
        width: dims?.width || 0,
        height: dims?.height || 0,
      });
    } catch (err) {
      console.warn(`[Cover] Skip EPUB image ${img.href}:`, (err as Error)?.message || err);
    }
  };

  const hasUsableCover = (): boolean => {
    const pick = pickBestCoverJpeg(candidates);
    if (!pick) return false;
    const c = candidates.find((x) => x.jpeg === pick);
    return Boolean(c && coverIsLargeEnough(c));
  };

  const rank01 = ordered.filter((img) => coverRank(img) <= 1);
  const rank2 = ordered.filter((img) => coverRank(img) >= 2);
  rank2.sort((a, b) => {
    const sizeOf = (href: string) => {
      const src = resolveZipPath(opfDir, href);
      try {
        return fs.existsSync(src) ? fs.statSync(src).size : 0;
      } catch {
        return 0;
      }
    };
    return sizeOf(b.href) - sizeOf(a.href);
  });

  for (const img of [...rank01, ...rank2]) {
    if (hasUsableCover()) break;
    await tryConvert(img);
  }

  const coverJpegPath = pickBestCoverJpeg(candidates);

  return {
    extractDir,
    coverJpegPath,
    artworks: coverJpegPath ? [coverJpegPath] : artworks.slice(0, 1),
  };
}

type CoverCandidate = {
  jpeg: string;
  href: string;
  isCover: boolean;
  width: number;
  height: number;
};

function coverIsLargeEnough(c: CoverCandidate): boolean {
  return Math.min(c.width, c.height) >= 200 || c.width * c.height >= 80_000;
}

/**
 * Books / iTunes show the first attached_pic. OPF order is often a logo or
 * publisher mark, not the jacket. Prefer the declared cover when it is large
 * enough; otherwise the largest portrait/square JPEG.
 */
function pickBestCoverJpeg(candidates: CoverCandidate[]): string | null {
  if (candidates.length === 0) return null;

  const declared = candidates.find((c) => c.isCover && coverIsLargeEnough(c));
  if (declared) return declared.jpeg;

  const named = candidates.find(
    (c) =>
      coverIsLargeEnough(c) && /cover|capa|front|jacket/i.test(path.basename(c.href))
  );
  if (named) return named.jpeg;

  const bookLike = candidates.filter((c) => {
    if (!coverIsLargeEnough(c) || c.width <= 0 || c.height <= 0) return false;
    const ratio = c.width / c.height;
    // Jackets are portrait or near-square; drop wide banners / chapter strips.
    return ratio <= 1.35 && ratio >= 0.4;
  });
  const pool = bookLike.length > 0 ? bookLike : candidates.filter(coverIsLargeEnough);
  const ranked = (pool.length > 0 ? pool : candidates).slice().sort((a, b) => {
    const areaA = a.width * a.height;
    const areaB = b.width * b.height;
    if (areaB !== areaA) return areaB - areaA;
    const portraitA = a.height >= a.width ? 1 : 0;
    const portraitB = b.height >= b.width ? 1 : 0;
    return portraitB - portraitA;
  });

  return ranked[0]?.jpeg ?? candidates[0].jpeg;
}

async function probeImageSize(
  imagePath: string
): Promise<{ width: number; height: number } | null> {
  return new Promise((resolve) => {
    const child = spawn(
      ffprobeBin(),
      [
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        imagePath,
      ],
      { stdio: ["ignore", "pipe", "pipe"] }
    );
    let out = "";
    child.stdout?.on("data", (d) => {
      out += String(d);
    });
    child.on("error", () => resolve(null));
    child.on("close", () => {
      const m = out.trim().match(/^(\d+)x(\d+)/);
      if (!m) return resolve(null);
      resolve({ width: Number(m[1]), height: Number(m[2]) });
    });
  });
}

export async function extractCoverFromPdf(
  pdfBytes: Buffer,
  coverPage: number
): Promise<CoverResult> {
  const rendered = await renderPdfPageToJpeg(pdfBytes, coverPage);
  return {
    jpegPath: rendered.jpegPath,
    width: rendered.width,
    height: rendered.height,
    source: "pdf-page",
  };
}

export async function extractCoverFromEpub(epubBytes: Buffer): Promise<CoverResult> {
  const { coverJpegPath, artworks } = await extractEpubImages(epubBytes);
  if (!coverJpegPath) {
    throw new Error("Capa não encontrada neste EPUB.");
  }
  const dims = (await probeImageSize(coverJpegPath)) || { width: 0, height: 0 };
  void artworks;
  return {
    jpegPath: coverJpegPath,
    width: dims.width,
    height: dims.height,
    source: "epub-cover",
  };
}

function stripHtmlSimple(html: string): string {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/p>/gi, "\n\n")
    .replace(/<\/h[1-6]>/gi, "\n\n")
    .replace(/<\/div>/gi, "\n")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

async function readEpubChapter(
  epubBytes: Buffer,
  chapterIndex: number
): Promise<{ title: string; text: string; index: number; total: number }> {
  const workDir = path.join(tmpDir(), `epub-ch-${Date.now().toString(36)}`);
  fs.mkdirSync(workDir, { recursive: true });
  const epubPath = path.join(workDir, "book.epub");
  await fs.promises.writeFile(epubPath, epubBytes);

  try {
    const { EPub } = await import("epub2");
    return await new Promise((resolve, reject) => {
      const epub = new EPub(epubPath);
      epub.on("error", reject);
      epub.on("end", () => {
        void (async () => {
          try {
            const flow = epub.flow || [];
            if (flow.length === 0) {
              reject(new Error("O EPUB não possui seções legíveis."));
              return;
            }
            if (chapterIndex < 1 || chapterIndex > flow.length) {
              reject(
                new Error(
                  `Seção ${chapterIndex} fora do intervalo (1–${flow.length}).`
                )
              );
              return;
            }
            const chapter = flow[chapterIndex - 1];
            const title =
              String(chapter.title || "").trim() || `Seção ${chapterIndex}`;
            const html = await new Promise<string>((res) => {
              epub.getChapter(chapter.id, (err, text) => {
                res(err ? "" : text || "");
              });
            });
            resolve({
              title,
              text: stripHtmlSimple(html),
              index: chapterIndex,
              total: flow.length,
            });
          } catch (err) {
            reject(err);
          }
        })();
      });
      epub.parse();
    });
  } finally {
    await fs.promises.rm(workDir, { recursive: true, force: true }).catch(() => undefined);
  }
}

/** Chapter/section text preview (no canvas — works in packaged app without @napi-rs/canvas). */
export async function getEpubChapterPreview(
  epubBytes: Buffer,
  chapterIndex: number
): Promise<{ title: string; text: string; index: number; total: number }> {
  const chapter = await readEpubChapter(epubBytes, chapterIndex);
  const excerpt =
    chapter.text.length > 900 ? `${chapter.text.slice(0, 900).trim()}…` : chapter.text;
  return { ...chapter, text: excerpt || "Sem texto legível nesta seção." };
}

function wrapCanvasLines(
  ctx: { measureText: (t: string) => { width: number } },
  text: string,
  maxWidth: number,
  maxLines: number
): string[] {
  const words = text.split(/\s+/).filter(Boolean);
  const lines: string[] = [];
  let current = "";
  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (ctx.measureText(next).width <= maxWidth) {
      current = next;
      continue;
    }
    if (current) lines.push(current);
    current = word;
    if (lines.length >= maxLines) break;
  }
  if (current && lines.length < maxLines) lines.push(current);
  if (words.length > 0 && lines.length >= maxLines) {
    const last = lines[lines.length - 1];
    lines[lines.length - 1] =
      last.length > 3 ? `${last.slice(0, Math.max(1, last.length - 1))}…` : `${last}…`;
  }
  return lines;
}

/** Render a chapter/section preview card (title + opening text) as JPEG. */
export async function renderEpubChapterPreview(
  epubBytes: Buffer,
  chapterIndex: number
): Promise<CoverResult> {
  const chapter = await readEpubChapter(epubBytes, chapterIndex);
  const { createCanvas } = await import("@napi-rs/canvas");

  const width = 480;
  const height = 640;
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "#f7f4ef";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "#e2ddd4";
  ctx.lineWidth = 2;
  ctx.strokeRect(16, 16, width - 32, height - 32);

  ctx.fillStyle = "#8a847a";
  ctx.font = "600 13px sans-serif";
  ctx.fillText(`Seção ${chapter.index} de ${chapter.total}`, 36, 52);

  ctx.fillStyle = "#1c1917";
  ctx.font = "700 28px Georgia, serif";
  const titleLines = wrapCanvasLines(ctx, chapter.title, width - 72, 3);
  let y = 96;
  for (const line of titleLines) {
    ctx.fillText(line, 36, y);
    y += 36;
  }

  y += 12;
  ctx.strokeStyle = "#d6d0c6";
  ctx.beginPath();
  ctx.moveTo(36, y);
  ctx.lineTo(width - 36, y);
  ctx.stroke();
  y += 36;

  const body = chapter.text || "Sem texto legível nesta seção.";
  ctx.fillStyle = "#44403c";
  ctx.font = "400 16px Georgia, serif";
  const bodyLines = wrapCanvasLines(ctx, body, width - 72, 22);
  for (const line of bodyLines) {
    ctx.fillText(line, 36, y);
    y += 24;
    if (y > height - 40) break;
  }

  const jpegPath = uniqueTmp("jpg");
  await fs.promises.writeFile(jpegPath, canvas.toBuffer("image/jpeg", 85));
  return {
    jpegPath,
    width,
    height,
    source: "epub-chapter",
  };
}

export async function extractCover(opts: {
  fileData: string;
  fileType: "pdf" | "epub";
  coverPage?: number | null;
}): Promise<CoverResult> {
  const bytes = Buffer.from(opts.fileData, "base64");
  if (opts.fileType === "pdf") {
    const page = opts.coverPage == null || opts.coverPage < 1 ? 1 : Math.floor(opts.coverPage);
    return extractCoverFromPdf(bytes, page);
  }
  return extractCoverFromEpub(bytes);
}

export async function coverToBase64Jpeg(jpegPath: string): Promise<{
  imageData: string;
  mimeType: string;
  width: number;
  height: number;
}> {
  const buf = await fs.promises.readFile(jpegPath);
  const dims = (await probeImageSize(jpegPath)) || { width: 0, height: 0 };
  return {
    imageData: buf.toString("base64"),
    mimeType: "image/jpeg",
    width: dims.width,
    height: dims.height,
  };
}

/** Validate PDF page count quickly with pdf-lib. */
export async function getPdfPageCount(pdfBytes: Buffer): Promise<number> {
  const doc = await PDFDocument.load(pdfBytes, { ignoreEncryption: true });
  return doc.getPageCount();
}
