"""
Pipeline de processamento de documentos com Docling.
Extraído do scan_doc.py — sem dependências de GUI.
Mantém: threading, gc.collect(), tradução opus-mt, OCR de imagens, YAML + JSONL.
"""

import gc
import json
import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import pypdfium2
from langdetect import detect, LangDetectException
from transformers import MarianMTModel, MarianTokenizer

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

# ═══════════════════════════════════  CONFIG  ═══════════════════════════════════

@dataclass
class AppConfig:
    input_dir: Path
    output_dir: Path
    log_file: Path
    max_workers: int
    file_batch_size: int
    pdf_page_chunk_size: int
    pdf_images_scale: float
    pdf_model_batch_size: int
    pdf_accel_threads: int
    force_overwrite: bool = False


DEFAULT_INPUT_DIR = Path("./entrada_cgr")
DEFAULT_OUTPUT_DIR = Path("./saida_ia")
DEFAULT_LOG_FILE = Path("docling_processamento.log")

CPU_COUNT = os.cpu_count() or 1
DEFAULT_MAX_WORKERS = CPU_COUNT
DEFAULT_FILE_BATCH_SIZE = max(2, CPU_COUNT)
DEFAULT_PDF_PAGE_CHUNK_SIZE = 5
DEFAULT_PDF_IMAGES_SCALE = 0.5
DEFAULT_PDF_MODEL_BATCH_SIZE = 1
DEFAULT_PDF_ACCEL_THREADS = max(1, CPU_COUNT // 2)

EXTENSION_MAP: dict[str, InputFormat] = {
    ".pdf": InputFormat.PDF,
    ".docx": InputFormat.DOCX,
    ".xlsx": InputFormat.XLSX,
    ".pptx": InputFormat.PPTX,
    ".html": InputFormat.HTML,
    ".htm": InputFormat.HTML,
    ".csv": InputFormat.CSV,
    ".md": InputFormat.MD,
    ".adoc": InputFormat.ASCIIDOC,
    ".asciidoc": InputFormat.ASCIIDOC,
    ".tex": InputFormat.LATEX,
    ".latex": InputFormat.LATEX,
    ".png": InputFormat.IMAGE,
    ".jpg": InputFormat.IMAGE,
    ".jpeg": InputFormat.IMAGE,
    ".bmp": InputFormat.IMAGE,
    ".tif": InputFormat.IMAGE,
    ".tiff": InputFormat.IMAGE,
    ".webp": InputFormat.IMAGE,
    ".gif": InputFormat.IMAGE,
}

SUPPORTED_EXTENSIONS = set(EXTENSION_MAP.keys())

THREAD_LOCAL = threading.local()
_JSONL_LOCK = threading.Lock()
OPUS_MT_MODEL_NAME = "Helsinki-NLP/opus-mt-en-ROMANCE"

logger = logging.getLogger("pipeline")

# ═══════════════════════════════  PIPELINE CORE  ════════════════════════════════

def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def build_config(
    input_dir: str = str(DEFAULT_INPUT_DIR),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    profile: str = "safe",
    force_overwrite: bool = False,
) -> AppConfig:
    cpu = CPU_COUNT
    if profile == "fast":
        _w, _b, _c = cpu, max(2, cpu), 8
        _s, _m, _a = 0.75, 2, max(1, cpu // 2)
    else:
        _w = DEFAULT_MAX_WORKERS
        _b = DEFAULT_FILE_BATCH_SIZE
        _c = DEFAULT_PDF_PAGE_CHUNK_SIZE
        _s = DEFAULT_PDF_IMAGES_SCALE
        _m = DEFAULT_PDF_MODEL_BATCH_SIZE
        _a = DEFAULT_PDF_ACCEL_THREADS
    return AppConfig(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        log_file=DEFAULT_LOG_FILE,
        max_workers=max(1, _w),
        file_batch_size=max(1, _b),
        pdf_page_chunk_size=max(1, _c),
        pdf_images_scale=max(0.2, min(2.0, _s)),
        pdf_model_batch_size=max(1, _m),
        pdf_accel_threads=max(1, _a),
        force_overwrite=force_overwrite,
    )


def chunked(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def ensure_directories(cfg: AppConfig) -> None:
    cfg.input_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)


# ─── Converter (Docling + OCR para imagens) ─────────────────────────────────

def build_converter(cfg: AppConfig) -> DocumentConverter:
    pdf_options = PdfPipelineOptions(
        do_table_structure=True,
        do_ocr=True,
        images_scale=cfg.pdf_images_scale,
        ocr_batch_size=cfg.pdf_model_batch_size,
        layout_batch_size=cfg.pdf_model_batch_size,
        table_batch_size=cfg.pdf_model_batch_size,
        accelerator_options=AcceleratorOptions(
            num_threads=cfg.pdf_accel_threads,
            device="cpu",
        ),
    )
    return DocumentConverter(
        allowed_formats=list(set(EXTENSION_MAP.values())),
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pdf_options),
        },
    )


def get_thread_converter(cfg: AppConfig) -> DocumentConverter:
    converter = getattr(THREAD_LOCAL, "converter", None)
    if converter is None:
        converter = build_converter(cfg)
        THREAD_LOCAL.converter = converter
    return converter


# ─── Tradução Automática (opus-mt Helsinki-NLP) ─────────────────────────────

def get_thread_translator() -> tuple[MarianTokenizer, MarianMTModel]:
    tok = getattr(THREAD_LOCAL, "mt_tokenizer", None)
    mdl = getattr(THREAD_LOCAL, "mt_model", None)
    if tok is None or mdl is None:
        tok = MarianTokenizer.from_pretrained(OPUS_MT_MODEL_NAME)
        mdl = MarianMTModel.from_pretrained(OPUS_MT_MODEL_NAME)
        THREAD_LOCAL.mt_tokenizer = tok
        THREAD_LOCAL.mt_model = mdl
    return tok, mdl


def detect_language(text: str) -> str:
    try:
        return detect(text[:3000])
    except LangDetectException:
        return "unknown"


def _translate_batch(sentences: list[str], tokenizer: MarianTokenizer,
                     model: MarianMTModel) -> list[str]:
    prefixed = [f">>pt<< {s}" for s in sentences]
    tokens = tokenizer(prefixed, return_tensors="pt", padding=True, truncation=True,
                       max_length=512)
    translated = model.generate(**tokens)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


def translate_text(text: str, target_lang: str = "pt") -> str:
    tokenizer, model = get_thread_translator()
    paragraphs = text.split("\n")
    translated_parts: list[str] = []
    batch: list[str] = []
    MAX_BATCH = 32

    def flush():
        if not batch:
            return
        for i in range(0, len(batch), MAX_BATCH):
            translated_parts.extend(_translate_batch(batch[i:i+MAX_BATCH], tokenizer, model))
        batch.clear()

    for para in paragraphs:
        s = para.strip()
        if not s or s.startswith(("|", "---", "```", "![", "<!--")):
            flush()
            translated_parts.append(para)
        else:
            batch.append(para)
    flush()
    return "\n".join(translated_parts)


# ─── Metadados YAML ─────────────────────────────────────────────────────────

def generate_yaml_header(original_filename: str, doc_type: str, traduzido: bool = False) -> str:
    flag = "sim" if traduzido else "não"
    return (
        "---\n"
        f'arquivo_original: "{original_filename}"\n'
        f'data_processamento: "{datetime.now().isoformat()}"\n'
        f'tipo_documento: "{doc_type}"\n'
        'setor: "CGR_Telecom"\n'
        f'traduzido: "{flag}"\n'
        "---\n\n"
    )


def enrich_md_with_metadata(md_path: Path, original_file: Path, traduzido: bool = False) -> None:
    content = md_path.read_text(encoding="utf-8")
    ext = original_file.suffix.lower()
    doc_type = EXTENSION_MAP.get(ext, InputFormat.PDF).name
    header = generate_yaml_header(original_file.name, doc_type, traduzido=traduzido)
    md_path.write_text(header + content, encoding="utf-8")


# ─── JSONL para RAG / Fine-Tuning ───────────────────────────────────────────

def append_to_jsonl(jsonl_path: Path, md_content: str, original_file: Path,
                    traduzido: bool = False) -> None:
    ext = original_file.suffix.lower()
    doc_type = EXTENSION_MAP.get(ext, InputFormat.PDF).name
    entry = {
        "text": md_content,
        "metadata": {
            "arquivo_original": original_file.name,
            "data_processamento": datetime.now().isoformat(),
            "tipo_documento": doc_type,
            "setor": "CGR_Telecom",
            "traduzido": "sim" if traduzido else "não",
        },
    }
    with _JSONL_LOCK:
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ─── Helpers PDF ─────────────────────────────────────────────────────────────

def collect_files(cfg: AppConfig) -> list[Path]:
    files = []
    for fp in cfg.input_dir.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in EXTENSION_MAP:
            files.append(fp)
    return sorted(files)


def output_path_for(file_path: Path, cfg: AppConfig) -> Path:
    relative = file_path.relative_to(cfg.input_dir)
    out = cfg.output_dir / relative.with_suffix(".md")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def checkpoint_path_for(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.checkpoint.json")


def save_checkpoint(cp: Path, payload: dict) -> None:
    tmp = cp.with_suffix(cp.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(cp)


def load_checkpoint(cp: Path) -> dict | None:
    if not cp.exists():
        return None
    try:
        return json.loads(cp.read_text(encoding="utf-8"))
    except Exception:
        return None


def count_pdf_pages(file_path: Path) -> int:
    doc = pypdfium2.PdfDocument(str(file_path))
    n = len(doc)
    doc.close()
    return n


def split_pdf_segment(src: Path, start: int, end: int, dst: Path) -> None:
    source = pypdfium2.PdfDocument(str(src))
    target = pypdfium2.PdfDocument.new()
    for i in range(start, min(end, len(source))):
        target.import_pages(source, pages=[i])
    target.save(str(dst))
    target.close()
    source.close()


def convert_large_pdf_in_chunks(converter, file_path, output_file, cfg):
    total_pages = count_pdf_pages(file_path)
    chunks = (total_pages + cfg.pdf_page_chunk_size - 1) // cfg.pdf_page_chunk_size
    cp_file = checkpoint_path_for(output_file)
    cp = load_checkpoint(cp_file)
    done = 0
    if cp and cp.get("source_file") == str(file_path):
        done = max(0, min(int(cp.get("completed_chunks", 0)), chunks))
    mode = "a" if (done > 0 and output_file.exists()) else "w"
    if mode == "w":
        done = 0
    save_checkpoint(cp_file, {
        "source_file": str(file_path), "output_file": str(output_file),
        "total_pages": total_pages, "total_chunks": chunks,
        "chunk_size": cfg.pdf_page_chunk_size, "completed_chunks": done,
        "updated_at": time.time(),
    })
    with tempfile.TemporaryDirectory(prefix="docling_chunk_") as tmp:
        root = Path(tmp)
        for ci in range(done, chunks):
            sp = ci * cfg.pdf_page_chunk_size
            ep = min(sp + cfg.pdf_page_chunk_size, total_pages)
            tmp_pdf = root / f"chunk_{ci:04d}.pdf"
            split_pdf_segment(file_path, sp, ep, tmp_pdf)
            conv = converter.convert(str(tmp_pdf))
            md = conv.document.export_to_markdown()
            with output_file.open(mode, encoding="utf-8") as f:
                if output_file.exists() and output_file.stat().st_size > 0:
                    f.write("\n\n")
                f.write(md)
            mode = "a"
            save_checkpoint(cp_file, {
                "source_file": str(file_path), "output_file": str(output_file),
                "total_pages": total_pages, "total_chunks": chunks,
                "chunk_size": cfg.pdf_page_chunk_size, "completed_chunks": ci + 1,
                "updated_at": time.time(),
            })
            del conv
            tmp_pdf.unlink(missing_ok=True)
            gc.collect()
    cp_file.unlink(missing_ok=True)


# ═══════════════════════════════  PROCESS FILE  ═════════════════════════════════

def process_file(file_path: Path, cfg: AppConfig) -> dict:
    started = time.perf_counter()
    output_file = output_path_for(file_path, cfg)
    cp_file = checkpoint_path_for(output_file)

    if output_file.exists() and not cp_file.exists() and not cfg.force_overwrite:
        return {"file": str(file_path), "status": "skipped",
                "seconds": time.perf_counter() - started, "error": "", "md_content": ""}
    try:
        converter = get_thread_converter(cfg)
        if file_path.suffix.lower() == ".pdf" and count_pdf_pages(file_path) > cfg.pdf_page_chunk_size:
            convert_large_pdf_in_chunks(converter, file_path, output_file, cfg)
        else:
            conv = converter.convert(str(file_path))
            md_text = conv.document.export_to_markdown()
            output_file.write_text(md_text, encoding="utf-8")
            del conv

        raw_md = output_file.read_text(encoding="utf-8")

        # Tradução automática EN → PT
        traduzido = False
        lang = detect_language(raw_md)
        if lang == "en":
            logger.info("🌐 Traduzindo de EN para PT: %s", file_path.name)
            raw_md = translate_text(raw_md, target_lang="pt")
            output_file.write_text(raw_md, encoding="utf-8")
            traduzido = True
            gc.collect()

        # Enriquecimento de Metadados YAML
        enrich_md_with_metadata(output_file, file_path, traduzido=traduzido)

        gc.collect()
        return {"file": str(file_path), "status": "success",
                "seconds": time.perf_counter() - started, "error": "",
                "md_content": raw_md, "traduzido": traduzido}
    except Exception as exc:
        gc.collect()
        return {"file": str(file_path), "status": "error",
                "seconds": time.perf_counter() - started, "error": str(exc), "md_content": ""}


# ═══════════════════════════════  RUN PIPELINE  ═════════════════════════════════

def run_pipeline(
    cfg: AppConfig,
    on_progress: Callable[[int, int, dict], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> list[dict]:
    configure_logging(cfg.log_file)
    ensure_directories(cfg)

    jsonl_path = cfg.output_dir / "conhecimento_cgr.jsonl"
    files = collect_files(cfg)
    if not files:
        if on_log:
            on_log("Nenhum arquivo compatível encontrado em " + str(cfg.input_dir))
        return []

    total = len(files)
    counters = {"success": 0, "skipped": 0, "error": 0}
    results: list[dict] = []
    processed = 0

    if on_log:
        on_log("═" * 52)
        on_log("   PIPELINE DE ALIMENTAÇÃO DE IA — CGR Telecom")
        on_log("═" * 52)
        on_log(f"Arquivos: {total}  |  Workers: {cfg.max_workers}")
        on_log("[ETAPA 1] Conversão + Enriquecimento de Metadados YAML")
        on_log("[ETAPA 2] OCR para imagens + Tradução EN→PT (opus-mt)")
        on_log(f"[ETAPA 3] Exportação JSONL → {jsonl_path.name}")
        on_log("─" * 52)

    executor = ThreadPoolExecutor(max_workers=cfg.max_workers)
    try:
        for file_batch in chunked(files, cfg.file_batch_size):
            if cancel_event and cancel_event.is_set():
                break
            futures = [executor.submit(process_file, fp, cfg) for fp in file_batch]
            for future in as_completed(futures):
                if cancel_event and cancel_event.is_set():
                    break
                result = future.result()
                results.append(result)
                counters[result["status"]] += 1
                processed += 1

                icon = {"success": "[OK]", "skipped": "[PULADO]", "error": "[ERRO]"}[result["status"]]
                msg = f"{icon}  {Path(result['file']).name}  ({result['seconds']:.1f}s)"
                if result["error"]:
                    msg += f"  — {result['error']}"
                if on_log:
                    on_log(msg)

                if result.get("traduzido") and on_log:
                    on_log(f"   🌐 Traduzido de EN para PT: {Path(result['file']).name}")

                if result["status"] == "success" and result.get("md_content"):
                    try:
                        append_to_jsonl(
                            jsonl_path, result["md_content"],
                            Path(result["file"]),
                            traduzido=result.get("traduzido", False),
                        )
                        if on_log:
                            on_log(f"   ↳ JSONL atualizado: {Path(result['file']).name}")
                    except Exception as jsonl_exc:
                        if on_log:
                            on_log(f"   ↳ [ERRO JSONL] {jsonl_exc}")

                if on_progress:
                    on_progress(processed, total, counters)
            gc.collect()
    except Exception as exc:
        if on_log:
            on_log(f"[ERRO FATAL] {exc}")
    finally:
        executor.shutdown(wait=True, cancel_futures=False)

    if on_log:
        on_log("")
        on_log("═" * 52)
        on_log(
            f"Finalizado — OK: {counters['success']}  |  "
            f"Pulados: {counters['skipped']}  |  Erros: {counters['error']}"
        )
        if jsonl_path.exists():
            lines = sum(1 for _ in jsonl_path.open(encoding="utf-8"))
            on_log(f"JSONL mestre: {jsonl_path} ({lines} entradas)")
        on_log("═" * 52)
    return results
