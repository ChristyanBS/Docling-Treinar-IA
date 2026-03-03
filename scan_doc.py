import gc
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog
from typing import Callable, Iterable

import customtkinter as ctk
import pypdfium2
from langdetect import detect, LangDetectException
from tkinterdnd2 import DND_FILES, TkinterDnD
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

THREAD_LOCAL = threading.local()
_JSONL_LOCK = threading.Lock()
OPUS_MT_MODEL_NAME = "Helsinki-NLP/opus-mt-en-ROMANCE"


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


def build_config_from_values(
    input_dir: str = str(DEFAULT_INPUT_DIR),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    profile: str = "safe",
    max_workers: int | None = None,
    file_batch_size: int | None = None,
    pdf_page_chunk_size: int | None = None,
    pdf_images_scale: float | None = None,
    pdf_model_batch_size: int | None = None,
    pdf_accel_threads: int | None = None,
    force_overwrite: bool = False,
) -> AppConfig:
    cpu = CPU_COUNT
    if profile == "fast":
        _w = max_workers or cpu
        _b = file_batch_size or max(2, cpu)
        _c = pdf_page_chunk_size or 8
        _s = pdf_images_scale or 0.75
        _m = pdf_model_batch_size or 2
        _a = pdf_accel_threads or max(1, cpu // 2)
    else:
        _w = max_workers or DEFAULT_MAX_WORKERS
        _b = file_batch_size or DEFAULT_FILE_BATCH_SIZE
        _c = pdf_page_chunk_size or DEFAULT_PDF_PAGE_CHUNK_SIZE
        _s = pdf_images_scale or DEFAULT_PDF_IMAGES_SCALE
        _m = pdf_model_batch_size or DEFAULT_PDF_MODEL_BATCH_SIZE
        _a = pdf_accel_threads or DEFAULT_PDF_ACCEL_THREADS

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


def chunked(items: list[Path], size: int) -> Iterable[list[Path]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def ensure_directories(cfg: AppConfig) -> None:
    cfg.input_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)


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


def get_thread_translator() -> tuple[MarianTokenizer, MarianMTModel]:
    """Retorna tokenizer + modelo opus-mt cacheados por thread (~300 MB, carrega 1x)."""
    tok = getattr(THREAD_LOCAL, "mt_tokenizer", None)
    mdl = getattr(THREAD_LOCAL, "mt_model", None)
    if tok is None or mdl is None:
        tok = MarianTokenizer.from_pretrained(OPUS_MT_MODEL_NAME)
        mdl = MarianMTModel.from_pretrained(OPUS_MT_MODEL_NAME)
        THREAD_LOCAL.mt_tokenizer = tok
        THREAD_LOCAL.mt_model = mdl
    return tok, mdl


# ─── Tradução Automática (opus-mt Helsinki-NLP) ───────────────────────────────

def detect_language(text: str) -> str:
    """Detecta o idioma do texto. Retorna código ISO-639-1 ou 'unknown'."""
    try:
        sample = text[:3000]
        return detect(sample)
    except LangDetectException:
        return "unknown"


def _translate_batch(sentences: list[str], tokenizer: MarianTokenizer,
                     model: MarianMTModel) -> list[str]:
    """Traduz um lote de frases EN→PT usando opus-mt."""
    # Prefixo >>pt<< seleciona português no modelo ROMANCE
    prefixed = [f">>pt<< {s}" for s in sentences]
    tokens = tokenizer(prefixed, return_tensors="pt", padding=True, truncation=True,
                       max_length=512)
    translated = model.generate(**tokens)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


def translate_text(text: str, target_lang: str = "pt") -> str:
    """Traduz texto EN→PT usando Helsinki-NLP/opus-mt via transformers."""
    tokenizer, model = get_thread_translator()
    paragraphs = text.split("\n")
    translated_parts: list[str] = []
    batch: list[str] = []

    MAX_BATCH = 32  # evita estouro de memória

    def flush_batch() -> None:
        if not batch:
            return
        # Processa em sub-lotes
        for i in range(0, len(batch), MAX_BATCH):
            sub = batch[i : i + MAX_BATCH]
            translated_parts.extend(_translate_batch(sub, tokenizer, model))
        batch.clear()

    for para in paragraphs:
        stripped = para.strip()
        # Preserva linhas vazias e marcadores Markdown sem traduzir
        if not stripped or stripped.startswith(("|", "---", "```", "![", "<!--")):
            flush_batch()
            translated_parts.append(para)
        else:
            batch.append(para)

    flush_batch()
    return "\n".join(translated_parts)


def collect_files(cfg: AppConfig) -> list[Path]:
    files: list[Path] = []
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


# ─── Etapa 1: Enriquecimento de Metadados ───────────────────────────────────

def generate_yaml_header(
    original_filename: str, doc_type: str, traduzido: bool = False
) -> str:
    """Gera cabeçalho YAML com metadados para enriquecimento do .md."""
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


def enrich_md_with_metadata(
    md_path: Path, original_file: Path, traduzido: bool = False
) -> None:
    """Lê o .md, prefixa cabeçalho YAML e salva de volta."""
    content = md_path.read_text(encoding="utf-8")
    ext = original_file.suffix.lower()
    doc_type = EXTENSION_MAP[ext].name if ext in EXTENSION_MAP else "UNKNOWN"
    header = generate_yaml_header(original_file.name, doc_type, traduzido=traduzido)
    md_path.write_text(header + content, encoding="utf-8")


# ─── Etapa 3: Preparação para RAG / Fine-Tuning ─────────────────────────────

def append_to_jsonl(jsonl_path: Path, md_content: str, original_file: Path,
                    traduzido: bool = False) -> None:
    """Anexa uma entrada ao arquivo JSONL mestre para RAG/Fine-Tuning."""
    ext = original_file.suffix.lower()
    doc_type = EXTENSION_MAP[ext].name if ext in EXTENSION_MAP else "UNKNOWN"
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


def convert_large_pdf_in_chunks(
    converter: DocumentConverter,
    file_path: Path,
    output_file: Path,
    cfg: AppConfig,
) -> None:
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

        # Conteúdo bruto (antes do YAML) — vai para o JSONL
        raw_md = output_file.read_text(encoding="utf-8")

        # ── Tradução automática EN → PT ──
        traduzido = False
        lang = detect_language(raw_md)
        if lang == "en":
            logging.info("🌐 Traduzindo de EN para PT: %s", file_path.name)
            raw_md = translate_text(raw_md, target_lang="pt")
            output_file.write_text(raw_md, encoding="utf-8")
            traduzido = True
            gc.collect()

        # ── Etapa 1: Enriquecimento de Metadados YAML ──
        enrich_md_with_metadata(output_file, file_path, traduzido=traduzido)

        gc.collect()
        return {"file": str(file_path), "status": "success",
                "seconds": time.perf_counter() - started, "error": "",
                "md_content": raw_md, "traduzido": traduzido}
    except Exception as exc:
        gc.collect()
        return {"file": str(file_path), "status": "error",
                "seconds": time.perf_counter() - started, "error": str(exc), "md_content": ""}


def run_pipeline(
    cfg: AppConfig,
    on_progress: Callable[[int, int, dict], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> list[dict]:
    """Executa o pipeline de alimentação de IA em 3 etapas."""
    configure_logging(cfg.log_file)
    ensure_directories(cfg)

    jsonl_path = cfg.output_dir / "conhecimento_cgr.jsonl"

    files = collect_files(cfg)
    if not files:
        if on_log:
            on_log(f"Nenhum arquivo compatível encontrado em {cfg.input_dir}")
        return []

    total = len(files)
    counters = {"success": 0, "skipped": 0, "error": 0}
    results: list[dict] = []
    processed = 0

    if on_log:
        on_log("═" * 52)
        on_log("   PIPELINE DE ALIMENTAÇÃO DE IA — 3 ETAPAS")
        on_log("═" * 52)
        on_log(f"Arquivos: {total}  |  Workers: {cfg.max_workers}")
        on_log("")
        on_log("[ETAPA 1] Conversão + Enriquecimento de Metadados YAML")
        on_log("[ETAPA 2] Visão Computacional (OCR) para imagens ativada")
        on_log("🌐 Tradução automática EN→PT (opus-mt) ativada")
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

                # Log de tradução
                if result.get("traduzido"):
                    if on_log:
                        on_log(f"   🌐 Traduzido de EN para PT: {Path(result['file']).name}")

                # ── Etapa 3: Exportar para JSONL ──
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


# ═══════════════════════════════  GUI (CustomTkinter)  ══════════════════════════

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class DoclingApp(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.TkdndVersion = TkinterDnD._require(self)
        self.title("Docling — Conversão de Documentos para Markdown")
        self.geometry("960x720")
        self.minsize(780, 580)

        self._cancel_event = threading.Event()
        self._running = False
        self._dropped_files: list[Path] = []

        self._build_ui()

    # ── Layout ──────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        # Grid principal: sidebar + conteúdo
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Sidebar ──
        sidebar = ctk.CTkFrame(self, width=260, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nswe")
        sidebar.grid_rowconfigure(20, weight=1)  # spacer

        ctk.CTkLabel(sidebar, text="⚙️  Configurações", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, padx=20, pady=(20, 10), sticky="w"
        )

        # Pasta entrada
        ctk.CTkLabel(sidebar, text="Pasta de entrada:").grid(row=1, column=0, padx=20, sticky="w")
        input_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        input_frame.grid(row=2, column=0, padx=20, pady=(0, 8), sticky="we")
        input_frame.grid_columnconfigure(0, weight=1)
        self._input_entry = ctk.CTkEntry(input_frame, placeholder_text=str(DEFAULT_INPUT_DIR))
        self._input_entry.grid(row=0, column=0, sticky="we")
        self._input_entry.insert(0, str(DEFAULT_INPUT_DIR))
        ctk.CTkButton(input_frame, text="📂", width=36, command=self._browse_input).grid(row=0, column=1, padx=(4, 0))

        # Pasta saída
        ctk.CTkLabel(sidebar, text="Pasta de saída:").grid(row=3, column=0, padx=20, sticky="w")
        output_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        output_frame.grid(row=4, column=0, padx=20, pady=(0, 8), sticky="we")
        output_frame.grid_columnconfigure(0, weight=1)
        self._output_entry = ctk.CTkEntry(output_frame, placeholder_text=str(DEFAULT_OUTPUT_DIR))
        self._output_entry.grid(row=0, column=0, sticky="we")
        self._output_entry.insert(0, str(DEFAULT_OUTPUT_DIR))
        ctk.CTkButton(output_frame, text="📂", width=36, command=self._browse_output).grid(row=0, column=1, padx=(4, 0))

        # Perfil
        ctk.CTkLabel(sidebar, text="Perfil:").grid(row=5, column=0, padx=20, sticky="w")
        self._profile_var = ctk.StringVar(value="safe")
        ctk.CTkSegmentedButton(
            sidebar, values=["safe", "fast"], variable=self._profile_var
        ).grid(row=6, column=0, padx=20, pady=(0, 12), sticky="we")

        # Checkbox reprocessar
        self._overwrite_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            sidebar, text="Reprocessar existentes",
            variable=self._overwrite_var,
            font=ctk.CTkFont(size=12),
            checkbox_width=20, checkbox_height=20,
        ).grid(row=7, column=0, padx=20, pady=(0, 8), sticky="w")

        # Sliders de configuração
        sep = ctk.CTkFrame(sidebar, height=2, fg_color="gray40")
        sep.grid(row=8, column=0, padx=20, pady=4, sticky="we")
        ctk.CTkLabel(sidebar, text="Ajustes avançados", font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=9, column=0, padx=20, pady=(4, 6), sticky="w"
        )

        self._workers_slider = self._add_slider(sidebar, 10, "Workers paralelos:", 1, CPU_COUNT * 2, CPU_COUNT)
        self._batch_slider = self._add_slider(sidebar, 12, "Tamanho do lote:", 1, 20, max(2, CPU_COUNT))
        self._chunk_slider = self._add_slider(sidebar, 14, "Páginas por chunk (PDF):", 1, 50, 5)
        self._scale_slider = self._add_slider(sidebar, 16, "Escala imagem OCR:", 0.2, 2.0, 0.5, resolution=0.05)
        self._model_batch_slider = self._add_slider(sidebar, 18, "Batch OCR/Layout:", 1, 8, 1)
        self._accel_slider = self._add_slider(sidebar, 20, "Threads pipeline PDF:", 1, CPU_COUNT, max(1, CPU_COUNT // 2))

        # Extensões suportadas
        exts_text = "  ".join(sorted(EXTENSION_MAP.keys()))
        ctk.CTkLabel(sidebar, text=f"Extensões: {exts_text}", wraplength=230,
                      font=ctk.CTkFont(size=11), text_color="gray60").grid(
            row=22, column=0, padx=20, pady=(8, 12), sticky="sw"
        )

        # ── Conteúdo principal ──
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.grid(row=0, column=1, sticky="nswe", padx=16, pady=16)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(4, weight=1)  # log expands

        ctk.CTkLabel(main_frame, text="📄  Docling — Conversão de Documentos",
                      font=ctk.CTkFont(size=22, weight="bold")).grid(row=0, column=0, sticky="w", pady=(0, 4))
        ctk.CTkLabel(main_frame, text="Converta PDF, DOCX, XLSX, PPTX, HTML, imagens e mais para Markdown.",
                      font=ctk.CTkFont(size=13), text_color="gray60").grid(row=1, column=0, sticky="w", pady=(0, 8))

        # ── Drop zone ──
        self._drop_frame = ctk.CTkFrame(main_frame, height=120, border_width=2,
                                         border_color="#3b8ed0", fg_color=("#f0f4fa", "#1a1a2e"))
        self._drop_frame.grid(row=2, column=0, sticky="we", pady=(0, 8))
        self._drop_frame.grid_propagate(False)
        self._drop_frame.grid_columnconfigure(0, weight=1)
        self._drop_frame.grid_rowconfigure(0, weight=1)

        self._drop_label = ctk.CTkLabel(
            self._drop_frame,
            text="📥  Arraste arquivos aqui\nou clique para selecionar",
            font=ctk.CTkFont(size=14),
            text_color=("#3b8ed0", "#6ea8d9"),
            justify="center",
        )
        self._drop_label.grid(row=0, column=0, columnspan=2, padx=20, pady=10)

        self._files_list_label = ctk.CTkLabel(
            self._drop_frame, text="", font=ctk.CTkFont(size=11),
            text_color="gray60", justify="left", wraplength=500,
        )
        self._files_list_label.grid(row=1, column=0, padx=20, pady=(0, 6), sticky="w")

        self._remove_files_btn = ctk.CTkButton(
            self._drop_frame, text="✖ Remover todos", width=120, height=28,
            font=ctk.CTkFont(size=12), fg_color="#c0392b", hover_color="#e74c3c",
            command=self._remove_files,
        )
        self._remove_files_btn.grid(row=1, column=1, padx=(4, 12), pady=(0, 6), sticky="e")
        self._remove_files_btn.grid_remove()  # esconde até ter arquivos

        # Bind drag-and-drop
        self._drop_frame.drop_target_register(DND_FILES)
        self._drop_frame.dnd_bind("<<Drop>>", self._on_drop)
        self._drop_frame.dnd_bind("<<DragEnter>>", self._on_drag_enter)
        self._drop_frame.dnd_bind("<<DragLeave>>", self._on_drag_leave)

        # Click to browse files
        self._drop_label.bind("<Button-1>", lambda e: self._browse_files())
        self._drop_frame.bind("<Button-1>", lambda e: self._browse_files())

        # Barra de progresso
        self._progress_bar = ctk.CTkProgressBar(main_frame, height=18)
        self._progress_bar.grid(row=3, column=0, sticky="we", pady=(0, 4))
        self._progress_bar.set(0)

        self._progress_label = ctk.CTkLabel(main_frame, text="Pronto", font=ctk.CTkFont(size=12))
        self._progress_label.grid(row=3, column=0, sticky="e", pady=(0, 4))

        # Log
        self._log_box = ctk.CTkTextbox(main_frame, font=ctk.CTkFont(family="Consolas", size=12),
                                         state="disabled", wrap="word")
        self._log_box.grid(row=4, column=0, sticky="nswe", pady=(4, 8))

        # Botões
        btn_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        btn_frame.grid(row=5, column=0, sticky="we")
        btn_frame.grid_columnconfigure(1, weight=1)

        self._start_btn = ctk.CTkButton(
            btn_frame, text="🚀  Iniciar conversão", font=ctk.CTkFont(size=14, weight="bold"),
            height=42, command=self._on_start
        )
        self._start_btn.grid(row=0, column=0, sticky="w")

        self._cancel_btn = ctk.CTkButton(
            btn_frame, text="⏹  Cancelar", font=ctk.CTkFont(size=14),
            height=42, fg_color="#c0392b", hover_color="#e74c3c",
            command=self._on_cancel, state="disabled"
        )
        self._cancel_btn.grid(row=0, column=1, sticky="w", padx=(8, 0))

        self._clear_btn = ctk.CTkButton(
            btn_frame, text="🗑  Limpar log", font=ctk.CTkFont(size=13),
            height=42, fg_color="gray40", hover_color="gray50",
            command=self._clear_log
        )
        self._clear_btn.grid(row=0, column=2, sticky="e")

    # ── Helpers ─────────────────────────────────────────────────────
    def _add_slider(
        self, parent, row: int, label: str, from_: float, to: float, default: float, resolution: float = 1.0
    ) -> ctk.CTkSlider:
        val_var = ctk.DoubleVar(value=default)
        fmt = ".2f" if resolution < 1 else ".0f"
        lbl = ctk.CTkLabel(parent, text=f"{label}  {default:{fmt}}", font=ctk.CTkFont(size=12))
        lbl.grid(row=row, column=0, padx=20, sticky="w")

        def _update(v):
            lbl.configure(text=f"{label}  {float(v):{fmt}}")

        slider = ctk.CTkSlider(parent, from_=from_, to=to, number_of_steps=int((to - from_) / resolution),
                                variable=val_var, command=_update)
        slider.grid(row=row + 1, column=0, padx=20, pady=(0, 6), sticky="we")
        return slider

    def _browse_input(self) -> None:
        d = filedialog.askdirectory(title="Selecione a pasta de entrada")
        if d:
            self._input_entry.delete(0, "end")
            self._input_entry.insert(0, d)

    def _browse_output(self) -> None:
        d = filedialog.askdirectory(title="Selecione a pasta de saída")
        if d:
            self._output_entry.delete(0, "end")
            self._output_entry.insert(0, d)

    def _browse_files(self) -> None:
        """Abre diálogo para selecionar arquivos (alternativa ao drag-and-drop)."""
        exts = [("Documentos suportados", " ".join(f"*{e}" for e in EXTENSION_MAP)), ("Todos", "*.*")]
        files = filedialog.askopenfilenames(title="Selecione arquivos para converter", filetypes=exts)
        if files:
            self._add_dropped_files([Path(f) for f in files])

    # ── Drag-and-drop ───────────────────────────────────────────────
    def _parse_drop_data(self, data: str) -> list[Path]:
        """Analisa a string de drop do tkdnd, lidando com caminhos com espaços."""
        paths: list[Path] = []
        raw = data.strip()
        i = 0
        while i < len(raw):
            if raw[i] == "{":
                end = raw.index("}", i)
                paths.append(Path(raw[i + 1 : end]))
                i = end + 2  # skip } and space
            else:
                end = raw.find(" ", i)
                if end == -1:
                    end = len(raw)
                paths.append(Path(raw[i:end]))
                i = end + 1
        return paths

    def _on_drop(self, event) -> None:
        """Callback quando arquivos são soltos na drop zone."""
        paths = self._parse_drop_data(event.data)
        valid: list[Path] = []
        for p in paths:
            if p.is_file() and p.suffix.lower() in EXTENSION_MAP:
                valid.append(p)
            elif p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file() and f.suffix.lower() in EXTENSION_MAP:
                        valid.append(f)
        if valid:
            self._add_dropped_files(valid)
        else:
            self._drop_label.configure(
                text="⚠️  Nenhum arquivo compatível encontrado.",
                text_color="#e74c3c",
            )
            self.after(2000, lambda: self._drop_label.configure(
                text="📥  Arraste arquivos aqui\nou clique para selecionar",
                text_color=("#3b8ed0", "#6ea8d9"),
            ))
        self._drop_frame.configure(border_color="#3b8ed0")

    def _on_drag_enter(self, event) -> None:
        self._drop_frame.configure(border_color="#27ae60", fg_color=("#e8f8f0", "#1a2e1a"))

    def _on_drag_leave(self, event) -> None:
        self._drop_frame.configure(border_color="#3b8ed0", fg_color=("#f0f4fa", "#1a1a2e"))

    def _add_dropped_files(self, files: list[Path]) -> None:
        """Copia os arquivos para a pasta de entrada e atualiza a lista."""
        input_dir = Path(self._input_entry.get())
        input_dir.mkdir(parents=True, exist_ok=True)

        added = 0
        for src in files:
            dst = input_dir / src.name
            if src.resolve() == dst.resolve():
                added += 1
                continue
            try:
                shutil.copy2(src, dst)
                added += 1
            except Exception as exc:
                self._log(f"[ERRO] Não foi possível copiar {src.name}: {exc}")

        self._log(f"+ {added} arquivo(s) adicionado(s) à pasta de entrada.")
        self._refresh_drop_zone()

    def _remove_files(self) -> None:
        """Remove todos os arquivos da pasta de entrada."""
        input_dir = Path(self._input_entry.get())
        if not input_dir.exists():
            return
        removed = 0
        for f in input_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in EXTENSION_MAP:
                try:
                    f.unlink()
                    removed += 1
                except Exception as exc:
                    self._log(f"[ERRO] Não foi possível remover {f.name}: {exc}")
        self._dropped_files.clear()
        self._log(f"🗑  {removed} arquivo(s) removido(s) da pasta de entrada.")
        self._refresh_drop_zone()

    def _refresh_drop_zone(self) -> None:
        """Atualiza o visual da drop zone com os arquivos atuais."""
        input_dir = Path(self._input_entry.get())
        cfg = build_config_from_values(input_dir=str(input_dir), output_dir=self._output_entry.get())
        all_files = collect_files(cfg) if input_dir.exists() else []
        self._dropped_files = all_files

        if all_files:
            names = [f.name for f in all_files[:8]]
            extra = f"  ... e mais {len(all_files) - 8}" if len(all_files) > 8 else ""
            self._drop_label.configure(
                text=f"✅  {len(all_files)} arquivo(s) na pasta de entrada",
                text_color=("#27ae60", "#2ecc71"),
            )
            self._files_list_label.configure(text="  •  ".join(names) + extra)
            self._remove_files_btn.grid()  # mostra o botão
        else:
            self._drop_label.configure(
                text="📥  Arraste arquivos aqui\nou clique para selecionar",
                text_color=("#3b8ed0", "#6ea8d9"),
            )
            self._files_list_label.configure(text="")
            self._remove_files_btn.grid_remove()  # esconde o botão

    def _log(self, msg: str) -> None:
        self._log_box.configure(state="normal")
        self._log_box.insert("end", msg + "\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _clear_log(self) -> None:
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")
        self._progress_bar.set(0)
        self._progress_label.configure(text="Pronto")

    # ── Callbacks de progresso (chamados da thread de trabalho) ────
    def _on_progress(self, done: int, total: int, counters: dict) -> None:
        frac = done / total if total else 0
        text = (
            f"{done}/{total}  —  OK: {counters['success']}  "
            f"Pulados: {counters['skipped']}  Erros: {counters['error']}"
        )
        self.after(0, self._progress_bar.set, frac)
        self.after(0, self._progress_label.configure, {"text": text})

    def _on_log_msg(self, msg: str) -> None:
        self.after(0, self._log, msg)

    # ── Start / Cancel ──────────────────────────────────────────────
    def _on_start(self) -> None:
        if self._running:
            return

        cfg = build_config_from_values(
            input_dir=self._input_entry.get(),
            output_dir=self._output_entry.get(),
            profile=self._profile_var.get(),
            max_workers=int(self._workers_slider.get()),
            file_batch_size=int(self._batch_slider.get()),
            pdf_page_chunk_size=int(self._chunk_slider.get()),
            pdf_images_scale=round(self._scale_slider.get(), 2),
            pdf_model_batch_size=int(self._model_batch_slider.get()),
            pdf_accel_threads=int(self._accel_slider.get()),
            force_overwrite=self._overwrite_var.get(),
        )

        self._running = True
        self._cancel_event.clear()
        self._start_btn.configure(state="disabled")
        self._cancel_btn.configure(state="normal")
        self._progress_bar.set(0)
        self._progress_label.configure(text="Preparando...")

        def _worker():
            try:
                run_pipeline(
                    cfg,
                    on_progress=self._on_progress,
                    on_log=self._on_log_msg,
                    cancel_event=self._cancel_event,
                )
            finally:
                self.after(0, self._finish_run)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_cancel(self) -> None:
        self._cancel_event.set()
        self._on_log_msg("⚠️  Cancelamento solicitado... aguardando tarefas em andamento.")

    def _finish_run(self) -> None:
        self._running = False
        self._start_btn.configure(state="normal")
        self._cancel_btn.configure(state="disabled")


# ═══════════════════════════════════  MAIN  ═════════════════════════════════════

def main() -> None:
    app = DoclingApp()
    app.mainloop()


if __name__ == "__main__":
    main()
