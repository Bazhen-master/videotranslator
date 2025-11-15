# PART 1 — Imports, DPI awareness, scaling, configuration

import glob
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import configparser
import ctypes

# GUI imports — may be unavailable in some environments (e.g. sandbox)
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ModuleNotFoundError:
    tk = None
    filedialog = None
    messagebox = None
    ttk = None

# OpenAI SDK import — may also be unavailable
try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

# ---------- DPI AWARENESS (sharp UI on high‑DPI screens) ----------
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # Avoid blurry scaling
except Exception:
    pass

# ---------- TKINTER SCALING (makes widgets larger on 125–150% DPI) ----------
# Will be applied later in main() after root creation

tk_scaling_multiplier = 1.3  # 130% scaling – adjustable


def apply_tk_scaling(root):
    try:
        root.tk.call("tk", "scaling", tk_scaling_multiplier)
    except Exception:
        pass


# ---------- DEFAULT FONT SIZE BOOST ----------

def apply_default_font(root, size=11):
    try:
        root.option_add("*Font", f"{size}")
    except Exception:
        pass


# ---------- CONFIG ----------
CONFIG_FILE = "config.ini"
MAX_WORKERS = 4
INPUT_DIR_DEFAULT = "input"
OUTPUT_DIR_DEFAULT = "output"
LOG_FILE = "logs.txt"
DEFAULT_FFMPEG_PATH = r"D:\\ffmpeg\\bin\\ffmpeg.exe"
CHUNK_DURATION_SECONDS = 9 * 60

FFMPEG_BIN = os.environ.get("FFMPEG_PATH", DEFAULT_FFMPEG_PATH)
FFPROBE_BIN = os.environ.get("FFPROBE_PATH")
if not FFPROBE_BIN and os.path.isabs(FFMPEG_BIN):
    probe_candidate = os.path.join(
        os.path.dirname(FFMPEG_BIN),
        "ffprobe.exe" if os.name == "nt" else "ffprobe",
    )
    if os.path.exists(probe_candidate):
        FFPROBE_BIN = probe_candidate
if not FFPROBE_BIN:
    FFPROBE_BIN = "ffprobe"


# ---------- LOAD API KEY ----------

def load_api_key():
    cfg = configparser.ConfigParser()
    if os.path.exists(CONFIG_FILE):
        cfg.read(CONFIG_FILE, encoding="utf-8")
        if "openai" in cfg and "api_key" in cfg["openai"]:
            return cfg["openai"]["api_key"].strip()
    return os.environ.get("OPENAI_API_KEY", "").strip()


def make_client():
    if OpenAI is None:
        # Friendly error if the SDK is not installed in this environment
        raise RuntimeError(
            "Python package 'openai' is not installed. "
            "Install it with 'pip install openai' and run the program again."
        )
    api_key = load_api_key()
    if not api_key:
        raise RuntimeError(
            "API key не задан. Установите в config.ini или через переменную среды OPENAI_API_KEY."
        )
    return OpenAI(api_key=api_key)


# ---------- LOGGING ----------
log_queue = queue.Queue()


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    log_queue.put(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# Simple internal self-test (not executed automatically) ---------------------

def _self_test_env():
    """Minimal self-test to verify core constants and environment wiring.

    This is *not* run by default, but can be called manually if needed.
    """
    assert isinstance(CONFIG_FILE, str)
    assert isinstance(INPUT_DIR_DEFAULT, str)
    assert isinstance(OUTPUT_DIR_DEFAULT, str)


# PART 2 — FFmpeg utilities, transcription, translation, TTS segment builder

# ---------- MEDIA UTILITIES ----------

def run_subprocess(cmd, error_message):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"{error_message}: {proc.stderr.decode(errors='ignore')}")
    return proc


def get_media_duration(path):
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    proc = run_subprocess(cmd, f"ffprobe error for {path}")
    try:
        return float(proc.stdout.decode().strip())
    except ValueError:
        raise RuntimeError(f"Не удалось определить длительность файла {path}")


def extract_mp3(video_path, audio_path):
    log(f"Extracting audio: {video_path} -> {audio_path}")
    cmd = [FFMPEG_BIN, "-y", "-i", video_path, "-vn", "-acodec", "mp3", audio_path]
    run_subprocess(cmd, f"ffmpeg audio extraction error for {video_path}")
    return audio_path


# ---------- TRANSCRIPTION (JSON with segments) ----------

def transcribe_audio(client, audio_path):
    log(f"Transcribing (gpt-4o-transcribe, json): {audio_path}")
    with open(audio_path, "rb") as f:
        tr = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f,
            response_format="json",
        )
    return tr


# ---------- TRANSLATION OF PER-SEGMENT TEXT ----------

def translate_segment_text(client, text: str) -> str:
    log("Translating segment (DE → EN)...")
    if not text.strip():
        return ""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You translate German spoken text into natural, neutral English. "
                    "Keep the meaning, make it sound like real spoken dialogue. "
                    "Do not add explanations."
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    message = response.choices[0].message
    return getattr(message, "content", "") if message else ""


# ---------- TTS FOR A SINGLE SEGMENT ----------

def text_to_speech_segment(client, text: str, output_path: str, voice: str):
    if not text.strip():
        raise RuntimeError("Empty segment text for TTS")
    log(f"Generating TTS (voice={voice}) -> {output_path}")
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        format="mp3",
        input=text,
    )
    audio_bytes = response.read()
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    return output_path


# ---------- AUDIO STRETCHING ----------

def stretch_audio_to_duration(input_path, output_path, target_duration):
    current_duration = max(get_media_duration(input_path), 0.01)
    target_duration = max(float(target_duration), 0.01)
    speed = current_duration / target_duration

    if speed < 0.5 or speed > 2.0:
        log(f"Warning: atempo={speed:.3f} out of range → clamped.")
        speed = max(0.5, min(2.0, speed))

    log(
        f"Stretching {input_path} ({current_duration:.2f}s → {target_duration:.2f}s) atempo={speed:.3f}"
    )

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        input_path,
        "-filter:a",
        f"atempo={speed:.6f}",
        output_path,
    ]
    run_subprocess(cmd, f"ffmpeg atempo error for {input_path}")
    return output_path


# ---------- SILENCE GENERATOR ----------

def make_silence(duration, output_path):
    duration = float(duration)
    if duration <= 0.01:
        duration = 0.01
    log(f"Generating silence {duration:.2f}s → {output_path}")
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=44100:cl=stereo",
        "-t",
        f"{duration:.6f}",
        "-q:a",
        "9",
        "-acodec",
        "mp3",
        output_path,
    ]
    run_subprocess(cmd, f"ffmpeg silence generation error for {output_path}")
    return output_path


# ---------- AUDIO NORMALIZATION (DYNAUDNORM) ----------

def normalize_audio_dynaudnorm(input_path, output_path):
    log(f"Normalizing audio (dynaudnorm): {input_path} → {output_path}")
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        input_path,
        "-af",
        "dynaudnorm",
        "-c:a",
        "mp3",
        "-b:a",
        "192k",
        output_path,
    ]
    run_subprocess(cmd, f"ffmpeg dynaudnorm error for {input_path}")
    return output_path


# ---------- SRT UTILITIES ----------

def _format_srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    ms = int(round(seconds * 1000))
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(subs, srt_path):
    if not subs:
        return None
    log(f"Writing SRT → {srt_path}")
    with open(srt_path, "w", encoding="utf-8") as f:
        for sub in subs:
            f.write(f"{sub['index']}\n")
            f.write(
                f"{_format_srt_time(sub['start'])} --> {_format_srt_time(sub['end'])}\n"
            )
            f.write(sub["text"].strip() + "\n\n")
    return srt_path


# PART 3 — Accurate TTS builder, Quick TTS builder, subtitles, burn‑in muxing

# ---------- ACCURATE MODE: SEGMENT‑ALIGNED TTS ----------

def build_aligned_tts_for_chunk(client, chunk_audio_path, work_dir, voice: str):
    log(f"Building ACCURATE TTS for chunk: {chunk_audio_path}")
    tr = transcribe_audio(client, chunk_audio_path)

    # Extract segments from JSON
    segments = tr.get("segments") if isinstance(tr, dict) else getattr(tr, "segments", None)

    if not segments:
        log("No segments detected → fallback to single-segment mode.")
        full_text = tr.get("text", "") if isinstance(tr, dict) else getattr(tr, "text", "")
        segments = [{"start": 0.0, "end": get_media_duration(chunk_audio_path), "text": full_text}]

    files_for_concat = []
    subs = []
    current_time = 0.0
    chunk_duration = get_media_duration(chunk_audio_path)

    for idx, seg in enumerate(segments):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        de_text = (seg.get("text") or "").strip()
        target_duration = max(end - start, 0.01)

        # Silence before the segment
        if start > current_time + 0.05:
            gap = start - current_time
            silence_out = os.path.join(work_dir, f"sil_{idx:03d}.mp3")
            make_silence(gap, silence_out)
            files_for_concat.append(silence_out)
            current_time += gap

        if not de_text:
            current_time = end
            continue

        en_text = translate_segment_text(client, de_text).strip()
        if not en_text:
            current_time = end
            continue

        raw_seg = os.path.join(work_dir, f"seg_{idx:03d}_raw.mp3")
        fixed_seg = os.path.join(work_dir, f"seg_{idx:03d}_fix.mp3")

        text_to_speech_segment(client, en_text, raw_seg, voice)
        stretch_audio_to_duration(raw_seg, fixed_seg, target_duration)

        files_for_concat.append(fixed_seg)

        subs.append({
            "index": len(subs) + 1,
            "start": start,
            "end": end,
            "text": en_text,
        })

        current_time = end

    # Tail silence
    if chunk_duration > current_time + 0.05:
        tail = chunk_duration - current_time
        tail_path = os.path.join(work_dir, "tail_sil.mp3")
        make_silence(tail, tail_path)
        files_for_concat.append(tail_path)

    # If totally empty
    if not files_for_concat:
        fallback_sil = os.path.join(work_dir, "empty_sil.mp3")
        make_silence(chunk_duration, fallback_sil)
        files_for_concat.append(fallback_sil)

    # Concat everything
    concat_list = os.path.join(work_dir, "concat_acc.txt")
    aligned_out = os.path.join(work_dir, "aligned_raw.mp3")

    with open(concat_list, "w", encoding="utf-8") as f:
        for p in files_for_concat:
            f.write(f"file '{p}'\n")

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_list,
        "-c",
        "copy",
        aligned_out,
    ]
    run_subprocess(cmd, "ffmpeg concat (accurate) failed")

    # Normalize audio using dynaudnorm
    norm_out = os.path.join(work_dir, "aligned_norm.mp3")
    normalize_audio_dynaudnorm(aligned_out, norm_out)

    # Write subtitles
    srt_path = None
    if subs:
        srt_path = os.path.join(work_dir, "subs_en.srt")
        write_srt(subs, srt_path)

    return norm_out, srt_path


# ---------- QUICK MODE: ONE TTS FOR WHOLE CHUNK ----------

def build_quick_tts_for_chunk(client, chunk_audio_path, work_dir, voice: str):
    log(f"Building QUICK TTS for chunk: {chunk_audio_path}")
    tr = transcribe_audio(client, chunk_audio_path)

    segments = tr.get("segments") if isinstance(tr, dict) else getattr(tr, "segments", None)
    chunk_duration = get_media_duration(chunk_audio_path)

    de_texts = []
    if segments:
        for seg in segments:
            t = (seg.get("text") or "").strip()
            if t:
                de_texts.append(t)
    else:
        full_text = tr.get("text", "") if isinstance(tr, dict) else getattr(tr, "text", "")
        if full_text:
            de_texts.append(full_text.strip())

    if not de_texts:
        de_texts = [""]

    full_de = " ".join(de_texts).strip()
    en_text = translate_segment_text(client, full_de).strip()

    raw_path = os.path.join(work_dir, "quick_raw.mp3")
    fixed_path = os.path.join(work_dir, "quick_fix.mp3")

    text_to_speech_segment(client, en_text, raw_path, voice)
    stretch_audio_to_duration(raw_path, fixed_path, chunk_duration)

    norm_path = os.path.join(work_dir, "quick_norm.mp3")
    normalize_audio_dynaudnorm(fixed_path, norm_path)

    # One subtitle
    srt_path = os.path.join(work_dir, "quick_en.srt")
    subs = [{"index": 1, "start": 0.0, "end": chunk_duration, "text": en_text or ""}]
    write_srt(subs, srt_path)

    return norm_path, srt_path


# ---------- BURN‑IN SUBTITLES + AUDIO MUXING ----------

def escape_ffmpeg_path(path: str):
    return (
        path.replace("\\", "\\\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace(",", r"\,")
    )


def replace_audio_with_tts(video_path, audio_path, output_path, srt_path=None, burn_subtitles=True):
    log(f"Muxing audio + burn‑in subtitles (exists={bool(srt_path)}): {video_path}")

    burn_in_subs = bool(burn_subtitles and srt_path and os.path.exists(srt_path))

    if burn_in_subs:
        srt_clean = srt_path.replace("\\", "/")
        escaped = escape_ffmpeg_path(srt_clean)
        filter_complex = f"[0:v]subtitles='{escaped}'[v]"

        cmd = [
            FFMPEG_BIN,
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            output_path,
        ]
    else:
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            output_path,
        ]

    run_subprocess(cmd, f"ffmpeg mux error for {video_path}")
    return output_path


# PART 4 — Chunk split/concat, workers, GUI, scaling, main()

# ---------- SPLIT VIDEO INTO CHUNKS ----------

def split_video_into_chunks(video_path, work_dir):
    base = os.path.splitext(os.path.basename(video_path))[0]
    chunk_pattern = os.path.join(work_dir, f"{base}_chunk_%03d.mp4")

    for old in glob.glob(os.path.join(work_dir, f"{base}_chunk_*.mp4")):
        os.remove(old)

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        video_path,
        "-c",
        "copy",
        "-map",
        "0",
        "-f",
        "segment",
        "-segment_time",
        str(CHUNK_DURATION_SECONDS),
        "-reset_timestamps",
        "1",
        chunk_pattern,
    ]
    run_subprocess(cmd, f"ffmpeg split error for {video_path}")

    chunks = sorted(glob.glob(os.path.join(work_dir, f"{base}_chunk_*.mp4")))
    if not chunks:
        raise RuntimeError("Не удалось нарезать видео на части.")
    return chunks


# ---------- CONCATENATE CHUNKS ----------

def concatenate_chunks(chunk_paths, output_path, work_dir):
    if len(chunk_paths) == 1:
        shutil.copy2(chunk_paths[0], output_path)
        return output_path

    list_path = os.path.join(work_dir, "concat_chunks.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in chunk_paths:
            f.write(f"file '{p}'\n")

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-c",
        "copy",
        output_path,
    ]
    run_subprocess(cmd, "ffmpeg final concat error")
    return output_path


# ---------- PROCESS SINGLE FILE ----------

def process_single_file(video_path, output_folder, voice: str, mode: str, burn_subtitles=True):
    os.makedirs(output_folder, exist_ok=True)
    client = make_client()
    base = os.path.splitext(os.path.basename(video_path))[0]
    work_dir = tempfile.mkdtemp(prefix=f"{base}_", dir=output_folder)
    final_chunks = []

    mode_lower = (mode or "").strip().lower()
    accurate = mode_lower.startswith("accurate")

    try:
        chunk_sources = split_video_into_chunks(video_path, work_dir)
        for idx, chunk_video in enumerate(chunk_sources):
            chunk_base = os.path.splitext(os.path.basename(chunk_video))[0]
            raw_mp3 = os.path.join(work_dir, f"{chunk_base}_raw.mp3")
            chunk_output = os.path.join(work_dir, f"{chunk_base}_out.mp4")

            extract_mp3(chunk_video, raw_mp3)

            if accurate:
                tts_mp3, srt_path = build_aligned_tts_for_chunk(
                    client, raw_mp3, work_dir, voice
                )
            else:
                tts_mp3, srt_path = build_quick_tts_for_chunk(
                    client, raw_mp3, work_dir, voice
                )

            replace_audio_with_tts(chunk_video, tts_mp3, chunk_output, srt_path, burn_subtitles)

            final_chunks.append(chunk_output)

            log(
                f"Chunk {idx+1}/{len(chunk_sources)} processed "
                f"(mode={'accurate' if accurate else 'quick'}, voice={voice})."
            )

        final_output = os.path.join(output_folder, f"{base}_dubbed.mp4")
        concatenate_chunks(final_chunks, final_output, work_dir)
        log(f"Finished video: {final_output}")
        return final_output

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------- MULTI-FILE WORKER ----------

def process_files_worker(
    file_list,
    output_folder,
    voice: str,
    mode: str,
    progress_callback=None,
    stop_event=None,
    burn_subtitles=True,
):
    os.makedirs(output_folder, exist_ok=True)
    total = len(file_list)
    log(f"Starting processing {total} files with {MAX_WORKERS} workers")
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(process_single_file, f, output_folder, voice, mode, burn_subtitles): f
            for f in file_list
        }

        completed = 0
        for future in as_completed(future_to_file):
            video = future_to_file[future]
            try:
                res = future.result()
                results.append((video, True, res))
            except Exception as e:
                log(f"Error processing {video}: {e}")
                results.append((video, False, str(e)))

            completed += 1
            if progress_callback:
                progress_callback(completed, total)

            if stop_event and stop_event.is_set():
                log("Stop event triggered – no new tasks will be started.")
                break

    return results


# ---------- GUI CLASS ----------


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Video → Transcribe → Translate → TTS")
        self.file_list = []
        self.stop_event = threading.Event()
        self.worker_thread = None

        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="Input folder:").grid(row=0, column=0, sticky="w")
        self.input_var = tk.StringVar(value=INPUT_DIR_DEFAULT)
        self.input_entry = ttk.Entry(top, textvariable=self.input_var, width=40)
        self.input_entry.grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(top, text="Browse", command=self.browse_input).grid(
            row=0, column=2, padx=4
        )

        ttk.Label(top, text="Output folder:").grid(row=1, column=0, sticky="w")
        self.output_var = tk.StringVar(value=OUTPUT_DIR_DEFAULT)
        self.output_entry = ttk.Entry(top, textvariable=self.output_var, width=40)
        self.output_entry.grid(row=1, column=1, sticky="we", padx=4)
        ttk.Button(top, text="Browse", command=self.browse_output).grid(
            row=1, column=2, padx=4
        )

        ttk.Button(top, text="Add files...", command=self.add_files_dialog).grid(
            row=2, column=0, pady=6
        )
        ttk.Button(top, text="Scan input folder", command=self.scan_input_folder).grid(
            row=2, column=1, pady=6
        )
        ttk.Button(top, text="Clear list", command=self.clear_list).grid(
            row=2, column=2, pady=6
        )

        # Voice selection
        ttk.Label(top, text="TTS voice:").grid(row=3, column=0, sticky="w")
        self.voice_var = tk.StringVar(value="ash")
        self.voice_combo = ttk.Combobox(
            top,
            textvariable=self.voice_var,
            values=["alloy", "verse", "aria", "nova", "shimmer", "ash"],
            state="readonly",
            width=12,
        )
        self.voice_combo.grid(row=3, column=1, sticky="w", padx=4)

        # Mode selection
        ttk.Label(top, text="Mode:").grid(row=4, column=0, sticky="w")
        self.mode_var = tk.StringVar(value="Accurate")
        self.mode_combo = ttk.Combobox(
            top,
            textvariable=self.mode_var,
            values=["Accurate", "Quick"],
            state="readonly",
            width=12,
        )
        self.mode_combo.grid(row=4, column=1, sticky="w", padx=4)

        self.subtitles_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            top,
            text="Burn-in subtitles",
            variable=self.subtitles_var,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))

        mid = ttk.Frame(root, padding=8)
        mid.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(mid, selectmode=tk.EXTENDED, height=10)
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(mid, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)

        bottom = ttk.Frame(root, padding=8)
        bottom.pack(fill="x")
        self.progress = ttk.Progressbar(bottom, length=400, mode="determinate")
        self.progress.pack(fill="x", pady=4)

        controls = ttk.Frame(bottom)
        controls.pack(fill="x")
        self.start_btn = ttk.Button(controls, text="Start", command=self.start_processing)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(
            controls, text="Stop", command=self.request_stop, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=4)
        ttk.Button(
            controls, text="Open output folder", command=self.open_output_folder
        ).pack(side="right", padx=4)

        logframe = ttk.Frame(root, padding=8)
        logframe.pack(fill="both", expand=True)
        ttk.Label(logframe, text="Logs:").pack(anchor="w")
        self.log_text = tk.Text(logframe, height=10, state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=True)

        self.root.after(200, self.process_log_queue)
        os.makedirs(self.input_var.get(), exist_ok=True)
        os.makedirs(self.output_var.get(), exist_ok=True)

    # GUI callbacks
    def browse_input(self):
        d = filedialog.askdirectory(initialdir=".", title="Select input folder")
        if d:
            self.input_var.set(d)
            os.makedirs(d, exist_ok=True)

    def browse_output(self):
        d = filedialog.askdirectory(initialdir=".", title="Select output folder")
        if d:
            self.output_var.set(d)
            os.makedirs(d, exist_ok=True)

    def add_files_dialog(self):
        files = filedialog.askopenfilenames(
            title="Select mp4 files",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
        )
        for f in files:
            abs_path = os.path.abspath(f)
            if abs_path not in self.file_list:
                self.file_list.append(abs_path)
                self.listbox.insert("end", os.path.basename(f))

    def scan_input_folder(self):
        folder = self.input_var.get()
        if not os.path.isdir(folder):
            messagebox.showwarning("Folder not found", "Input folder does not exist.")
            return
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".mp4")
        ]
        added = 0
        for f in files:
            abs_path = os.path.abspath(f)
            if abs_path not in self.file_list:
                self.file_list.append(abs_path)
                self.listbox.insert("end", os.path.basename(f))
                added += 1
        log(f"Scanned: added {added} videos.")

    def clear_list(self):
        self.file_list.clear()
        self.listbox.delete(0, "end")

    def open_output_folder(self):
        out = os.path.abspath(self.output_var.get())
        if not os.path.isdir(out):
            messagebox.showwarning("No output folder", "Output folder does not exist.")
            return
        if os.name == "nt":
            os.startfile(out)
        else:
            subprocess.Popen(["xdg-open", out])

    def process_log_queue(self):
        while not log_queue.empty():
            line = log_queue.get()
            self.log_text.config(state="normal")
            self.log_text.insert("end", line + "\n")
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        self.root.after(200, self.process_log_queue)

    def start_processing(self):
        try:
            make_client()
        except RuntimeError as e:
            messagebox.showerror("API key / OpenAI error", str(e))
            return

        if not self.file_list:
            self.scan_input_folder()
            if not self.file_list:
                messagebox.showinfo("No files", "No MP4 files found.")
                return

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress["value"] = 0
        self.progress["maximum"] = len(self.file_list)
        self.stop_event.clear()

        files_to_process = list(self.file_list)
        out_folder = self.output_var.get()

        voice = (self.voice_var.get() or "ash").strip()
        if voice not in ["alloy", "verse", "aria", "nova", "shimmer", "ash"]:
            voice = "ash"

        mode = (self.mode_var.get() or "Accurate").strip()
        burn_subtitles = bool(self.subtitles_var.get())

        def progress_cb(done, total):
            self.root.after(0, lambda d=done: self.progress_step(d))

        def worker():
            try:
                process_files_worker(
                    files_to_process,
                    out_folder,
                    voice,
                    mode,
                    progress_callback=progress_cb,
                    stop_event=self.stop_event,
                    burn_subtitles=burn_subtitles,
                )
                log("All tasks finished.")
            except Exception as e:
                log(f"Worker exception: {e}")
            finally:
                self.root.after(0, self.worker_done)

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def progress_step(self, done_count):
        self.progress["value"] = done_count

    def worker_done(self):
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        messagebox.showinfo("Done", "Processing finished.")

    def request_stop(self):
        if messagebox.askyesno(
            "Stop",
            "Остановить запуск? Текущие задачи продолжат выполнение.",
        ):
            self.stop_event.set()
            log("Stop requested by user.")
            self.stop_btn.config(state="disabled")


# ---------- MAIN ----------

def main():
    # FFmpeg availability check
    if not os.path.isfile(FFMPEG_BIN):
        if messagebox:
            msg = (
                f"FFmpeg was not found at:\n{FFMPEG_BIN}\n\n"
                "Please install FFmpeg and update the FFMPEG_PATH environment "
                "variable or config."
            )
            messagebox.showerror("FFmpeg not found", msg)
        else:
            print(f"FFmpeg not found: {FFMPEG_BIN}")
        return

    if tk is None:
        # In environments without tkinter (e.g. headless/sandbox), explain clearly
        print(
            "Error: tkinter GUI library is not available in this Python environment.\n"
            "Please run this program with a standard desktop Python installation "
            "that includes tkinter (on Windows use the official python.org installer)."
        )
        return

    root = tk.Tk()
    # Apply scaling and default font for better look on high DPI
    apply_tk_scaling(root)
    apply_default_font(root, size=11)

    app = App(root)
    # Larger default window for modern screens
    root.geometry("1200x900")
    root.mainloop()


if __name__ == "__main__":
    main()
