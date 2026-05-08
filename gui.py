import sys
import queue
import threading
import time
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from audio_processor import ProcessingParams, process_audio, load_audio_info, load_waveform

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_DIVIDER_COLOR = "#3a3a3a"


def _resource(name: str) -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / name
    return Path(__file__).parent / name


def _divider(parent) -> ctk.CTkFrame:
    return ctk.CTkFrame(parent, height=1, fg_color=_DIVIDER_COLOR, corner_radius=0)


class VoiceCleanerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Voice Cleaner")
        self.geometry("1280x780")
        self.minsize(960, 640)

        icon = _resource("icon.ico")
        if icon.exists():
            self.iconbitmap(str(icon))

        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._stop_flag = [False]
        self._input_file: str | None = None
        self._output_dir: str | None = None
        self._audio_in: tuple | None = None    # (audio_array, sr)
        self._audio_out: tuple | None = None   # (audio_array, sr)
        self._playing: str | None = None       # "in" | "out" | None
        self._cursor_in = None                 # matplotlib Line2D
        self._cursor_out = None
        self._play_start_time: float | None = None
        self._play_start_pos: float = 0.0      # posição normalizada 0–1
        self._play_total_samples: int = 0
        self._play_sr: int = 0

        self._build_ui()
        self.after(80, self._poll_queue)

    # ── Layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()
        self._build_bottom_bar()

    def _build_sidebar(self):
        # Frame externo: largura fixa 300px
        sb_outer = ctk.CTkFrame(self, width=300, corner_radius=0)
        sb_outer.grid(row=0, column=0, sticky="nsew")
        sb_outer.grid_propagate(False)
        sb_outer.grid_columnconfigure(0, weight=1)
        sb_outer.grid_rowconfigure(0, weight=1)

        # Frame interno scrollável: contém todos os controles
        sb = ctk.CTkScrollableFrame(sb_outer, fg_color="transparent", corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_columnconfigure(0, weight=1)
        r = 0

        ctk.CTkLabel(sb, text="🎙️ Voice Cleaner",
                     font=ctk.CTkFont(size=20, weight="bold")).grid(
            row=r, column=0, padx=16, pady=(20, 2), sticky="w"); r += 1
        ctk.CTkLabel(sb, text="Tratamento de voz com IA",
                     font=ctk.CTkFont(size=12), text_color="gray").grid(
            row=r, column=0, padx=16, pady=(0, 10), sticky="w"); r += 1
        _divider(sb).grid(row=r, column=0, padx=16, sticky="ew"); r += 1

        # ── Arquivo de entrada ────────────────────────────────────────────
        ctk.CTkLabel(sb, text="ARQUIVO DE ENTRADA",
                     font=ctk.CTkFont(size=11), text_color="gray").grid(
            row=r, column=0, padx=16, pady=(8, 0), sticky="w"); r += 1
        ctk.CTkButton(sb, text="Selecionar arquivo…",
                      command=self._select_file).grid(
            row=r, column=0, padx=16, pady=(6, 2), sticky="ew"); r += 1
        self._file_label = ctk.CTkLabel(sb, text="Nenhum arquivo selecionado",
                                        font=ctk.CTkFont(size=11), text_color="gray",
                                        wraplength=264, justify="left")
        self._file_label.grid(row=r, column=0, padx=16, pady=(0, 2), sticky="w"); r += 1
        self._info_label = ctk.CTkLabel(sb, text="",
                                         font=ctk.CTkFont(size=11), text_color="#888",
                                         wraplength=264, justify="left")
        self._info_label.grid(row=r, column=0, padx=16, pady=(0, 4), sticky="w"); r += 1
        _divider(sb).grid(row=r, column=0, padx=16, sticky="ew"); r += 1

        # ── Redução de ruído ──────────────────────────────────────────────
        noise_title_row = ctk.CTkFrame(sb, fg_color="transparent")
        noise_title_row.grid(row=r, column=0, padx=16, pady=(8, 0), sticky="ew")
        noise_title_row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(noise_title_row, text="REDUÇÃO DE RUÍDO",
                     font=ctk.CTkFont(size=11), text_color="gray").grid(
            row=0, column=0, sticky="w")
        self._noise_enabled_var = ctk.BooleanVar(value=True)
        self._noise_switch = ctk.CTkSwitch(
            noise_title_row, text="", width=40,
            variable=self._noise_enabled_var,
            command=self._on_noise_enabled_toggle)
        self._noise_switch.grid(row=0, column=1, sticky="e"); r += 1

        row_hdr = ctk.CTkFrame(sb, fg_color="transparent")
        row_hdr.grid(row=r, column=0, padx=16, pady=(6, 0), sticky="ew")
        row_hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(row_hdr, text="Intensidade",
                     font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w")
        self._noise_val = ctk.CTkLabel(row_hdr, text="75%",
                                       font=ctk.CTkFont(size=12, weight="bold"))
        self._noise_val.grid(row=0, column=1, sticky="e"); r += 1

        self._noise_slider = ctk.CTkSlider(sb, from_=0, to=100, number_of_steps=100,
                                           command=self._on_noise_change)
        self._noise_slider.set(75)
        self._noise_slider.grid(row=r, column=0, padx=16, pady=(4, 8), sticky="ew"); r += 1

        ctk.CTkLabel(sb, text="Tipo de ruído de fundo",
                     font=ctk.CTkFont(size=12)).grid(
            row=r, column=0, padx=16, pady=(0, 4), sticky="w"); r += 1
        self._noise_mode = ctk.StringVar(value="adaptive")
        self._noise_radios = []
        for val, label in [
            ("adaptive",    "Ambiente variável  (vento, trânsito)"),
            ("stationary",  "Constante  (AC, ventilador, chiado)"),
            ("percussive",  "Batidas / percussão  (drums, música)"),
        ]:
            rb = ctk.CTkRadioButton(sb, text=label, font=ctk.CTkFont(size=11),
                                    variable=self._noise_mode, value=val)
            rb.grid(row=r, column=0, padx=20, pady=(0, 4), sticky="w")
            self._noise_radios.append(rb); r += 1
        _divider(sb).grid(row=r, column=0, padx=16, sticky="ew"); r += 1

        # ── Melhoria de voz ───────────────────────────────────────────────
        ctk.CTkLabel(sb, text="MELHORIA DE VOZ",
                     font=ctk.CTkFont(size=11), text_color="gray").grid(
            row=r, column=0, padx=16, pady=(8, 0), sticky="w"); r += 1

        boost_hdr = ctk.CTkFrame(sb, fg_color="transparent")
        boost_hdr.grid(row=r, column=0, padx=16, pady=(6, 0), sticky="ew")
        boost_hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(boost_hdr, text="Presença vocal (+3 kHz)",
                     font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w")
        self._boost_val = ctk.CTkLabel(boost_hdr, text="0.0 dB",
                                       font=ctk.CTkFont(size=12, weight="bold"))
        self._boost_val.grid(row=0, column=1, sticky="e"); r += 1

        self._boost_slider = ctk.CTkSlider(sb, from_=0, to=12, number_of_steps=24,
                                           command=self._on_boost_change)
        self._boost_slider.set(0)
        self._boost_slider.grid(row=r, column=0, padx=16, pady=(4, 8), sticky="ew"); r += 1

        self._normalize_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(sb, text="Normalizar volume de saída",
                        variable=self._normalize_var).grid(
            row=r, column=0, padx=16, pady=(2, 2), sticky="w"); r += 1
        self._trim_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(sb, text="Cortar silêncio inicial/final",
                        variable=self._trim_var).grid(
            row=r, column=0, padx=16, pady=(0, 8), sticky="w"); r += 1
        _divider(sb).grid(row=r, column=0, padx=16, sticky="ew"); r += 1

        # ── Compressão de dinâmica ────────────────────────────────────────
        comp_title_row = ctk.CTkFrame(sb, fg_color="transparent")
        comp_title_row.grid(row=r, column=0, padx=16, pady=(8, 0), sticky="ew")
        comp_title_row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(comp_title_row, text="COMPRESSÃO DE DINÂMICA",
                     font=ctk.CTkFont(size=11), text_color="gray").grid(
            row=0, column=0, sticky="w")
        self._compress_var = ctk.BooleanVar(value=False)
        self._compress_switch = ctk.CTkSwitch(
            comp_title_row, text="", width=40,
            variable=self._compress_var,
            command=self._on_compress_toggle)
        self._compress_switch.grid(row=0, column=1, sticky="e"); r += 1

        ctk.CTkLabel(sb, text="Nivela picos altos com partes baixas da voz",
                     font=ctk.CTkFont(size=10), text_color="#777",
                     wraplength=264).grid(
            row=r, column=0, padx=16, pady=(4, 6), sticky="w"); r += 1

        thr_hdr = ctk.CTkFrame(sb, fg_color="transparent")
        thr_hdr.grid(row=r, column=0, padx=16, pady=(0, 0), sticky="ew")
        thr_hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(thr_hdr, text="Limiar",
                     font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w")
        self._thr_val = ctk.CTkLabel(thr_hdr, text="-20 dB",
                                     font=ctk.CTkFont(size=12, weight="bold"))
        self._thr_val.grid(row=0, column=1, sticky="e"); r += 1

        self._thr_slider = ctk.CTkSlider(sb, from_=-40, to=0, number_of_steps=40,
                                         command=self._on_thr_change, state="disabled")
        self._thr_slider.set(-20)
        self._thr_slider.grid(row=r, column=0, padx=16, pady=(4, 8), sticky="ew"); r += 1

        ratio_hdr = ctk.CTkFrame(sb, fg_color="transparent")
        ratio_hdr.grid(row=r, column=0, padx=16, pady=(0, 0), sticky="ew")
        ratio_hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(ratio_hdr, text="Proporção",
                     font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w")
        self._ratio_val = ctk.CTkLabel(ratio_hdr, text="4 : 1",
                                       font=ctk.CTkFont(size=12, weight="bold"))
        self._ratio_val.grid(row=0, column=1, sticky="e"); r += 1

        self._ratio_slider = ctk.CTkSlider(sb, from_=1, to=20, number_of_steps=19,
                                           command=self._on_ratio_change, state="disabled")
        self._ratio_slider.set(4)
        self._ratio_slider.grid(row=r, column=0, padx=16, pady=(4, 8), sticky="ew"); r += 1
        _divider(sb).grid(row=r, column=0, padx=16, sticky="ew"); r += 1

        # ── Pasta de saída ────────────────────────────────────────────────
        ctk.CTkLabel(sb, text="PASTA DE SAÍDA",
                     font=ctk.CTkFont(size=11), text_color="gray").grid(
            row=r, column=0, padx=16, pady=(8, 0), sticky="w"); r += 1
        ctk.CTkButton(sb, text="Selecionar pasta…",
                      command=self._select_output).grid(
            row=r, column=0, padx=16, pady=(6, 2), sticky="ew"); r += 1
        self._output_label = ctk.CTkLabel(sb, text="Mesma pasta do arquivo",
                                           font=ctk.CTkFont(size=11), text_color="gray",
                                           wraplength=264, justify="left")
        self._output_label.grid(row=r, column=0, padx=16, pady=(0, 8), sticky="w")

    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=(8, 8), pady=(8, 0))
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=3)
        main.grid_rowconfigure(1, weight=1)

        # ── Painel de forma de onda ───────────────────────────────────────
        wave_frame = ctk.CTkFrame(main)
        wave_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        wave_frame.grid_columnconfigure(0, weight=1)
        wave_frame.grid_rowconfigure(1, weight=1)

        # Header: título + botões de play
        hdr = ctk.CTkFrame(wave_frame, fg_color="transparent")
        hdr.grid(row=0, column=0, padx=8, pady=(10, 4), sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(hdr, text="Comparação de Forma de Onda",
                     font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, sticky="w")

        self._play_in_btn = ctk.CTkButton(
            hdr, text="▶ Entrada", width=90, height=28,
            font=ctk.CTkFont(size=12),
            fg_color="#1a6aa8", hover_color="#1558885",
            state="disabled", command=lambda: self._play_audio("in"))
        self._play_in_btn.grid(row=0, column=1, padx=(4, 4))

        self._play_out_btn = ctk.CTkButton(
            hdr, text="▶ Saída", width=90, height=28,
            font=ctk.CTkFont(size=12),
            fg_color="#1a7a44", hover_color="#145e34",
            state="disabled", command=lambda: self._play_audio("out"))
        self._play_out_btn.grid(row=0, column=2, padx=(0, 4))

        self._stop_audio_btn = ctk.CTkButton(
            hdr, text="⏹", width=36, height=28,
            font=ctk.CTkFont(size=12),
            fg_color="#444", hover_color="#555",
            command=self._stop_audio)
        self._stop_audio_btn.grid(row=0, column=3, padx=(0, 0))

        self._fig = Figure(figsize=(7, 4), dpi=96, facecolor="#2b2b2b")
        self._ax_in = self._fig.add_subplot(211)
        self._ax_out = self._fig.add_subplot(212)

        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=wave_frame)
        self._mpl_canvas.get_tk_widget().grid(
            row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self._mpl_canvas.mpl_connect("button_press_event", self._on_waveform_click)
        self._reset_axes()

        # ── Painel de log ─────────────────────────────────────────────────
        log_frame = ctk.CTkFrame(main)
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(log_frame, text="Registro",
                     font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, padx=16, pady=(10, 4), sticky="w")

        self._log = ctk.CTkTextbox(
            log_frame, height=90,
            font=ctk.CTkFont(family="Consolas", size=11))
        self._log.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

    def _build_bottom_bar(self):
        bar = ctk.CTkFrame(self, height=56, corner_radius=0)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(1, weight=1)

        self._stop_btn = ctk.CTkButton(
            bar, text="⏹ Parar", width=90,
            fg_color="#c0392b", hover_color="#922b21",
            command=self._stop_processing, state="disabled")
        self._stop_btn.grid(row=0, column=0, padx=(12, 8), pady=8)

        self._progress = ctk.CTkProgressBar(bar, height=8)
        self._progress.set(0)
        self._progress.grid(row=0, column=1, padx=8, pady=8, sticky="ew")

        self._status = ctk.CTkLabel(bar, text="Aguardando arquivo…",
                                    font=ctk.CTkFont(size=12), width=220,
                                    anchor="w")
        self._status.grid(row=0, column=2, padx=8, pady=8)

        self._start_btn = ctk.CTkButton(
            bar, text="▶  Processar", width=140, height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._start_processing)
        self._start_btn.grid(row=0, column=3, padx=(8, 12), pady=8)

    # ── Helpers de plot ────────────────────────────────────────────────────

    def _reset_axes(self):
        self._cursor_in = None
        self._cursor_out = None
        for ax, title in [(self._ax_in, "Entrada"), (self._ax_out, "Saída (processado)")]:
            ax.clear()
            ax.set_facecolor("#1a1a1a")
            ax.set_title(title, color="#888", fontsize=8, pad=2)
            ax.set_yticks([])
            ax.tick_params(axis="x", colors="#555", labelsize=7)
            for sp in ax.spines.values():
                sp.set_color("#3a3a3a")
            ax.text(0.5, 0.5, "— sem dados —", transform=ax.transAxes,
                    ha="center", va="center", color="#444", fontsize=11)
        self._fig.tight_layout(pad=1.5)
        self._mpl_canvas.draw()

    def _draw_waveform(self, ax, envelope: np.ndarray, color: str, title: str):
        ax.clear()
        ax.set_facecolor("#1a1a1a")
        ax.set_title(title, color="#999", fontsize=8, pad=2)
        ax.set_yticks([])
        ax.tick_params(axis="x", colors="#555", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#3a3a3a")
        x = np.linspace(0, 1, len(envelope))
        ax.fill_between(x, envelope, -envelope, color=color, alpha=0.75, linewidth=0)
        ax.set_xlim(0, 1)
        lim = max(np.max(envelope) * 1.05, 0.01)
        ax.set_ylim(-lim, lim)
        # Cursor de posição de reprodução (inicialmente fora da área)
        cursor = ax.axvline(x=-1, color="#ffffff", alpha=0.85, linewidth=1.5, linestyle="--", zorder=5)
        if ax is self._ax_in:
            self._cursor_in = cursor
        else:
            self._cursor_out = cursor
        self._fig.tight_layout(pad=1.5)
        self._mpl_canvas.draw()

    def _update_cursor(self, which: str, pos: float):
        cursor = self._cursor_in if which == "in" else self._cursor_out
        if cursor is not None:
            cursor.set_xdata([pos, pos])
            cursor.set_visible(True)
            self._mpl_canvas.draw_idle()

    # ── Playback ───────────────────────────────────────────────────────────

    def _on_waveform_click(self, event):
        if event.inaxes is None or event.xdata is None:
            return
        which = "in" if event.inaxes is self._ax_in else "out"
        data = self._audio_in if which == "in" else self._audio_out
        if data is None:
            return
        pos = max(0.0, min(1.0, event.xdata))
        audio, sr = data
        start_sample = int(pos * len(audio))
        self._play_audio(which, start_sample=start_sample)

    def _play_audio(self, which: str, start_sample: int = 0):
        import sounddevice as sd
        sd.stop()
        # Toggle off se já tocando o mesmo e sem seek
        if self._playing == which and start_sample == 0:
            self._playing = None
            self._play_start_time = None
            self._update_play_buttons()
            return
        data = self._audio_in if which == "in" else self._audio_out
        if data is None:
            return
        audio, sr = data
        start_sample = max(0, min(start_sample, len(audio) - 1))
        self._playing = which
        self._play_start_time = time.time()
        self._play_start_pos = start_sample / len(audio)
        self._play_total_samples = len(audio)
        self._play_sr = sr
        self._update_play_buttons()
        self._update_cursor(which, self._play_start_pos)
        threading.Thread(target=self._playback_worker,
                         args=(audio[start_sample:], sr, which), daemon=True).start()

    def _playback_worker(self, audio: np.ndarray, sr: int, which: str):
        import sounddevice as sd
        try:
            sd.play(audio, sr)
            sd.wait()
        finally:
            self._queue.put(("play_done", which))

    def _stop_audio(self):
        import sounddevice as sd
        sd.stop()
        self._playing = None
        self._update_play_buttons()

    def _update_play_buttons(self):
        self._play_in_btn.configure(
            text="⏸ Entrada" if self._playing == "in" else "▶ Entrada",
            state="normal" if self._audio_in else "disabled")
        self._play_out_btn.configure(
            text="⏸ Saída" if self._playing == "out" else "▶ Saída",
            state="normal" if self._audio_out else "disabled")

    # ── Callbacks de controles ─────────────────────────────────────────────

    def _on_noise_change(self, val):
        self._noise_val.configure(text=f"{int(val)}%")

    def _on_noise_enabled_toggle(self):
        state = "normal" if self._noise_enabled_var.get() else "disabled"
        self._noise_slider.configure(state=state)
        for rb in self._noise_radios:
            rb.configure(state=state)

    def _on_boost_change(self, val):
        self._boost_val.configure(text=f"{val:.1f} dB")

    def _on_compress_toggle(self):
        state = "normal" if self._compress_var.get() else "disabled"
        self._thr_slider.configure(state=state)
        self._ratio_slider.configure(state=state)

    def _on_thr_change(self, val):
        self._thr_val.configure(text=f"{int(val)} dB")

    def _on_ratio_change(self, val):
        self._ratio_val.configure(text=f"{int(val)} : 1")

    def _select_file(self):
        path = filedialog.askopenfilename(
            title="Selecionar arquivo de áudio",
            filetypes=[
                ("Áudio", "*.wav *.flac *.ogg *.mp3 *.m4a *.aac *.opus"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if not path:
            return
        self._stop_audio()
        self._audio_in = None
        self._audio_out = None
        self._input_file = path
        self._file_label.configure(text=Path(path).name)
        self._reset_axes()
        self._update_play_buttons()
        self._queue.put(("log", f"📂 Arquivo: {Path(path).name}"))
        threading.Thread(target=self._load_info_and_waveform,
                         args=(path,), daemon=True).start()

    def _load_info_and_waveform(self, path: str):
        try:
            info = load_audio_info(path)
            mins = int(info["duration"] // 60)
            secs = int(info["duration"] % 60)
            self._queue.put(("file_info",
                             f"{info['channels']}ch · {info['sample_rate']} Hz · "
                             f"{mins}m{secs:02d}s · {info['size_mb']:.1f} MB"))
        except Exception as e:
            self._queue.put(("file_info", f"Erro ao ler info: {e}"))
        try:
            envelope, audio, sr = load_waveform(path)
            self._queue.put(("wave_in", envelope, audio, sr))
        except Exception as e:
            self._queue.put(("log", f"⚠ Forma de onda: {e}"))

    def _select_output(self):
        path = filedialog.askdirectory(title="Selecionar pasta de saída")
        if path:
            self._output_dir = path
            self._output_label.configure(text=path)

    # ── Processamento ──────────────────────────────────────────────────────

    def _start_processing(self):
        if not self._input_file:
            self._log_msg("⚠ Selecione um arquivo de entrada primeiro.")
            return
        if self._running:
            return

        self._stop_audio()
        inp = Path(self._input_file)
        out_dir = Path(self._output_dir) if self._output_dir else inp.parent
        out_path = str(out_dir / f"{inp.stem}_clean.wav")

        params = ProcessingParams(
            noise_enabled=self._noise_enabled_var.get(),
            noise_intensity=self._noise_slider.get() / 100,
            noise_type=self._noise_mode.get(),
            voice_boost_db=self._boost_slider.get(),
            normalize=self._normalize_var.get(),
            trim_silence=self._trim_var.get(),
            compress=self._compress_var.get(),
            compress_threshold_db=self._thr_slider.get(),
            compress_ratio=self._ratio_slider.get(),
        )

        self._running = True
        self._stop_flag[0] = False
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._progress.set(0)

        self._log_msg("▶ Iniciando processamento…")
        tipo = {"adaptive": "ambiente variável", "stationary": "constante", "percussive": "batidas/percussão"}
        ruido_str = f"Ruído: {int(params.noise_intensity * 100)}% ({tipo.get(params.noise_type)})" if params.noise_enabled else "Ruído: desativado"
        comp_str = f"  Compressão: {int(params.compress_threshold_db)}dB {int(params.compress_ratio)}:1" if params.compress else ""
        self._log_msg(f"  {ruido_str}  Boost: {params.voice_boost_db:.1f} dB{comp_str}")
        self._log_msg(f"  Saída: {out_path}")

        threading.Thread(
            target=self._worker,
            args=(self._input_file, out_path, params),
            daemon=True,
        ).start()

    def _stop_processing(self):
        if self._running:
            self._stop_flag[0] = True
            self._stop_btn.configure(state="disabled")
            self._log_msg("⏹ Solicitando cancelamento…")

    def _worker(self, inp: str, out: str, params: ProcessingParams):
        t0 = time.time()
        try:
            result = process_audio(
                inp, out, params,
                progress_cb=lambda pct, msg: self._queue.put(("progress", pct, msg)),
                stop_flag=self._stop_flag,
            )
            self._queue.put(("done", result, out, time.time() - t0))
        except InterruptedError:
            self._queue.put(("cancelled",))
        except Exception as exc:
            import traceback
            self._queue.put(("error", f"{exc}\n{traceback.format_exc()}"))

    # ── Polling da fila ────────────────────────────────────────────────────

    def _poll_queue(self):
        try:
            while True:
                msg = self._queue.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, pct, text = msg
                    self._progress.set(pct)
                    self._status.configure(text=text)
                    self._log_msg(f"  {text}")

                elif kind == "done":
                    _, result, out_path, elapsed = msg
                    self._running = False
                    self._progress.set(1.0)
                    self._start_btn.configure(state="normal")
                    self._stop_btn.configure(state="disabled")
                    mins = int(result["duration"] // 60)
                    secs = int(result["duration"] % 60)
                    self._status.configure(text=f"✅ Concluído em {elapsed:.1f}s")
                    self._log_msg(
                        f"✅ Concluído em {elapsed:.1f}s  |  "
                        f"{mins}m{secs:02d}s  |  {result['channels']}ch · {result['sample_rate']} Hz")
                    self._log_msg(f"   Salvo: {out_path}")
                    threading.Thread(target=self._load_output_waveform,
                                     args=(out_path,), daemon=True).start()

                elif kind == "cancelled":
                    self._running = False
                    self._start_btn.configure(state="normal")
                    self._stop_btn.configure(state="disabled")
                    self._status.configure(text="Cancelado")
                    self._log_msg("⏹ Processamento cancelado.")

                elif kind == "error":
                    _, err = msg
                    self._running = False
                    self._start_btn.configure(state="normal")
                    self._stop_btn.configure(state="disabled")
                    self._status.configure(text="❌ Erro!")
                    self._log_msg(f"❌ Erro:\n{err}")

                elif kind == "log":
                    _, text = msg
                    self._log_msg(text)

                elif kind == "file_info":
                    _, text = msg
                    self._info_label.configure(text=text)
                    self._status.configure(text="Pronto para processar")

                elif kind == "wave_in":
                    _, env, audio, sr = msg
                    self._audio_in = (audio, sr)
                    self._draw_waveform(self._ax_in, env, "#4da6ff", "Entrada")
                    self._update_play_buttons()

                elif kind == "wave_out":
                    _, env, audio, sr = msg
                    self._audio_out = (audio, sr)
                    self._draw_waveform(self._ax_out, env, "#4dff91", "Saída (processado)")
                    self._update_play_buttons()

                elif kind == "play_done":
                    _, which = msg
                    if self._playing == which:
                        self._playing = None
                        self._play_start_time = None
                        self._update_play_buttons()
                        # Esconde cursor ao terminar
                        cursor = self._cursor_in if which == "in" else self._cursor_out
                        if cursor is not None:
                            cursor.set_visible(False)
                            self._mpl_canvas.draw_idle()

        except queue.Empty:
            pass

        # Atualiza cursor de posição durante reprodução (a cada 80ms)
        if self._playing and self._play_start_time is not None and self._play_total_samples > 0:
            elapsed = time.time() - self._play_start_time
            pos = self._play_start_pos + elapsed * self._play_sr / self._play_total_samples
            if pos <= 1.0:
                self._update_cursor(self._playing, pos)

        self.after(80, self._poll_queue)

    def _load_output_waveform(self, path: str):
        try:
            envelope, audio, sr = load_waveform(path)
            self._queue.put(("wave_out", envelope, audio, sr))
        except Exception as e:
            self._queue.put(("log", f"⚠ Forma de onda saída: {e}"))

    def _log_msg(self, msg: str):
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")


if __name__ == "__main__":
    try:
        app = VoiceCleanerApp()
        app.mainloop()
    except Exception as exc:
        import traceback
        import tkinter as tk
        from tkinter import messagebox
        _root = tk.Tk()
        _root.withdraw()
        messagebox.showerror("Erro ao iniciar", f"{exc}\n\n{traceback.format_exc()}")
        _root.destroy()
