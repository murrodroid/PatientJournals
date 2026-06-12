from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from patientjournals.app.catalog import list_google_model_options, list_schema_options
from patientjournals.app.dashboard import (
    dashboard_summary_json,
    find_dataset_files,
    latest_dataset_path,
    summarize_dashboard,
)
from patientjournals.app.datasets import (
    inspect_local_dataset,
    list_cloud_dataset_choices,
)
from patientjournals.app.jobs import (
    JobRegistry,
    aggregate_batch_state,
    batch_run_provider,
    build_submit_command,
    build_validation_command,
    cancel_batch_run,
    find_dataset_near,
    recover_dataset_gaps,
    list_batch_chunks,
    list_batch_chunks_with_state,
    list_submit_jobs,
    local_image_names,
    poll_local_batch_states,
    read_dataset_preview,
    read_recorded_results,
    read_run_error,
    resubmit_failed_requests,
    run_batch_draft_direct,
    run_batch_rerun_direct,
    run_local_draft_direct,
    run_retrieve_direct,
    start_command,
)
from patientjournals.app.models import AppSettings, SubmitJobDraft, app_settings_path
from patientjournals.app.settings_store import load_app_settings, save_app_settings


BG = "#FFFFFF"
ACCENT = "#00B2CA"
INK = "#1E1E24"
MUTED_BG = "#F4F8F9"
SOFT_BORDER = "#D7E5E8"


def _truncate_cell(value: object, limit: int = 60) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _open_in_file_browser(path: str | Path) -> None:
    target = str(path)
    if sys.platform == "darwin":
        subprocess.Popen(["open", target])  # noqa: S603, S607
    elif sys.platform.startswith("win"):
        subprocess.Popen(["explorer", target])  # noqa: S603, S607
    else:
        subprocess.Popen(["xdg-open", target])  # noqa: S603, S607


class PatientJournalsApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("PatientJournals")
        self.root.geometry("1180x760")
        self.root.minsize(980, 660)
        self.root.configure(bg=BG)
        self._setup_theme()

        self.settings_path = app_settings_path()
        self.settings = load_app_settings(self.settings_path)
        self.registry = JobRegistry()
        self.schema_options = list_schema_options()
        self.model_options = list_google_model_options()
        self.nav_items: dict[str, tk.Label] = {}
        self._live_batch_status: dict[str, str] = {}
        self._jobs_refresh_after_id = None
        self._jobs_repaint = None
        self._preset_validation_dataset = ""

        self.sidebar = tk.Frame(self.root, bg=INK, padx=16, pady=18, width=210)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self.content = ttk.Frame(self.root, padding=(28, 24), style="App.TFrame")
        self.content.pack(side="right", fill="both", expand=True)

        self._build_sidebar()
        if self.settings_path.exists():
            self._navigate("submit", self.show_submit)
        else:
            self._navigate("settings", self.show_settings)

        self._start_launch_batch_poll()

    def _start_launch_batch_poll(self) -> None:
        """One-shot API poll on launch so finished batch jobs show up immediately."""
        if not self.settings_path.exists():
            return

        def worker() -> None:
            try:
                states = poll_local_batch_states(self.settings)
            except Exception:  # noqa: BLE001
                return
            if states:
                self.root.after(0, lambda: self._apply_live_batch_status(states))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_live_batch_status(self, states: dict[str, str]) -> None:
        self._live_batch_status.update(states)
        if callable(self._jobs_repaint):
            self._jobs_repaint()

    def run(self) -> None:
        self.root.mainloop()

    def _setup_theme(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        default_font = ("Helvetica", 12)
        self.root.option_add("*Font", default_font)
        self.root.option_add("*Entry.Font", default_font)

        style.configure("TFrame", background=BG)
        style.configure("App.TFrame", background=BG)
        style.configure("Panel.TFrame", background=MUTED_BG)
        style.configure("TLabel", background=BG, foreground=INK, font=default_font)
        style.configure("App.TLabel", background=BG, foreground=INK, font=default_font)
        style.configure(
            "Muted.TLabel",
            background=BG,
            foreground="#5F6B6E",
            font=("Helvetica", 11),
        )
        style.configure(
            "Heading.TLabel",
            background=BG,
            foreground=INK,
            font=("Helvetica", 24, "bold"),
        )
        style.configure(
            "Section.TLabelframe",
            background=BG,
            foreground=INK,
            bordercolor=SOFT_BORDER,
            relief="solid",
        )
        style.configure(
            "Section.TLabelframe.Label",
            background=BG,
            foreground=INK,
            font=("Helvetica", 13, "bold"),
        )
        style.configure(
            "TEntry",
            fieldbackground=BG,
            foreground=INK,
            bordercolor=SOFT_BORDER,
            lightcolor=SOFT_BORDER,
            darkcolor=SOFT_BORDER,
            padding=8,
        )
        style.configure(
            "TCombobox",
            fieldbackground=BG,
            background=BG,
            foreground=INK,
            arrowcolor=INK,
            bordercolor=SOFT_BORDER,
            padding=7,
        )
        style.configure(
            "Treeview",
            background=BG,
            fieldbackground=BG,
            foreground=INK,
            rowheight=30,
            bordercolor=SOFT_BORDER,
            font=("Helvetica", 11),
        )
        style.configure(
            "Treeview.Heading",
            background=MUTED_BG,
            foreground=INK,
            font=("Helvetica", 11, "bold"),
            padding=8,
        )
        style.map("Treeview", background=[("selected", ACCENT)], foreground=[("selected", BG)])
        style.configure("TProgressbar", troughcolor=MUTED_BG, background=ACCENT)

    def _build_sidebar(self) -> None:
        title = tk.Label(
            self.sidebar,
            text="Patient\nJournals",
            bg=INK,
            fg=BG,
            justify="left",
            font=("Helvetica", 18, "bold"),
        )
        title.pack(anchor="w", pady=(0, 22))
        for key, label, command in (
            ("dashboard", "Dashboard", self.show_dashboard),
            ("submit", "Submit", self.show_submit),
            ("jobs", "Jobs", self.show_jobs),
            ("settings", "Settings", self.show_settings),
        ):
            button = tk.Label(
                self.sidebar,
                text=label,
                bg=INK,
                fg=BG,
                anchor="w",
                padx=14,
                pady=12,
                font=("Helvetica", 13, "bold"),
                cursor="hand2",
            )
            button.bind("<Button-1>", lambda _event, nav_key=key, action=command: self._navigate(nav_key, action))
            button.bind(
                "<Enter>",
                lambda _event, item=button: item.configure(bg="#2B2B33", fg=BG),
            )
            button.bind("<Leave>", lambda _event, nav_key=key, item=button: self._style_nav_item(nav_key, item))
            button.pack(fill="x", pady=3)
            self.nav_items[key] = button

    def _style_nav_item(self, key: str, item: tk.Label) -> None:
        active = getattr(self, "_active_nav", "") == key
        item.configure(
            bg=ACCENT if active else INK,
            fg=INK if active else BG,
        )

    def _navigate(self, key: str, command) -> None:
        self._active_nav = key
        for nav_key, item in self.nav_items.items():
            self._style_nav_item(nav_key, item)
        command()

    def _clear_content(self) -> None:
        after_id = getattr(self, "_jobs_refresh_after_id", None)
        if after_id is not None:
            try:
                self.root.after_cancel(after_id)
            except Exception:  # noqa: BLE001
                pass
            self._jobs_refresh_after_id = None
        self._jobs_repaint = None
        for child in self.content.winfo_children():
            child.destroy()

    def _heading(self, text: str) -> None:
        label = ttk.Label(self.content, text=text, style="Heading.TLabel")
        label.pack(anchor="w", pady=(0, 8))

    def _subheading(self, text: str) -> None:
        label = ttk.Label(self.content, text=text, style="Muted.TLabel")
        label.pack(anchor="w", pady=(0, 18))

    def _field(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int) -> ttk.Entry:
        ttk.Label(parent, text=label, style="App.TLabel").grid(row=row, column=0, sticky="w", pady=8)
        entry = ttk.Entry(parent, textvariable=variable, width=64)
        entry.grid(row=row, column=1, sticky="ew", pady=8, padx=(14, 0))
        return entry

    def _section(self, parent: tk.Misc, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=(16, 14), style="Section.TLabelframe")
        frame.pack(fill="x", anchor="n", pady=(0, 14))
        return frame

    def _advanced_section(self, parent: tk.Misc, title: str = "Advanced") -> tuple[ttk.Frame, tk.BooleanVar]:
        open_var = tk.BooleanVar(value=False)
        shell = ttk.Frame(parent, style="App.TFrame")
        shell.pack(fill="x", anchor="n", pady=(0, 14))
        body = ttk.LabelFrame(shell, text=title, padding=(16, 14), style="Section.TLabelframe")

        def toggle() -> None:
            if open_var.get():
                body.pack_forget()
                open_var.set(False)
                button.configure(text=f"Show {title.lower()}")
            else:
                body.pack(fill="x", anchor="n", pady=(8, 0))
                open_var.set(True)
                button.configure(text=f"Hide {title.lower()}")

        button = self._button(shell, f"Show {title.lower()}", toggle, kind="secondary")
        button.pack(anchor="w")
        return body, open_var

    def _button(self, parent: tk.Misc, text: str, command, *, kind: str = "primary") -> tk.Button:
        primary = kind == "primary"
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=ACCENT if primary else BG,
            fg=INK,
            activebackground=INK if primary else ACCENT,
            activeforeground=BG if primary else INK,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=ACCENT if primary else SOFT_BORDER,
            padx=18,
            pady=12 if primary else 10,
            font=("Helvetica", 13, "bold" if primary else "normal"),
            cursor="hand2",
        )

    def _button_row(self, parent: tk.Misc) -> ttk.Frame:
        frame = ttk.Frame(parent, style="App.TFrame")
        frame.pack(anchor="w", fill="x", pady=(8, 0))
        return frame

    def _scrollable_frame(self, parent: tk.Misc) -> ttk.Frame:
        outer = ttk.Frame(parent, style="App.TFrame")
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        inner = ttk.Frame(canvas, style="App.TFrame")
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def sync_scroll_region(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def sync_width(event) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        def on_mousewheel(event) -> None:
            if getattr(event, "num", None) == 4:
                canvas.yview_scroll(-1, "units")
                return
            if getattr(event, "num", None) == 5:
                canvas.yview_scroll(1, "units")
                return
            delta = getattr(event, "delta", 0)
            if delta:
                steps = int(-delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
                canvas.yview_scroll(steps, "units")

        def bind_scroll(_event=None) -> None:
            canvas.bind_all("<MouseWheel>", on_mousewheel)
            canvas.bind_all("<Button-4>", on_mousewheel)
            canvas.bind_all("<Button-5>", on_mousewheel)

        def unbind_scroll(_event=None) -> None:
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        inner.bind("<Configure>", sync_scroll_region)
        canvas.bind("<Configure>", sync_width)
        canvas.bind("<Enter>", bind_scroll)
        canvas.bind("<Leave>", unbind_scroll)
        inner.bind("<Enter>", bind_scroll)
        inner.bind("<Leave>", unbind_scroll)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y", padx=(8, 0))
        return inner

    def _select_file(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            title="Select service account JSON",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if path:
            variable.set(path)

    def _select_folder(self, variable: tk.StringVar) -> None:
        path = filedialog.askdirectory(title="Select image folder")
        if path:
            variable.set(path)

    def _select_dataset_file(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            title="Select dataset",
            filetypes=(
                ("Datasets", "*.jsonl *.csv"),
                ("JSONL", "*.jsonl"),
                ("CSV", "*.csv"),
                ("All files", "*.*"),
            ),
        )
        if path:
            variable.set(path)

    def _metric_row(self, parent: ttk.Frame, label: str, value: object, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Label(parent, text=str(value)).grid(row=row, column=1, sticky="w", pady=3, padx=(12, 0))

    def _bar_group(self, parent: ttk.Frame, title: str, counts: dict[str, int]) -> None:
        frame = ttk.LabelFrame(parent, text=title, padding=(16, 12), style="Section.TLabelframe")
        frame.pack(fill="x", pady=(8, 0))
        if not counts:
            ttk.Label(frame, text="No data", style="App.TLabel").pack(anchor="w")
            return
        max_count = max(counts.values()) if counts else 1
        for label, count in list(counts.items())[:10]:
            row = ttk.Frame(frame, style="App.TFrame")
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=24, style="App.TLabel").pack(side="left")
            bar = ttk.Progressbar(row, maximum=max_count, value=count)
            bar.pack(side="left", fill="x", expand=True, padx=(8, 8))
            ttk.Label(row, text=str(count), width=8, style="App.TLabel").pack(side="left")

    def show_dashboard(self) -> None:
        self._clear_content()
        self._heading("Dashboard")
        self._subheading("Review run measurements and launch validation from one place.")
        page = self._scrollable_frame(self.content)

        run_root_var = tk.StringVar(value=self.settings.local_runs_root)
        validations_root_var = tk.StringVar(value="validations")
        dataset_var = tk.StringVar(value="")
        images_var = tk.StringVar(value=self.settings.validation_images_root)
        username_var = tk.StringVar(value="unspecified")
        corrections_var = tk.BooleanVar(value=True)
        status_var = tk.StringVar(value="")
        dataset_path_by_label: dict[str, str] = {}

        controls = self._section(page, "Validate")
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Dataset", style="App.TLabel").grid(row=0, column=0, sticky="w", pady=8)
        dataset_combo = ttk.Combobox(
            controls,
            textvariable=dataset_var,
            values=(),
            state="readonly",
        )
        dataset_combo.grid(row=0, column=1, sticky="ew", pady=8, padx=(14, 0))

        def resolve_dataset_selection() -> str:
            return dataset_path_by_label.get(dataset_var.get(), dataset_var.get())

        def load_dataset_choices() -> None:
            files = find_dataset_files(run_root_var.get() or "runs")
            dataset_path_by_label.clear()
            labels: list[str] = []
            for path in files[:100]:
                label = f"{path.parent.name} / {path.name}"
                dataset_path_by_label[label] = str(path)
                labels.append(label)
            dataset_combo.configure(values=labels)
            if labels and dataset_var.get() not in labels:
                dataset_var.set(labels[0])
            status_var.set(f"Loaded {len(labels)} local dataset choice(s).")

        def browse_dataset() -> None:
            self._select_dataset_file(dataset_var)
            if dataset_var.get():
                dataset_path_by_label[dataset_var.get()] = dataset_var.get()
                values = list(dataset_combo.cget("values"))
                if dataset_var.get() not in values:
                    dataset_combo.configure(values=[dataset_var.get(), *values])

        self._button(controls, "Load datasets", load_dataset_choices, kind="secondary").grid(
            row=0, column=2, padx=(10, 0), pady=8
        )
        self._button(controls, "Browse", browse_dataset, kind="secondary").grid(
            row=0, column=3, padx=(8, 0), pady=8
        )
        self._field(controls, "Images", images_var, 1)
        self._button(controls, "Browse", lambda: self._select_folder(images_var), kind="secondary").grid(
            row=1, column=2, padx=(10, 0), pady=8
        )
        self._field(controls, "Validator", username_var, 2)
        ttk.Checkbutton(controls, text="Corrections", variable=corrections_var).grid(
            row=3, column=1, sticky="w", pady=8, padx=(14, 0)
        )

        advanced, _advanced_open = self._advanced_section(page)
        advanced.columnconfigure(1, weight=1)
        self._field(advanced, "Runs folder", run_root_var, 0)
        self._field(advanced, "Validations folder", validations_root_var, 1)

        body = ttk.Frame(page)
        body.pack(fill="both", expand=True, pady=(14, 0))

        def fill_summary() -> None:
            for child in body.winfo_children():
                child.destroy()
            try:
                summary = summarize_dashboard(
                    run_root=run_root_var.get() or "runs",
                    validations_root=validations_root_var.get() or "validations",
                )
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Dashboard failed", str(exc))
                return

            if summary.latest_dataset and not dataset_var.get().strip():
                label = f"{Path(summary.latest_dataset).parent.name} / {Path(summary.latest_dataset).name}"
                dataset_path_by_label[label] = summary.latest_dataset
                values = list(dataset_combo.cget("values"))
                if label not in values:
                    dataset_combo.configure(values=[label, *values])
                dataset_var.set(label)

            left = ttk.Frame(body)
            right = ttk.Frame(body)
            left.pack(side="left", fill="both", expand=True, padx=(0, 12))
            right.pack(side="right", fill="both", expand=True)

            metrics = ttk.LabelFrame(left, text="Run Measurements", padding=(16, 12), style="Section.TLabelframe")
            metrics.pack(fill="x")
            self._metric_row(metrics, "Datasets", summary.dataset_count, 0)
            self._metric_row(metrics, "Dataset rows", summary.dataset_rows, 1)
            self._metric_row(metrics, "Processing records", summary.processing_record_count, 2)
            self._metric_row(metrics, "Images measured", summary.processing_image_count, 3)
            self._metric_row(metrics, "Validation decisions", summary.validation_count, 4)
            self._metric_row(metrics, "Latest dataset", summary.latest_dataset or "none", 5)

            attempts = summary.attempts
            generation = summary.generation_seconds
            dist = ttk.LabelFrame(left, text="Distributions", padding=(16, 12), style="Section.TLabelframe")
            dist.pack(fill="x", pady=(8, 0))
            self._metric_row(dist, "Attempts mean", attempts.get("mean"), 0)
            self._metric_row(dist, "Attempts median", attempts.get("median"), 1)
            self._metric_row(dist, "Attempts max", attempts.get("max"), 2)
            self._metric_row(dist, "Generation seconds mean", generation.get("mean"), 3)
            self._metric_row(dist, "Generation seconds median", generation.get("median"), 4)

            self._bar_group(right, "Processing Status", summary.status_counts)
            self._bar_group(right, "Processing Source", summary.source_counts)
            self._bar_group(right, "Validation Labels", summary.validation_label_counts)
            self._bar_group(right, "Failure Reasons", summary.failure_reasons)
            self._bar_group(right, "Duplicate Actions", summary.duplicate_actions)

            status_var.set("Dashboard refreshed.")

        def use_latest_dataset() -> None:
            latest = latest_dataset_path(run_root_var.get() or "runs")
            if latest is None:
                status_var.set("No dataset found.")
                return
            label = f"{latest.parent.name} / {latest.name}"
            dataset_path_by_label[label] = str(latest)
            values = list(dataset_combo.cget("values"))
            if label not in values:
                dataset_combo.configure(values=[label, *values])
            dataset_var.set(label)
            status_var.set(f"Selected {latest}")

        def launch_validator() -> None:
            try:
                command = build_validation_command(
                    self.settings,
                    images=images_var.get(),
                    results=resolve_dataset_selection(),
                    username=username_var.get(),
                    corrections=corrections_var.get(),
                )
                job = start_command(command, registry=self.registry, kind="validation")
                status_var.set(f"Started validator {job.job_id} pid={job.pid}")
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Validation failed", str(exc))

        buttons = self._button_row(page)
        self._button(buttons, "Launch validator", launch_validator).pack(side="left")
        self._button(buttons, "Use latest dataset", use_latest_dataset, kind="secondary").pack(
            side="left", padx=(10, 0)
        )
        self._button(buttons, "Refresh", fill_summary, kind="secondary").pack(
            side="left", padx=(10, 0)
        )
        self._button(
            buttons,
            "Copy summary JSON",
            lambda: self.root.clipboard_append(
                dashboard_summary_json(
                    summarize_dashboard(
                        run_root=run_root_var.get() or "runs",
                        validations_root=validations_root_var.get() or "validations",
                    )
                )
            ),
            kind="secondary",
        ).pack(side="left", padx=(10, 0))
        ttk.Label(page, textvariable=status_var, style="Muted.TLabel").pack(anchor="w", pady=(10, 18))

        preset = self._preset_validation_dataset
        if preset:
            self._preset_validation_dataset = ""
            preset_path = Path(preset)
            label = f"{preset_path.parent.name} / {preset_path.name}"
            dataset_path_by_label[label] = preset
            dataset_combo.configure(values=[label, *list(dataset_combo.cget("values"))])
            dataset_var.set(label)
            status_var.set(f"Loaded dataset from job: {preset_path.name}")

        fill_summary()

    def show_settings(self) -> None:
        self._clear_content()
        self._heading("Settings")
        self._subheading("Keep the daily defaults visible; cloud routing details live under Advanced.")

        frame = self._section(self.content, "Workspace")
        frame.columnconfigure(1, weight=1)

        auth_var = tk.StringVar(value=self.settings.auth_mode)
        backend_var = tk.StringVar(value=self.settings.batch_backend)
        service_var = tk.StringVar(value=self.settings.service_account_file)
        project_var = tk.StringVar(value=self.settings.gcp_project_id)
        location_var = tk.StringVar(value=self.settings.gcp_location)
        vertex_location_var = tk.StringVar(value=self.settings.vertex_model_location)
        bucket_var = tk.StringVar(value=self.settings.gcs_bucket_name)
        pages_var = tk.StringVar(value=self.settings.gcs_pages_prefix)
        request_prefix_var = tk.StringVar(value=self.settings.batch_requests_gcs_prefix)
        output_prefix_var = tk.StringVar(value=self.settings.batch_outputs_gcs_prefix)
        runs_var = tk.StringVar(value=self.settings.local_runs_root)
        api_env_var = tk.StringVar(value=self.settings.gemini_api_key_env)
        validation_images_var = tk.StringVar(value=self.settings.validation_images_root)
        duplicate_strategy_var = tk.StringVar(value=self.settings.batch_duplicate_strategy)
        cost_rate_var = tk.StringVar(
            value=str(self.settings.estimated_cost_per_1k_images or "")
        )
        api_threshold_var = tk.StringVar(
            value=str(self.settings.api_recovery_threshold)
        )

        ttk.Label(frame, text="Auth mode", style="App.TLabel").grid(row=0, column=0, sticky="w", pady=8)
        auth_combo = ttk.Combobox(
            frame,
            textvariable=auth_var,
            values=("service_account", "adc", "api_key"),
            state="readonly",
            width=24,
        )
        auth_combo.grid(row=0, column=1, sticky="w", pady=8, padx=(14, 0))

        self._field(frame, "Service account JSON", service_var, 1)
        self._button(frame, "Browse", lambda: self._select_file(service_var), kind="secondary").grid(
            row=1, column=2, padx=(10, 0), pady=8
        )
        self._field(frame, "GCP project", project_var, 2)
        self._field(frame, "GCS bucket", bucket_var, 3)
        self._field(frame, "Runs folder", runs_var, 4)
        self._field(frame, "Validation image folder", validation_images_var, 5)
        self._button(frame, "Browse", lambda: self._select_folder(validation_images_var), kind="secondary").grid(
            row=5, column=2, padx=(10, 0), pady=8
        )

        advanced, _advanced_open = self._advanced_section(self.content)
        advanced.columnconfigure(1, weight=1)
        ttk.Label(advanced, text="Batch backend", style="App.TLabel").grid(row=0, column=0, sticky="w", pady=8)
        backend_combo = ttk.Combobox(
            advanced,
            textvariable=backend_var,
            values=("vertex", "mldev"),
            state="readonly",
            width=24,
        )
        backend_combo.grid(row=0, column=1, sticky="w", pady=8, padx=(14, 0))

        self._field(advanced, "GCP location", location_var, 1)
        self._field(advanced, "Vertex model location", vertex_location_var, 2)
        self._field(advanced, "Pages prefix", pages_var, 3)
        self._field(advanced, "Requests prefix", request_prefix_var, 4)
        self._field(advanced, "Outputs prefix", output_prefix_var, 5)
        self._field(advanced, "Gemini API key env", api_env_var, 6)
        ttk.Label(advanced, text="Duplicate strategy", style="App.TLabel").grid(row=7, column=0, sticky="w", pady=8)
        ttk.Combobox(
            advanced,
            textvariable=duplicate_strategy_var,
            values=("first_successful", "provide_all"),
            state="readonly",
            width=24,
        ).grid(row=7, column=1, sticky="w", pady=8, padx=(14, 0))
        self._field(advanced, "Est. cost per 1k images ($)", cost_rate_var, 8)
        self._field(advanced, "API recovery threshold (failures)", api_threshold_var, 9)

        status_var = tk.StringVar(value="")
        ttk.Label(self.content, textvariable=status_var, style="Muted.TLabel").pack(anchor="w", pady=(12, 0))

        def save() -> None:
            try:
                cost_rate = float(cost_rate_var.get().strip() or 0.0)
            except ValueError:
                messagebox.showerror(
                    "Invalid setting",
                    "Est. cost per 1k images must be a number (or blank).",
                )
                return
            try:
                api_threshold = int(float(api_threshold_var.get().strip() or 0))
            except ValueError:
                messagebox.showerror(
                    "Invalid setting",
                    "API recovery threshold must be a whole number.",
                )
                return
            self.settings = AppSettings(
                auth_mode=auth_var.get(),  # type: ignore[arg-type]
                batch_backend=backend_var.get(),
                service_account_file=service_var.get(),
                gcp_project_id=project_var.get(),
                gcp_location=location_var.get(),
                vertex_model_location=vertex_location_var.get(),
                gcs_bucket_name=bucket_var.get(),
                gcs_pages_prefix=pages_var.get(),
                batch_requests_gcs_prefix=request_prefix_var.get(),
                batch_outputs_gcs_prefix=output_prefix_var.get(),
                local_runs_root=runs_var.get(),
                gemini_api_key_env=api_env_var.get(),
                validation_images_root=validation_images_var.get(),
                batch_duplicate_strategy=duplicate_strategy_var.get(),  # type: ignore[arg-type]
                estimated_cost_per_1k_images=cost_rate,
                api_recovery_threshold=api_threshold,
            )
            path = save_app_settings(self.settings, self.settings_path)
            status_var.set(f"Saved {path}")

        buttons = self._button_row(self.content)
        self._button(buttons, "Save settings", save).pack(side="left")
        self._button(
            buttons,
            "Submit job",
            lambda: self._navigate("submit", self.show_submit),
            kind="secondary",
        ).pack(side="left", padx=(10, 0))

    def show_submit(self) -> None:
        self._clear_content()
        self._heading("Submit")
        self._subheading("Choose the input and model, then start the run.")

        frame = self._section(self.content, "Run")
        frame.columnconfigure(1, weight=1)

        source_var = tk.StringVar(value="local")
        mode_var = tk.StringVar(value="local_api")
        local_path_var = tk.StringVar(value="")
        cloud_prefix_var = tk.StringVar(value=self.settings.gcs_pages_prefix)
        schema_names = [option.name for option in self.schema_options]
        model_names = [option.name for option in self.model_options]
        schema_var = tk.StringVar(value=schema_names[0] if schema_names else "")
        model_var = tk.StringVar(value=model_names[0] if model_names else "")
        output_format_var = tk.StringVar(value="jsonl")
        continue_var = tk.StringVar(value="")
        num_batches_var = tk.StringVar(value="")
        status_var = tk.StringVar(value="")
        command_var = tk.StringVar(value="")

        ttk.Label(frame, text="Dataset source", style="App.TLabel").grid(row=0, column=0, sticky="w", pady=8)
        ttk.Combobox(
            frame,
            textvariable=source_var,
            values=("local", "cloud"),
            state="readonly",
            width=18,
        ).grid(row=0, column=1, sticky="w", pady=8, padx=(14, 0))

        ttk.Label(frame, text="Run mode", style="App.TLabel").grid(row=1, column=0, sticky="w", pady=8)
        ttk.Combobox(
            frame,
            textvariable=mode_var,
            values=("local_api", "cloud_batch"),
            state="readonly",
            width=18,
        ).grid(row=1, column=1, sticky="w", pady=8, padx=(14, 0))

        local_label = ttk.Label(frame, text="Local folder", style="App.TLabel")
        local_entry = ttk.Entry(frame, textvariable=local_path_var, width=64)
        local_button = self._button(
            frame,
            "Browse",
            lambda: self._select_folder(local_path_var),
            kind="secondary",
        )

        cloud_label = ttk.Label(frame, text="Cloud dataset", style="App.TLabel")
        cloud_panel = ttk.Frame(frame, style="App.TFrame")
        cloud_panel.columnconfigure(0, weight=1)
        cloud_tree = ttk.Treeview(
            cloud_panel,
            columns=("prefix", "images", "updated"),
            show="headings",
            height=7,
            selectmode="extended",
        )
        cloud_tree.heading("prefix", text="Folder")
        cloud_tree.heading("images", text="Images")
        cloud_tree.heading("updated", text="Updated")
        cloud_tree.column("prefix", anchor="w", width=520)
        cloud_tree.column("images", anchor="e", width=100)
        cloud_tree.column("updated", anchor="w", width=140)
        cloud_tree.grid(row=0, column=0, sticky="ew")
        cloud_scroll = ttk.Scrollbar(cloud_panel, orient="vertical", command=cloud_tree.yview)
        cloud_tree.configure(yscrollcommand=cloud_scroll.set)
        cloud_scroll.grid(row=0, column=1, sticky="ns")
        cloud_actions = ttk.Frame(cloud_panel, style="App.TFrame")
        cloud_actions.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        cloud_prefix_by_iid: dict[str, str] = {}
        cloud_count_by_iid: dict[str, int] = {}
        cloud_select_all_var = tk.BooleanVar(value=False)

        def selected_cloud_prefixes() -> tuple[str, ...]:
            selected = tuple(cloud_tree.selection())
            prefixes = [
                cloud_prefix_by_iid[item_id]
                for item_id in selected
                if item_id in cloud_prefix_by_iid
            ]
            return tuple(dict.fromkeys(prefixes))

        def set_cloud_select_all() -> None:
            rows = cloud_tree.get_children("")
            if cloud_select_all_var.get():
                cloud_tree.selection_set(rows)
                if rows:
                    cloud_tree.focus(rows[0])
                status_var.set(f"Selected {len(rows)} cloud folder(s).")
            else:
                cloud_tree.selection_remove(rows)
                status_var.set("Cleared cloud folder selection.")
            update_cloud_prefix_var()

        def load_cloud_data() -> None:
            try:
                choices = list_cloud_dataset_choices(
                    bucket_name=self.settings.gcs_bucket_name,
                    pages_prefix=self.settings.gcs_pages_prefix,
                    limit=200,
                )
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Cloud list failed", str(exc))
                return
            for item_id in cloud_tree.get_children(""):
                cloud_tree.delete(item_id)
            cloud_prefix_by_iid.clear()
            cloud_count_by_iid.clear()
            cloud_select_all_var.set(False)
            cloud_prefix_var.set("")
            if not choices:
                status_var.set(
                    f"No datasets found in bucket {self.settings.gcs_bucket_name}."
                )
                return
            for index, choice in enumerate(choices, start=1):
                iid = f"cloud_{index}"
                cloud_prefix_by_iid[iid] = choice.prefix
                cloud_count_by_iid[iid] = choice.image_count
                cloud_tree.insert(
                    "",
                    "end",
                    iid=iid,
                    values=(choice.prefix, choice.image_count, choice.updated_at),
                )
            cloud_select_all_var.set(False)
            cloud_tree.selection_remove(cloud_tree.get_children(""))
            status_var.set(
                f"Loaded {len(choices)} cloud folder(s) "
                f"from {self.settings.gcs_bucket_name}."
            )

        def inspect_selected_cloud_data() -> None:
            selected = tuple(cloud_tree.selection())
            if not selected:
                status_var.set("Select one or more cloud folders first.")
                return
            image_count = sum(cloud_count_by_iid.get(item_id, 0) for item_id in selected)
            prefixes = selected_cloud_prefixes()
            if len(prefixes) == 1:
                label = prefixes[0]
            else:
                label = f"{len(prefixes)} folders"
            status_var.set(
                f"Cloud images: {image_count}; selection={label}"
            )

        cloud_inspect_button = self._button(
            cloud_actions,
            "Inspect",
            inspect_selected_cloud_data,
            kind="secondary",
        )
        cloud_select_all_check = ttk.Checkbutton(
            cloud_actions,
            text="Select all",
            variable=cloud_select_all_var,
            command=set_cloud_select_all,
        )
        cloud_load_button = self._button(
            cloud_actions,
            "Load bucket",
            load_cloud_data,
            kind="secondary",
        )
        cloud_load_button.pack(side="left")
        cloud_select_all_check.pack(side="left", padx=(12, 0))
        cloud_inspect_button.pack(side="left", padx=(8, 0))

        def update_cloud_prefix_var(*_args) -> None:
            prefixes = selected_cloud_prefixes()
            if prefixes:
                cloud_prefix_var.set(prefixes[0])
            else:
                cloud_prefix_var.set("")
            selected_count = len(cloud_tree.selection())
            total_count = len(cloud_tree.get_children(""))
            cloud_select_all_var.set(bool(total_count and selected_count == total_count))

        cloud_tree.bind("<<TreeviewSelect>>", update_cloud_prefix_var)

        def show_local_source() -> None:
            cloud_label.grid_remove()
            cloud_panel.grid_remove()
            local_label.grid(row=2, column=0, sticky="w", pady=8)
            local_entry.grid(row=2, column=1, sticky="ew", pady=8, padx=(14, 0))
            local_button.grid(row=2, column=2, sticky="w", padx=(10, 0), pady=8)

        def show_cloud_source() -> None:
            local_label.grid_remove()
            local_entry.grid_remove()
            local_button.grid_remove()
            cloud_label.grid(row=2, column=0, sticky="w", pady=8)
            cloud_panel.grid(row=2, column=1, columnspan=3, sticky="ew", pady=8, padx=(14, 0))

        def update_source_view(*_args) -> None:
            if source_var.get() == "cloud":
                mode_var.set("cloud_batch")
                show_cloud_source()
            else:
                show_local_source()

        source_var.trace_add("write", update_source_view)
        update_source_view()

        ttk.Label(frame, text="Schema", style="App.TLabel").grid(row=3, column=0, sticky="w", pady=8)
        ttk.Combobox(frame, textvariable=schema_var, values=schema_names, state="readonly").grid(
            row=3, column=1, sticky="ew", pady=8, padx=(14, 0)
        )
        ttk.Label(frame, text="Model", style="App.TLabel").grid(row=4, column=0, sticky="w", pady=8)
        ttk.Combobox(frame, textvariable=model_var, values=model_names, state="readonly").grid(
            row=4, column=1, sticky="ew", pady=8, padx=(14, 0)
        )

        advanced, _advanced_open = self._advanced_section(self.content)
        advanced.columnconfigure(1, weight=1)
        ttk.Label(advanced, text="Output format", style="App.TLabel").grid(row=0, column=0, sticky="w", pady=8)
        ttk.Combobox(
            advanced,
            textvariable=output_format_var,
            values=("jsonl", "csv"),
            state="readonly",
            width=18,
        ).grid(row=0, column=1, sticky="w", pady=8, padx=(14, 0))
        self._field(advanced, "Continue dataset", continue_var, 1)
        self._field(advanced, "Batch chunks", num_batches_var, 2)

        preview = ttk.Label(advanced, textvariable=command_var, wraplength=850, style="Muted.TLabel")
        preview.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        ttk.Label(self.content, textvariable=status_var, style="Muted.TLabel").pack(anchor="w", pady=(8, 0))

        def draft() -> SubmitJobDraft:
            num_batches = None
            if num_batches_var.get().strip():
                num_batches = max(1, int(num_batches_var.get().strip()))
            cloud_prefixes = selected_cloud_prefixes() if source_var.get() == "cloud" else ()
            if (
                source_var.get() == "cloud"
                and not cloud_prefixes
            ):
                raise ValueError("Load the bucket and select one or more cloud folders.")
            return SubmitJobDraft(
                dataset_source=source_var.get(),  # type: ignore[arg-type]
                run_mode=mode_var.get(),  # type: ignore[arg-type]
                schema_name=schema_var.get(),
                model_name=model_var.get(),
                output_format=output_format_var.get(),  # type: ignore[arg-type]
                local_path=local_path_var.get(),
                cloud_prefix=cloud_prefixes[0] if cloud_prefixes else cloud_prefix_var.get(),
                cloud_prefixes=cloud_prefixes,
                continue_dataset=continue_var.get(),
                num_batches=num_batches,
            )

        def preview_command() -> None:
            try:
                current = draft()
                if current.local_path:
                    summary = inspect_local_dataset(current.local_path)
                    status_var.set(
                        f"Local images: {summary.image_count}; status={summary.status}"
                    )
                command = build_submit_command(current, self.settings)
                command_var.set(command.display())
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Preview failed", str(exc))

        def run_job() -> None:
            try:
                current = draft()
                if current.run_mode == "local_api":
                    status_var.set("Starting local run...")

                    def progress(progress_event) -> None:
                        if progress_event.event == "image_processed":
                            self.root.after(
                                0,
                                lambda: status_var.set(
                                    "Processed "
                                    f"{progress_event.processed_images}/"
                                    f"{progress_event.total_images} image(s)."
                                ),
                            )

                    def worker() -> None:
                        try:
                            result = asyncio.run(
                                run_local_draft_direct(
                                    current,
                                    self.settings,
                                    progress_callback=progress,
                                )
                            )
                            self.root.after(
                                0,
                                lambda: status_var.set(
                                    f"Local run {result.status}: {result.dataset_path}"
                                ),
                            )
                        except Exception as exc:  # noqa: BLE001
                            self.root.after(
                                0,
                                lambda: messagebox.showerror("Submit failed", str(exc)),
                            )

                    threading.Thread(target=worker, daemon=True).start()
                    return

                # cloud_batch: confirm the exact scope before spending money.
                if current.dataset_source == "local":
                    names = local_image_names(current.local_path)
                    expected = len(names)
                    if not expected:
                        messagebox.showerror(
                            "Submit failed",
                            "No batch input images were found in the selected folder.",
                        )
                        return
                    scope_text = (
                        f"{expected} image(s) from the local folder:\n"
                        f"{current.local_path}"
                    )
                else:
                    expected = None
                    folders = current.cloud_prefixes or (
                        (current.cloud_prefix,) if current.cloud_prefix else ()
                    )
                    scope_text = "the selected cloud folder(s):\n" + "\n".join(folders)

                cost_line = ""
                rate = self.settings.estimated_cost_per_1k_images or 0.0
                if rate and expected:
                    cost_line = (
                        f"\n\nRough estimate: ~${expected / 1000 * rate:,.2f} "
                        f"(at ${rate:g}/1k images — not a billed figure)."
                    )
                elif expected:
                    cost_line = (
                        "\n\nSet 'Est. cost per 1k images' in Settings to see a "
                        "cost estimate here."
                    )

                if not messagebox.askyesno(
                    "Confirm batch submission",
                    "This will submit a paid batch job for "
                    f"{scope_text}{cost_line}\n\nProceed?",
                    icon="warning",
                ):
                    status_var.set("Submission cancelled.")
                    return

                # run in-process so the run directory, model, and image count
                # are captured immediately as a single tracked job.
                command_var.set(build_submit_command(current, self.settings).display())
                status_var.set("Submitting batch... this may take a moment.")

                def batch_worker() -> None:
                    try:
                        outcome = run_batch_draft_direct(current, self.settings)
                    except Exception as exc:  # noqa: BLE001
                        self.root.after(
                            0,
                            lambda: messagebox.showerror("Submit failed", str(exc)),
                        )
                        return
                    if outcome is None:
                        self.root.after(
                            0,
                            lambda: status_var.set(
                                "Nothing to submit; dataset already covers the inputs."
                            ),
                        )
                        return
                    if expected is not None and outcome.request_count > expected:
                        # Scope guard failed: submitted more than the folder held.
                        self.root.after(
                            0,
                            lambda: messagebox.showwarning(
                                "Unexpected submission size",
                                f"Submitted {outcome.request_count} requests but the "
                                f"selected folder only had {expected} image(s). "
                                f"Review run {Path(outcome.run_dir).name} before "
                                "retrieving.",
                            ),
                        )
                    self.root.after(
                        0,
                        lambda: status_var.set(
                            f"Submitted {outcome.request_count} image(s) in "
                            f"{outcome.chunk_count} chunk(s) "
                            f"[{outcome.model}] -> {Path(outcome.run_dir).name}. "
                            "See the Jobs tab."
                        ),
                    )

                threading.Thread(target=batch_worker, daemon=True).start()
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Submit failed", str(exc))

        buttons = self._button_row(self.content)
        self._button(buttons, "Submit", run_job).pack(side="left")
        self._button(buttons, "Preview command", preview_command, kind="secondary").pack(side="left", padx=(10, 0))

    def _open_results_window(
        self,
        job,
        *,
        on_change=None,
        recover_missing: bool = False,
        duplicate_strategy: str = "",
    ) -> None:
        """Retrieve a job's outputs, show a summary + sample, then let the user
        keep the results or rerun the work that failed."""
        win = tk.Toplevel(self.root)
        win.title(f"Results — {job.job_id}")
        win.geometry("920x620")
        win.minsize(720, 480)
        win.configure(bg=BG)
        win.transient(self.root)

        header = ttk.Frame(win, style="App.TFrame", padding=(18, 14))
        header.pack(fill="x")
        ttk.Label(
            header,
            text=f"{job.model or 'batch'} — {job.image_count} image(s)",
            style="Heading.TLabel",
        ).pack(anchor="w")
        ttk.Label(header, text=job.input_location, style="Muted.TLabel").pack(anchor="w")

        state_var = tk.StringVar(value="Retrieving batch outputs…")
        ttk.Label(win, textvariable=state_var, style="Muted.TLabel").pack(
            anchor="w", padx=18
        )

        metrics = ttk.Frame(win, style="App.TFrame", padding=(18, 8))
        metrics.pack(fill="x")
        total_var = tk.StringVar(value="—")
        ok_var = tk.StringVar(value="—")
        fail_var = tk.StringVar(value="—")
        for index, (label, var) in enumerate(
            (("Total", total_var), ("Succeeded", ok_var), ("Failed / missing", fail_var))
        ):
            cell = ttk.Frame(metrics, style="App.TFrame")
            cell.grid(row=0, column=index, padx=(0, 28), sticky="w")
            ttk.Label(cell, textvariable=var, style="Heading.TLabel").pack(anchor="w")
            ttk.Label(cell, text=label, style="Muted.TLabel").pack(anchor="w")

        path_var = tk.StringVar(value="")
        ttk.Label(win, textvariable=path_var, style="Muted.TLabel").pack(
            anchor="w", padx=18, pady=(0, 4)
        )

        preview_frame = ttk.LabelFrame(
            win, text="Sample of extracted rows", padding=(8, 8), style="Section.TLabelframe"
        )
        preview_frame.pack(fill="both", expand=True, padx=18, pady=(8, 8))
        preview = ttk.Treeview(preview_frame, show="headings", height=10)
        preview.pack(side="left", fill="both", expand=True)
        pscroll = ttk.Scrollbar(preview_frame, orient="vertical", command=preview.yview)
        preview.configure(yscrollcommand=pscroll.set)
        pscroll.pack(side="right", fill="y")

        error_box = tk.Text(
            win, height=4, wrap="word", bg=MUTED_BG, fg=INK, relief="flat", padx=10, pady=8
        )

        footer = ttk.Frame(win, style="App.TFrame", padding=(18, 12))
        footer.pack(fill="x", side="bottom")

        state = {
            "dataset": "",
            "failed": 0,
            "done": False,
            "mode": "request",
            "rerun_method": "batch",
        }

        def populate_preview(columns, rows) -> None:
            # Clear any rows from a previous load so re-retrieve/recover can repaint.
            preview.delete(*preview.get_children())
            preview["columns"] = list(columns)
            for column in columns:
                preview.heading(column, text=column)
                preview.column(column, width=160, anchor="w", stretch=True)
            for row_index, row in enumerate(rows):
                preview.insert(
                    "",
                    "end",
                    iid=str(row_index),
                    values=[_truncate_cell(row.get(column, "")) for column in columns],
                )

        def show_error(text: str) -> None:
            if not text:
                return
            error_box.pack(fill="x", padx=18, pady=(0, 8))
            error_box.delete("1.0", "end")
            error_box.insert("1.0", f"Error / notes:\n{text}")
            error_box.configure(state="disabled")

        def finish(payload, dataset_path, columns, rows, error_text) -> None:
            state["done"] = True
            if payload is not None:
                expected = int(payload.get("expected_pages") or job.image_count or 0)
                succeeded = int(payload.get("successful_pages") or 0)
                missing = int(payload.get("missing_pages") or max(0, expected - succeeded))
                total_var.set(str(expected))
                ok_var.set(str(succeeded))
                fail_var.set(str(missing))
                state["dataset"] = dataset_path or str(payload.get("dataset_path") or "")
                state["failed"] = missing
                state["mode"] = "request"
                populate_preview(columns, rows)
                if rows:
                    state_var.set(
                        "Review the sample below, then keep the results or rerun "
                        "what failed."
                    )
                else:
                    state_var.set(
                        f"Retrieved {succeeded} row(s), but no preview rows could be "
                        "read from the dataset (see path below)."
                    )
                path_var.set(f"Dataset: {state['dataset'] or '(none written)'}")
                if missing > 0:
                    provider = batch_run_provider(job.run_dir)
                    threshold = int(self.settings.api_recovery_threshold or 0)
                    use_api = provider == "gemini" and missing <= threshold
                    state["rerun_method"] = "api" if use_api else "batch"
                    rerun_button.configure(
                        text=(
                            f"Recover {missing} via API"
                            if use_api
                            else f"Resubmit {missing} as batch"
                        ),
                        state="normal",
                    )
                else:
                    state["rerun_method"] = "batch"
                    rerun_button.configure(text="Rerun failed", state="disabled")
                validate_button.configure(
                    state="normal" if state["dataset"] else "disabled"
                )
            else:
                # Retrieval produced nothing (the batch itself failed); offer a
                # whole-job rerun of the failed chunks instead.
                expected = int(job.image_count or 0)
                total_var.set(str(expected) if expected else "—")
                ok_var.set("0")
                fail_var.set(str(expected) if expected else "—")
                state["mode"] = "chunk"
                state["failed"] = expected or 1
                state_var.set(
                    "No results could be retrieved. You can rerun the failed job."
                )
                rerun_button.configure(text="Rerun job", state="normal")
            show_error(error_text)

        def load(force_retrieve: bool = False) -> None:
            import traceback

            try:
                # Fast path: if this job was already retrieved and its dataset is
                # still on disk, show it directly instead of re-running a full
                # (slow, networked) retrieve.
                payload: dict | None = None
                if not force_retrieve:
                    recorded = read_recorded_results(job.run_dir)
                    if recorded:
                        existing = find_dataset_near(recorded.get("dataset_path") or "")
                        if existing:
                            payload = dict(recorded)
                            payload["dataset_path"] = existing

                if payload is None:
                    self.root.after(
                        0, lambda: state_var.set("Retrieving batch outputs… (this can take a while)")
                    )
                    payload = run_retrieve_direct(
                        job.run_dir,
                        self.settings,
                        allow_partial=True,
                        recover_missing_with_api=recover_missing,
                        duplicate_strategy=duplicate_strategy,
                    )

                dataset_path = find_dataset_near(payload.get("dataset_path") or "")
                columns, rows = read_dataset_preview(dataset_path, limit=25)
                local = read_run_error(job.run_dir)
                self.root.after(
                    0,
                    lambda: finish(payload, dataset_path, columns, rows, local),
                )
            except Exception:  # noqa: BLE001
                message = traceback.format_exc()
                local = read_run_error(job.run_dir)
                if local:
                    message = f"{message}\n\n{local}"
                self.root.after(0, lambda: finish(None, "", [], [], message))

        def keep_results() -> None:
            win.destroy()
            if on_change:
                on_change(f"Kept results for {job.job_id}.")

        def rerun() -> None:
            if not state.get("failed"):
                return
            mode = state.get("mode")
            method = state.get("rerun_method", "batch")
            rerun_button.configure(state="disabled")

            if mode == "chunk":
                state_var.set("Resubmitting the failed job as a batch…")
                action = lambda: run_batch_rerun_direct(job.run_dir, self.settings)
                reload_after = False
            elif method == "api":
                state_var.set(
                    f"Recovering {state['failed']} missing page(s) via API…"
                )
                action = lambda: recover_dataset_gaps(job.run_dir, self.settings)
                reload_after = True
            else:
                state_var.set(
                    f"Resubmitting {state['failed']} failed page(s) as a batch…"
                )
                action = lambda: resubmit_failed_requests(job.run_dir, self.settings)
                reload_after = False

            def worker() -> None:
                try:
                    action()
                except Exception as exc:  # noqa: BLE001

                    def on_err() -> None:
                        messagebox.showerror("Rerun failed", str(exc), parent=win)
                        rerun_button.configure(state="normal")

                    self.root.after(0, on_err)
                    return

                def done() -> None:
                    self._live_batch_status.pop(job.run_dir, None)
                    if reload_after:
                        # API recovery completes the dataset immediately; refresh
                        # this window in place instead of closing it.
                        if on_change:
                            on_change(f"Recovered failed pages via API for {job.job_id}.")
                        start_load()
                    else:
                        win.destroy()
                        if on_change:
                            on_change(
                                f"Resubmitted failed work for {job.job_id}. "
                                "Refresh status, then view results again."
                            )

                self.root.after(0, done)

            threading.Thread(target=worker, daemon=True).start()

        def validate_in_dashboard() -> None:
            self._preset_validation_dataset = state.get("dataset", "")
            win.destroy()
            self._navigate("dashboard", self.show_dashboard)

        def start_load(force_retrieve: bool = False) -> None:
            state_var.set(
                "Re-retrieving batch outputs…" if force_retrieve else "Loading results…"
            )
            threading.Thread(
                target=lambda: load(force_retrieve=force_retrieve), daemon=True
            ).start()

        self._button(footer, "Keep results", keep_results).pack(side="left")
        rerun_button = self._button(footer, "Rerun failed", rerun, kind="secondary")
        rerun_button.pack(side="left", padx=(10, 0))
        rerun_button.configure(state="disabled")
        self._button(
            footer,
            "Re-retrieve",
            lambda: start_load(force_retrieve=True),
            kind="secondary",
        ).pack(side="left", padx=(10, 0))
        validate_button = self._button(
            footer, "Validate in Dashboard", validate_in_dashboard, kind="secondary"
        )
        validate_button.pack(side="left", padx=(10, 0))
        validate_button.configure(state="disabled")
        self._button(
            footer,
            "Open folder",
            lambda: _open_in_file_browser(job.run_dir),
            kind="secondary",
        ).pack(side="left", padx=(10, 0))
        self._button(footer, "Close", win.destroy, kind="secondary").pack(
            side="right"
        )

        start_load()

    def show_jobs(self) -> None:
        self._clear_content()
        self._heading("Jobs")
        self._subheading(
            "One row per submitted batch. Select a job to view results, "
            "refresh status, or cancel."
        )

        # Reserve the bottom for the controls so the action buttons stay visible
        # no matter how many rows the tables hold.
        footer = ttk.Frame(self.content, style="App.TFrame")
        footer.pack(side="bottom", fill="x", pady=(8, 0))

        columns = ("created", "model", "folder", "images", "status", "success", "failed")
        headings = {
            "created": "Created",
            "model": "Model",
            "folder": "Folder / cloud location",
            "images": "Images",
            "status": "Status",
            "success": "Success",
            "failed": "Failed",
        }
        widths = {
            "created": 150,
            "model": 150,
            "folder": 330,
            "images": 80,
            "status": 110,
            "success": 80,
            "failed": 70,
        }
        tree = ttk.Treeview(
            self.content,
            columns=columns,
            show="headings",
            height=11,
            selectmode="browse",
        )
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(
                column,
                anchor="e" if column in {"images", "success", "failed"} else "w",
                width=widths[column],
            )
        tree.pack(side="top", fill="both", expand=True)

        chunk_label = ttk.Label(self.content, text="Chunks (status detail)")
        chunk_label.pack(side="top", anchor="w", pady=(10, 0))
        chunk_columns = ("chunk", "status", "requests", "provider", "destination")
        chunk_tree = ttk.Treeview(
            self.content,
            columns=chunk_columns,
            show="headings",
            height=4,
            selectmode="none",
        )
        for column in chunk_columns:
            chunk_tree.heading(column, text=column.title())
            chunk_tree.column(column, anchor="w", width=140)
        chunk_tree.column("destination", width=420)
        chunk_tree.pack(side="top", fill="x")

        status_var = tk.StringVar(value="")
        duplicate_strategy_var = tk.StringVar(value=self.settings.batch_duplicate_strategy)
        ttk.Label(footer, textvariable=status_var, style="Muted.TLabel").pack(
            side="bottom", anchor="w", pady=(8, 0)
        )
        all_jobs_by_id: dict[str, object] = {}
        AUTO_REFRESH_MS = 5000

        def effective_status(job) -> str:
            if job.retrieved:
                return job.status
            live = self._live_batch_status.get(job.run_dir)
            return live or job.status

        def job_values(job):
            return (
                job.created_at,
                job.model,
                job.input_location,
                job.image_count,
                effective_status(job),
                "" if job.succeeded is None else str(job.succeeded),
                "" if job.failed is None else str(job.failed),
            )

        def insert_jobs(jobs, *, quiet: bool = False) -> None:
            kept_selection = set(tree.selection())
            all_jobs_by_id.clear()
            for job in jobs:
                all_jobs_by_id[job.job_id] = job
            new_ids = [job.job_id for job in jobs]
            existing_ids = list(tree.get_children())
            if new_ids != existing_ids:
                for item in existing_ids:
                    tree.delete(item)
                for job in jobs:
                    tree.insert("", "end", iid=job.job_id, values=job_values(job))
            else:
                for job in jobs:
                    tree.item(job.job_id, values=job_values(job))
            restore = [iid for iid in kept_selection if iid in all_jobs_by_id]
            if restore:
                tree.selection_set(restore)
            if not quiet:
                status_var.set(f"{len(jobs)} batch job(s).")

        def refresh(*, quiet: bool = False) -> None:
            insert_jobs(list_submit_jobs(self.settings.local_runs_root), quiet=quiet)

        def auto_refresh() -> None:
            if not tree.winfo_exists():
                return
            try:
                refresh(quiet=True)
            except Exception:  # noqa: BLE001
                pass
            self._jobs_refresh_after_id = self.root.after(AUTO_REFRESH_MS, auto_refresh)

        def insert_chunks(chunks) -> None:
            for item in chunk_tree.get_children():
                chunk_tree.delete(item)
            for chunk in chunks:
                chunk_tree.insert(
                    "",
                    "end",
                    iid=chunk.batch_job_name,
                    values=(
                        chunk.chunk_label,
                        chunk.status,
                        chunk.request_count,
                        chunk.provider,
                        chunk.output_destination,
                    ),
                )

        def selected_job():
            selection = tree.selection()
            if not selection:
                return None
            return all_jobs_by_id.get(selection[0])

        def show_chunks_for_selection(*, with_state: bool = False) -> None:
            job = selected_job()
            if job is None or not job.run_dir:
                insert_chunks([])
                return
            if not with_state:
                insert_chunks(list_batch_chunks(job.run_dir))
                return

            status_var.set("Querying batch status from the API...")
            run_dir = job.run_dir

            def worker() -> None:
                try:
                    chunks = list_batch_chunks_with_state(run_dir)
                except Exception as exc:  # noqa: BLE001
                    self.root.after(
                        0,
                        lambda: messagebox.showerror("Status check failed", str(exc)),
                    )
                    return

                def apply() -> None:
                    insert_chunks(chunks)
                    state = aggregate_batch_state(chunks)
                    if state:
                        self._live_batch_status[run_dir] = state
                        refresh(quiet=True)
                        status_var.set(f"Status: {state} ({len(chunks)} chunk(s)).")

                self.root.after(0, apply)

            threading.Thread(target=worker, daemon=True).start()

        recover_missing_var = tk.BooleanVar(value=False)

        def after_results(message: str = "") -> None:
            refresh(quiet=True)
            if message:
                status_var.set(message)

        def view_results() -> None:
            job = selected_job()
            if job is None or not job.run_dir:
                status_var.set("Select a batch job to view results.")
                return
            self._open_results_window(
                job,
                on_change=after_results,
                recover_missing=recover_missing_var.get(),
                duplicate_strategy=duplicate_strategy_var.get(),
            )

        def cancel_selected() -> None:
            job = selected_job()
            if job is None or not job.run_dir:
                status_var.set("Select a running job to cancel.")
                return
            if not messagebox.askyesno(
                "Cancel batch",
                f"Request cancellation of batch job {job.job_id}?",
                icon="warning",
            ):
                return
            run_dir = job.run_dir
            status_var.set(f"Cancelling {job.job_id}…")

            def worker() -> None:
                try:
                    cancelled = cancel_batch_run(run_dir, self.settings)
                except Exception as exc:  # noqa: BLE001
                    self.root.after(
                        0, lambda: messagebox.showerror("Cancel failed", str(exc))
                    )
                    return

                def done() -> None:
                    self._live_batch_status.pop(run_dir, None)
                    refresh(quiet=True)
                    status_var.set(
                        f"Cancellation requested for {cancelled} batch job(s)."
                        if cancelled
                        else "No non-terminal jobs to cancel."
                    )

                self.root.after(0, done)

            threading.Thread(target=worker, daemon=True).start()

        tree.bind(
            "<<TreeviewSelect>>",
            lambda _event: show_chunks_for_selection(with_state=False),
        )
        tree.bind("<Double-1>", lambda _event: view_results())

        buttons = self._button_row(footer)
        self._button(buttons, "View results", view_results).pack(side="left")
        self._button(
            buttons,
            "Refresh status",
            lambda: show_chunks_for_selection(with_state=True),
            kind="secondary",
        ).pack(side="left", padx=(10, 0))
        self._button(buttons, "Cancel job", cancel_selected, kind="secondary").pack(
            side="left", padx=(10, 0)
        )
        self._button(buttons, "Refresh", lambda: refresh(), kind="secondary").pack(
            side="left", padx=(18, 0)
        )

        advanced, _advanced_open = self._advanced_section(footer)
        advanced.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            advanced,
            text="Recover missing pages via API when viewing results",
            variable=recover_missing_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(advanced, text="Duplicates", style="App.TLabel").grid(
            row=1, column=0, sticky="w", pady=(14, 0)
        )
        ttk.Combobox(
            advanced,
            textvariable=duplicate_strategy_var,
            values=("first_successful", "provide_all"),
            state="readonly",
            width=18,
        ).grid(row=1, column=1, sticky="w", pady=(14, 0), padx=(14, 0))
        self._jobs_repaint = lambda: refresh(quiet=True)
        refresh()
        self._jobs_refresh_after_id = self.root.after(AUTO_REFRESH_MS, auto_refresh)


def main() -> None:
    PatientJournalsApp().run()


if __name__ == "__main__":
    main()
