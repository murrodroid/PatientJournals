import argparse
import csv
import json
import math
import random
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import Tk, Label, Button, StringVar, messagebox, Frame, Canvas, Scrollbar, Entry
from typing import Literal, get_args, get_origin

import pandas as pd
from PIL import Image, ImageTk
from pydantic import BaseModel, TypeAdapter, ValidationError

from patientjournals.config import config
from patientjournals.config.schemas import FrontPage
from patientjournals.shared.field_classification import is_metadata_field as is_support_field
from patientjournals.shared.identity import row_image_name
from patientjournals.validation.sync import (
    upload_validation_run,
    write_validation_metadata,
)

SamplingMode = Literal["random", "balanced_ucb"]

_SCORE_BY_LABEL = {
    "accept": 1.0,
    "somewhat_accept": 0.5,
    "reject": 0.0,
    "corrected": 0.0,
}
_BALANCED_UCB_LAMBDA = 0.5
_BALANCED_UCB_GAMMA = 0.25


@dataclass(frozen=True)
class ValidationDatapoint:
    row: dict
    image_path: Path
    image_name: str
    field_name: str
    field_value: object

def build_image_index(root_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name in index:
            continue
        index[name] = path
    return index


def load_dataset(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, sep="$")
        return df.to_dict(orient="records")
    if path.suffix.lower() == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    raise ValueError(f"Unsupported dataset format: {path}")


def flatten_row(row: dict) -> dict:
    flat = pd.json_normalize(row, sep=".").to_dict(orient="records")
    return flat[0] if flat else {}


def eligible_flat_fields(row: dict) -> list[tuple[str, object]]:
    flat = flatten_row(row)
    candidates: list[tuple[str, object]] = []
    for key, value in flat.items():
        if not _is_validation_schema_field(key):
            continue
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        if isinstance(value, (dict, list)):
            continue
        candidates.append((key, value))
    return candidates


def pick_flat_field(row: dict, rng: random.Random) -> tuple[str, object] | None:
    candidates = eligible_flat_fields(row)
    return rng.choice(candidates) if candidates else None


def build_validation_datapoints(
    rows: list[dict],
    image_index: dict[str, Path],
) -> list[ValidationDatapoint]:
    datapoints: list[ValidationDatapoint] = []
    for row in rows:
        image_name = display_image_name(row)
        if not image_name:
            continue
        image_path = image_index.get(image_name)
        if image_path is None:
            continue
        for field_name, field_value in eligible_flat_fields(row):
            datapoints.append(
                ValidationDatapoint(
                    row=row,
                    image_path=image_path,
                    image_name=image_name,
                    field_name=field_name,
                    field_value=field_value,
                )
            )
    return datapoints


def _score_for_label(label: str) -> float | None:
    return _SCORE_BY_LABEL.get(label)


def choose_random_datapoint(
    datapoints: list[ValidationDatapoint],
    validated_pairs: set[tuple[str, str]],
    rng: random.Random,
) -> ValidationDatapoint | None:
    remaining = [
        item
        for item in datapoints
        if (item.image_name, item.field_name) not in validated_pairs
    ]
    return rng.choice(remaining) if remaining else None


def choose_balanced_ucb_datapoint(
    datapoints: list[ValidationDatapoint],
    validated_pairs: set[tuple[str, str]],
    selection_counts: dict[str, int],
    scored_counts: dict[str, int],
    score_sums: dict[str, float],
    rng: random.Random,
    *,
    target_lambda: float = _BALANCED_UCB_LAMBDA,
    gamma: float = _BALANCED_UCB_GAMMA,
) -> ValidationDatapoint | None:
    remaining_by_field: dict[str, list[ValidationDatapoint]] = {}
    total_by_field: dict[str, int] = {}
    for item in datapoints:
        total_by_field[item.field_name] = total_by_field.get(item.field_name, 0) + 1
        if (item.image_name, item.field_name) not in validated_pairs:
            remaining_by_field.setdefault(item.field_name, []).append(item)
    if not remaining_by_field:
        return None

    total_datapoints = max(1, len(datapoints))
    group_count = max(1, len(total_by_field))
    t = sum(selection_counts.values())
    denominator = max(t, 1)
    log_term = math.log(max(t, 2))
    scored_groups = []
    for field_name in remaining_by_field:
        n_g = selection_counts.get(field_name, 0)
        m_g = scored_counts.get(field_name, 0)
        # Smoothed field accuracy is kept for diagnostics and future reporting;
        # selection uses coverage deficit plus uncertainty, not reward maximization.
        _p_hat_g = (score_sums.get(field_name, 0.0) + 1.0) / (m_g + 2.0)
        q_g = (1.0 - target_lambda) * (1.0 / group_count) + target_lambda * (
            total_by_field[field_name] / total_datapoints
        )
        deficit = q_g - (n_g / denominator)
        uncertainty = math.sqrt((2.0 * log_term) / (m_g + 1.0))
        score = deficit + gamma * uncertainty
        scored_groups.append((score, rng.random(), field_name))

    _score, _tie_breaker, selected_field = max(scored_groups)
    return rng.choice(remaining_by_field[selected_field])

def _stringify_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)

def _unwrap_optional(field_type: object) -> object:
    origin = get_origin(field_type)
    if origin is None:
        return field_type
    if origin is list or origin is tuple or origin is set:
        return field_type
    args = [arg for arg in get_args(field_type) if arg is not type(None)]
    return args[0] if len(args) == 1 else field_type


def _schema_model() -> type[BaseModel]:
    candidate = getattr(config, "output_model", None)
    if isinstance(candidate, type) and hasattr(candidate, "model_fields"):
        return candidate
    return FrontPage


def _is_metadata_field(path: str) -> bool:
    return is_support_field(path)


def _is_validation_schema_field(path: str) -> bool:
    return not _is_metadata_field(path) and _get_field_type(path) is not None


def _get_field_type(path: str) -> object | None:
    current = _schema_model()
    field_type: object | None = None
    for part in path.split("."):
        if not hasattr(current, "model_fields"):
            return None
        field_info = current.model_fields.get(part)
        if field_info is None:
            return None
        field_type = _unwrap_optional(field_info.annotation)
        origin = get_origin(field_type)
        if origin in {list, tuple, set}:
            args = get_args(field_type)
            field_type = args[0] if args else None
            origin = get_origin(field_type)
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            current = field_type
        else:
            current = None
    return field_type

def _parse_corrected_value(field_name: str, text: str) -> object:
    stripped = text.strip()
    if stripped == "":
        return ""
    field_type = _get_field_type(field_name)
    if field_type is None:
        return stripped
    try:
        adapter = TypeAdapter(field_type)
        return adapter.validate_python(stripped)
    except ValidationError as exc:
        raise ValueError(f"Value '{stripped}' is invalid for {field_name}.") from exc


def resolve_image_path(row: dict, image_index: dict[str, Path]) -> Path | None:
    image_name = row_image_name(row)
    if not image_name:
        return None
    return image_index.get(image_name)


def display_image_name(row: dict) -> str:
    return row_image_name(row) or ""


class ValidatorApp:
    def __init__(
        self,
        dataset_path: Path,
        image_root: Path,
        username: str,
        allow_corrections: bool,
        sampling_mode: SamplingMode = "random",
    ):
        self.dataset_path = dataset_path
        self.image_root = image_root
        self.username = username
        self.allow_corrections = allow_corrections
        self.sampling_mode = sampling_mode
        self.seed = secrets.randbits(64)
        self.rng = random.Random(self.seed)
        self.rows = load_dataset(dataset_path)
        self.image_index = build_image_index(image_root)
        self.datapoints = build_validation_datapoints(self.rows, self.image_index)
        self.results: list[dict] = []
        self.validated_pairs: set[tuple[str, str]] = set()
        self.selection_counts: dict[str, int] = {}
        self.scored_counts: dict[str, int] = {}
        self.score_sums: dict[str, float] = {}
        self.total_pairs = len(self.datapoints)
        self.current_row = None
        self.current_field = None
        self.current_image = None
        self.current_datapoint: ValidationDatapoint | None = None
        self.run_dir = self._create_run_dir()
        self.log_path = self.run_dir / "validation.log"

        self.root = Tk()
        self.root.title("Validation")
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        self._set_window_size()
        self.root.bind("<Escape>", self._exit_fullscreen)
        self.root.bind("<plus>", lambda _e: self.adjust_zoom(1.25))
        self.root.bind("<minus>", lambda _e: self.adjust_zoom(0.8))
        self.root.bind("<Key-0>", lambda _e: self.reset_zoom())

        self.text_frame = Frame(self.root)
        self.text_frame.pack(fill="x", padx=16, pady=(12, 6))

        self.field_text = StringVar()
        self.field_label = Label(
            self.text_frame,
            textvariable=self.field_text,
            wraplength=1100,
            justify="left",
            anchor="w",
            font=("Helvetica", 14, "bold"),
        )
        self.field_label.pack(fill="x")

        self.corrected_var = StringVar()
        if self.allow_corrections:
            self.corrected_entry = Entry(self.text_frame, textvariable=self.corrected_var)
            self.corrected_entry.pack(fill="x", pady=(6, 0))
            self.corrected_var.trace_add("write", lambda *_: self._update_correct_state())
        else:
            self.corrected_entry = None

        self.canvas_frame = Frame(self.root)
        self.canvas_frame.pack(fill="both", expand=True, padx=12, pady=8)

        self.canvas = Canvas(self.canvas_frame, highlightthickness=0)
        self.h_scroll = Scrollbar(self.canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.v_scroll = Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        self.canvas.bind("<Control-MouseWheel>", self._on_zoom_wheel)
        self.canvas.bind("<Control-Button-4>", self._on_zoom_wheel)
        self.canvas.bind("<Control-Button-5>", self._on_zoom_wheel)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_move)
        try:
            self.canvas.bind("<Magnify>", self._on_magnify)
        except Exception:
            pass

        self.button_frame = Frame(self.root)
        self.button_frame.pack(fill="x", padx=16, pady=(6, 12))

        self.mark_frame = Frame(self.button_frame)
        self.mark_frame.pack(side="left")

        self.accept_button = Label(
            self.mark_frame,
            text="Accept",
            bg="#2e7d32",
            fg="white",
            padx=14,
            pady=8,
            relief="raised",
            bd=2,
            cursor="hand2",
        )
        self.accept_button.pack(side="left", padx=(0, 8))
        self.accept_button.bind("<Button-1>", lambda _e: self.on_mark("accept"))

        self.somewhat_button = Label(
            self.mark_frame,
            text="Somewhat Accept",
            bg="#7cb342",
            fg="white",
            padx=14,
            pady=8,
            relief="raised",
            bd=2,
            cursor="hand2",
        )
        self.somewhat_button.pack(side="left", padx=(0, 8))
        self.somewhat_button.bind("<Button-1>", lambda _e: self.on_mark("somewhat_accept"))

        self.reject_button = Label(
            self.mark_frame,
            text="Reject",
            bg="#c62828",
            fg="white",
            padx=14,
            pady=8,
            relief="raised",
            bd=2,
            cursor="hand2",
        )
        self.reject_button.pack(side="left")
        self.reject_button.bind("<Button-1>", lambda _e: self.on_mark("reject"))

        self.unsure_button = Label(
            self.mark_frame,
            text="Unsure",
            bg="#f9a825",
            fg="black",
            padx=14,
            pady=8,
            relief="raised",
            bd=2,
            cursor="hand2",
        )
        self.unsure_button.pack(side="left", padx=(8, 0))
        self.unsure_button.bind("<Button-1>", lambda _e: self.on_mark("unsure"))

        if self.allow_corrections:
            self.save_correction_button = Label(
                self.mark_frame,
                text="Save Correction",
                bg="#1565c0",
                fg="white",
                padx=14,
                pady=8,
                relief="raised",
                bd=2,
                cursor="hand2",
            )
            self.save_correction_button.bind("<Button-1>", lambda _e: self.on_mark("corrected"))
            self.save_correction_button.pack_forget()
        else:
            self.save_correction_button = None

        self._mark_buttons = {
            self.accept_button: "#2e7d32",
            self.somewhat_button: "#7cb342",
            self.reject_button: "#c62828",
            self.unsure_button: "#f9a825",
        }
        self.mark_enabled = True

        self.control_frame = Frame(self.button_frame)
        self.control_frame.pack(side="right")

        self.zoom_frame = Frame(self.control_frame)
        self.zoom_frame.pack(side="left", padx=(0, 12))

        self.zoom_out_button = Button(self.zoom_frame, text="−", width=3, command=lambda: self.adjust_zoom(0.8))
        self.zoom_out_button.pack(side="left")

        self.zoom_reset_button = Button(self.zoom_frame, text="Fit", width=4, command=self.reset_zoom)
        self.zoom_reset_button.pack(side="left", padx=4)

        self.zoom_in_button = Button(self.zoom_frame, text="+", width=3, command=lambda: self.adjust_zoom(1.25))
        self.zoom_in_button.pack(side="left")

        self.exit_button = Button(self.control_frame, text="Save & Exit", command=self.on_exit)
        self.exit_button.pack(side="left")

        if not self.rows:
            messagebox.showerror("No data", "Dataset is empty.")
            self.on_exit()
            return

        self.log(f"Started validation. Seed={self.seed} SamplingMode={self.sampling_mode}")
        self.next_sample()

    def _set_window_size(self) -> None:
        try:
            self.root.state("zoomed")
        except Exception:
            self.root.attributes("-fullscreen", True)

    def _exit_fullscreen(self, _event=None) -> None:
        try:
            self.root.attributes("-fullscreen", False)
        except Exception:
            pass
        self.root.state("normal")

    def _get_image_bounds(self) -> tuple[int, int]:
        self.root.update_idletasks()
        canvas_w = max(400, self.canvas.winfo_width())
        canvas_h = max(300, self.canvas.winfo_height())
        return canvas_w, canvas_h

    def _on_zoom_wheel(self, event) -> None:
        if hasattr(event, "delta") and event.delta:
            if event.delta > 0:
                self.adjust_zoom(1.25)
            else:
                self.adjust_zoom(0.8)
            return
        if getattr(event, "num", None) == 4:
            self.adjust_zoom(1.25)
        elif getattr(event, "num", None) == 5:
            self.adjust_zoom(0.8)

    def _on_mouse_wheel(self, event) -> None:
        if getattr(event, "num", None) == 4:
            self.canvas.yview_scroll(-1, "units")
            return
        if getattr(event, "num", None) == 5:
            self.canvas.yview_scroll(1, "units")
            return
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        steps = int(-delta / 120) if abs(delta) >= 120 else int(-delta)
        if steps == 0:
            steps = -1 if delta > 0 else 1
        self.canvas.yview_scroll(steps, "units")

    def _on_shift_wheel(self, event) -> None:
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        steps = int(-delta / 120) if abs(delta) >= 120 else int(-delta)
        if steps == 0:
            steps = -1 if delta > 0 else 1
        self.canvas.xview_scroll(steps, "units")

    def _on_pan_start(self, event) -> None:
        self.canvas.scan_mark(event.x, event.y)

    def _on_pan_move(self, event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_magnify(self, event) -> None:
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        factor = 1.0 + (delta / 100.0)
        self.adjust_zoom(factor)

    def _create_run_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.username}_{stamp}"
        base_dir = Path.cwd() / "validations"
        run_dir = base_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def log(self, message: str) -> None:
        stamp = datetime.now().isoformat(timespec="seconds")
        line = f"[{stamp}] {message}\n"
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line)

    def next_sample(self):
        datapoint = self._choose_next_datapoint()
        if datapoint is not None:
            self.current_datapoint = datapoint
            self.current_row = datapoint.row
            self.current_image = datapoint.image_path
            self.current_field = (datapoint.field_name, datapoint.field_value)
            self.selection_counts[datapoint.field_name] = (
                self.selection_counts.get(datapoint.field_name, 0) + 1
            )
            self.show_sample()
            return
        if len(self.validated_pairs) >= self.total_pairs:
            print("All datapoints have been validated.")
            self.log("All datapoints have been validated.")
            self.on_exit()
            return
        messagebox.showerror("No valid samples", "Unable to find a valid sample.")
        self.log("No valid samples found.")
        self.on_exit()

    def _choose_next_datapoint(self) -> ValidationDatapoint | None:
        if self.sampling_mode == "balanced_ucb":
            return choose_balanced_ucb_datapoint(
                self.datapoints,
                self.validated_pairs,
                self.selection_counts,
                self.scored_counts,
                self.score_sums,
                self.rng,
            )
        return choose_random_datapoint(self.datapoints, self.validated_pairs, self.rng)

    def show_sample(self):
        self.original_image = Image.open(self.current_image)
        self.reset_zoom()

        field_name, field_value = self.current_field
        file_name = display_image_name(self.current_row)
        self.field_text.set(f"{file_name}\n{field_name}: {field_value}")
        self.original_field_raw = field_value
        self.original_field_value = _stringify_value(field_value)
        self.corrected_var.set(self.original_field_value)
        if self.allow_corrections:
            self._update_correct_state()
        self.log(f"Showing {file_name} {field_name}")

    def reset_zoom(self) -> None:
        if not hasattr(self, "original_image") or self.original_image is None:
            return
        max_w, max_h = self._get_image_bounds()
        img_w, img_h = self.original_image.size
        fit = min(max_w / img_w, max_h / img_h, 1.0)
        self.zoom = max(0.1, fit)
        self.render_image()

    def adjust_zoom(self, factor: float) -> None:
        if not hasattr(self, "original_image") or self.original_image is None:
            return
        self.zoom = max(0.1, min(5.0, self.zoom * factor))
        self.render_image()

    def render_image(self) -> None:
        img_w, img_h = self.original_image.size
        scaled_w = max(1, int(img_w * self.zoom))
        scaled_h = max(1, int(img_h * self.zoom))
        display_img = self.original_image.resize((scaled_w, scaled_h), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(display_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, scaled_w, scaled_h))

    def on_mark(self, label: str):
        if label in {"accept", "somewhat_accept", "reject", "unsure"} and not self.mark_enabled:
            messagebox.showinfo("Correction changed", "Use Save Correction after editing the field.")
            return
        if label == "corrected" and self.mark_enabled:
            messagebox.showinfo("No changes", "Edit the field before saving a correction.")
            return
        if label == "corrected" and not self.allow_corrections:
            messagebox.showinfo("Corrections disabled", "Run with --corrections to enable edits.")
            return
        field_name, _ = self.current_field
        file_name = display_image_name(self.current_row)
        dataset_name = self.dataset_path.name
        decided_at = datetime.now().isoformat(timespec="seconds")
        corrected_field = None
        if self.allow_corrections:
            corrected_value = self.corrected_var.get().strip()
            if corrected_value == self.original_field_value:
                corrected_field = None
            else:
                try:
                    corrected_field = _parse_corrected_value(field_name, corrected_value)
                except ValueError as exc:
                    messagebox.showerror("Invalid value", str(exc))
                    return
        self.results.append(
            {
                "label": label,
                "column_name": field_name,
                "image_name": file_name,
                "file_name": file_name,
                "dataset_file": dataset_name,
                "validator_id": self.username,
                "decided_at": decided_at,
                "corrected_field": corrected_field,
                "sampling_mode": self.sampling_mode,
            }
        )
        self.validated_pairs.add((file_name, field_name))
        score = _score_for_label(label)
        if score is not None:
            self.scored_counts[field_name] = self.scored_counts.get(field_name, 0) + 1
            self.score_sums[field_name] = self.score_sums.get(field_name, 0.0) + score
        self.log(f"Marked {file_name} {field_name} label={label}")
        self.next_sample()

    def _update_correct_state(self) -> None:
        current_value = self.corrected_var.get()
        self.mark_enabled = current_value == getattr(self, "original_field_value", "")
        if self.mark_enabled:
            for button, color in self._mark_buttons.items():
                button.configure(bg=color, fg="white")
            if self.save_correction_button:
                self.save_correction_button.pack_forget()
        else:
            for button in self._mark_buttons.keys():
                button.configure(bg="#9e9e9e", fg="white")
            if self.save_correction_button and not self.save_correction_button.winfo_ismapped():
                self.save_correction_button.pack(side="left", padx=(8, 0))

    def on_exit(self):
        self.save_results()
        self.log("Exiting validation.")
        self.root.destroy()

    def save_results(self):
        if not self.results:
            return
        run_name = self.run_dir.name
        out_name = f"{run_name}_validations.csv"
        out_path = self.run_dir / out_name
        with open(out_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "label",
                    "column_name",
                    "image_name",
                    "file_name",
                    "dataset_file",
                    "validator_id",
                    "decided_at",
                    "corrected_field",
                    "sampling_mode",
                ],
            )
            writer.writeheader()
            writer.writerows(self.results)
        metadata_path = write_validation_metadata(
            run_dir=self.run_dir,
            csv_path=out_path,
            dataset_path=self.dataset_path,
            validator_id=self.username,
            decision_count=len(self.results),
            sampling_mode=self.sampling_mode,
        )
        try:
            uploaded = upload_validation_run(
                run_dir=self.run_dir,
                csv_path=out_path,
                metadata_path=metadata_path,
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"Validation upload skipped or failed: {type(exc).__name__}: {exc}")
        else:
            if uploaded:
                self.log(
                    "Uploaded validation results: "
                    f"{uploaded.get('validation_csv_uri', '')}"
                )


def main():
    parser = argparse.ArgumentParser(description="Validate transcriptions against source images.")
    parser.add_argument("--user", dest="username", default="unspecified", help="Validator username")
    parser.add_argument("--images", required=True, help="Root folder containing images")
    parser.add_argument("--results", required=True, help="Path to dataset file (.csv or .jsonl)")
    parser.add_argument("--corrections", action="store_true", help="Enable correction editing")
    parser.add_argument(
        "--sampling-mode",
        choices=("random", "balanced_ucb"),
        default="random",
        help="Datapoint sampling strategy for validation.",
    )
    args = parser.parse_args()

    username = args.username.strip()
    if not username:
        raise SystemExit("Username cannot be empty.")

    dataset_path = Path(args.results)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    image_root = Path(args.images)
    if not image_root.exists() or not image_root.is_dir():
        raise SystemExit(f"Data folder not found or not a directory: {image_root}")

    app = ValidatorApp(
        dataset_path,
        image_root,
        username=username,
        allow_corrections=args.corrections,
        sampling_mode=args.sampling_mode,
    )
    app.root.mainloop()


if __name__ == "__main__":
    main()
