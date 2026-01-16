import argparse
import csv
import json
import random
import secrets
from datetime import datetime
from pathlib import Path
from tkinter import Tk, Label, Button, StringVar, messagebox, Frame, Canvas, Scrollbar, Entry
from typing import get_args, get_origin

import pandas as pd
from PIL import Image, ImageTk
from pydantic import BaseModel, TypeAdapter, ValidationError

from schemas import Journal

_COLUMNS_NOT_INCLUDED = ['generation_seconds','file_name']

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


def pick_flat_field(row: dict, rng: random.Random) -> tuple[str, object] | None:
    flat = flatten_row(row)
    candidates = []
    for key, value in flat.items():
        if key in _COLUMNS_NOT_INCLUDED:
            continue
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        if isinstance(value, (dict, list)):
            continue
        candidates.append((key, value))
    if not candidates:
        return None
    return rng.choice(candidates)

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

def _get_field_type(path: str) -> object | None:
    current = Journal
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
    file_name = row.get("file_name")
    if not file_name:
        return None
    return image_index.get(Path(file_name).name)


class ValidatorApp:
    def __init__(self, dataset_path: Path, image_root: Path, username: str):
        self.dataset_path = dataset_path
        self.image_root = image_root
        self.username = username
        self.seed = secrets.randbits(64)
        self.rng = random.Random(self.seed)
        self.rows = load_dataset(dataset_path)
        self.image_index = build_image_index(image_root)
        self.results: list[dict] = []
        self.validated_pairs: set[tuple[str, str]] = set()
        self.total_pairs = self._count_total_pairs()
        self.current_row = None
        self.current_field = None
        self.current_image = None
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
        self.corrected_entry = Entry(self.text_frame, textvariable=self.corrected_var)
        self.corrected_entry.pack(fill="x", pady=(6, 0))
        self.corrected_var.trace_add("write", lambda *_: self._update_correct_state())

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

        self._mark_buttons = {
            self.accept_button: "#2e7d32",
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

        self.log(f"Started validation. Seed={self.seed}")
        self.next_sample()

    def _count_total_pairs(self) -> int:
        total = 0
        for row in self.rows:
            file_name = Path(row.get("file_name", "")).name
            if not file_name:
                continue
            flat = flatten_row(row)
            for key, value in flat.items():
                if key in _COLUMNS_NOT_INCLUDED:
                    continue
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    continue
                if isinstance(value, (dict, list)):
                    continue
                total += 1
        return total

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
        for _ in range(1000):
            row = self.rng.choice(self.rows)
            image_path = resolve_image_path(row, self.image_index)
            field = pick_flat_field(row, self.rng)
            if image_path and field:
                file_name = Path(row.get("file_name", "")).name
                pair = (file_name, field[0])
                if pair in self.validated_pairs:
                    continue
                self.current_row = row
                self.current_image = image_path
                self.current_field = field
                self.show_sample()
                return
        if len(self.validated_pairs) >= self.total_pairs:
            print("All datapoints have been validated.")
            self.log("All datapoints have been validated.")
            self.on_exit()
            return
        messagebox.showerror("No valid samples", "Unable to find a valid sample.")
        self.log("No valid samples found after 1000 attempts.")
        self.on_exit()

    def show_sample(self):
        self.original_image = Image.open(self.current_image)
        self.reset_zoom()

        field_name, field_value = self.current_field
        file_name = Path(self.current_row.get("file_name", "")).name
        self.field_text.set(f"{file_name}\n{field_name}: {field_value}")
        self.original_field_raw = field_value
        self.original_field_value = _stringify_value(field_value)
        self.corrected_var.set(self.original_field_value)
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
        if label in {"accept", "reject", "unsure"} and not self.mark_enabled:
            messagebox.showinfo("Correction changed", "Use Save Correction after editing the field.")
            return
        if label == "corrected" and self.mark_enabled:
            messagebox.showinfo("No changes", "Edit the field before saving a correction.")
            return
        field_name, _ = self.current_field
        file_name = Path(self.current_row.get("file_name", "")).name
        dataset_name = self.dataset_path.name
        decided_at = datetime.now().isoformat(timespec="seconds")
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
                "file_name": file_name,
                "dataset_file": dataset_name,
                "validator_id": self.username,
                "decided_at": decided_at,
                "corrected_field": corrected_field,
            }
        )
        self.validated_pairs.add((file_name, field_name))
        self.log(f"Marked {file_name} {field_name} label={label}")
        self.next_sample()

    def _update_correct_state(self) -> None:
        current_value = self.corrected_var.get()
        self.mark_enabled = current_value == getattr(self, "original_field_value", "")
        if self.mark_enabled:
            for button, color in self._mark_buttons.items():
                button.configure(bg=color, fg="white")
            self.save_correction_button.pack_forget()
        else:
            for button in self._mark_buttons.keys():
                button.configure(bg="#9e9e9e", fg="white")
            if not self.save_correction_button.winfo_ismapped():
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
                    "file_name",
                    "dataset_file",
                    "validator_id",
                    "decided_at",
                    "corrected_field",
                ],
            )
            writer.writeheader()
            writer.writerows(self.results)


def main():
    parser = argparse.ArgumentParser(description="Validate transcriptions against source images.")
    parser.add_argument("--user", dest="username", default="unspecified", help="Validator username")
    parser.add_argument("--images", required=True, help="Root folder containing images")
    parser.add_argument("--results", required=True, help="Path to dataset file (.csv or .jsonl)")
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

    app = ValidatorApp(dataset_path, image_root, username=username)
    app.root.mainloop()


if __name__ == "__main__":
    main()
