from __future__ import annotations

import json
import shlex
from dataclasses import asdict, is_dataclass
from pathlib import Path

from invoke import Collection, task

import patientjournals.config.settings as config_module
from patientjournals.config import config


def module_command(module: str, args: list[str] | None = None) -> str:
    command = ["python", "-m", module]
    command.extend(args or [])
    return " ".join(shlex.quote(part) for part in command)


def _split_extra(extra: str | None) -> list[str]:
    if not extra:
        return []
    return shlex.split(extra)


def _add_option(args: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        args.extend([flag, text])


def _add_flag(args: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        args.append(flag)


def _run_module(context, module: str, args: list[str]) -> None:
    context.run(module_command(module, args), pty=True)


@task(name="batch")
def data_batch(
    context,
    summary: bool = False,
    validate: bool = False,
    root: str | None = None,
    bucket: bool = False,
    bucket_name: str | None = None,
    prefix: str | None = None,
    glob: str | None = None,
    no_recursive: bool = False,
    summaries_dir: str | None = None,
    validations_dir: str | None = None,
    allow_failures: bool = False,
    extra: str = "",
) -> None:
    args: list[str] = []
    _add_flag(args, "--summary", summary)
    _add_flag(args, "--validate", validate)
    _add_option(args, "--root", root)
    _add_flag(args, "--bucket", bucket)
    _add_option(args, "--bucket-name", bucket_name)
    _add_option(args, "--prefix", prefix)
    _add_option(args, "--glob", glob)
    _add_flag(args, "--no-recursive", no_recursive)
    _add_option(args, "--summaries-dir", summaries_dir)
    _add_option(args, "--validations-dir", validations_dir)
    _add_flag(args, "--allow-failures", allow_failures)
    args.extend(_split_extra(extra))
    _run_module(context, "patientjournals.data.batch", args)


@task(name="run")
def local_run(
    context,
    data_folder: str | None = None,
    continue_dataset: str | None = None,
    verbose: bool = False,
    extra: str = "",
) -> None:
    args: list[str] = []
    _add_option(args, "--data-folder", data_folder)
    _add_option(args, "--continue-dataset", continue_dataset)
    _add_flag(args, "--verbose", verbose)
    args.extend(_split_extra(extra))
    _run_module(context, "patientjournals.local.cli", args)


@task
def upload(context, extra: str = "") -> None:
    _run_module(context, "patientjournals.batch.upload", _split_extra(extra))


@task
def submit(
    context,
    num_batches: int | None = None,
    rerun: bool = False,
    run_dir: str | None = None,
    downscale: float | None = None,
    extra: str = "",
) -> None:
    args: list[str] = []
    _add_option(args, "--num-batches", num_batches)
    _add_flag(args, "--rerun", rerun)
    _add_option(args, "--run-dir", run_dir)
    _add_option(args, "--downscale", downscale)
    args.extend(_split_extra(extra))
    _run_module(context, "patientjournals.batch.submit", args)


@task
def status(
    context,
    batch_name: str | None = None,
    run_dir: str | None = None,
    watch: bool = False,
    cancel: bool = False,
    simple: bool = False,
    extra: str = "",
) -> None:
    args: list[str] = []
    _add_option(args, "--batch-name", batch_name)
    _add_option(args, "--run-dir", run_dir)
    _add_flag(args, "--watch", watch)
    _add_flag(args, "--cancel", cancel)
    _add_flag(args, "--simple", simple)
    args.extend(_split_extra(extra))
    _run_module(context, "patientjournals.batch.status", args)


@task
def retrieve(
    context,
    batch_name: str | None = None,
    run_dir: str | None = None,
    wait: bool = False,
    allow_partial: bool = False,
    submit_failed: bool = False,
    extra: str = "",
) -> None:
    args: list[str] = []
    _add_option(args, "--batch-name", batch_name)
    _add_option(args, "--run-dir", run_dir)
    _add_flag(args, "--wait", wait)
    _add_flag(args, "--allow-partial", allow_partial)
    _add_flag(args, "--submit-failed", submit_failed)
    args.extend(_split_extra(extra))
    _run_module(context, "patientjournals.batch.retrieve", args)


@task(name="check-models")
def check_models(
    context,
    backend: str | None = None,
    project: str | None = None,
    location: str | None = None,
    contains: str | None = None,
    limit: int | None = None,
    show_actions: bool = False,
    check: str | None = None,
    extra: str = "",
) -> None:
    args: list[str] = []
    _add_option(args, "--backend", backend)
    _add_option(args, "--project", project)
    _add_option(args, "--location", location)
    _add_option(args, "--contains", contains)
    _add_option(args, "--limit", limit)
    _add_flag(args, "--show-actions", show_actions)
    _add_option(args, "--check", check)
    args.extend(_split_extra(extra))
    _run_module(context, "patientjournals.batch.check_models", args)


@task
def validate(
    context,
    images: str | None = None,
    results: str | None = None,
    user: str = "unspecified",
    corrections: bool = False,
    extra: str = "",
) -> None:
    args: list[str] = []
    _add_option(args, "--user", user)
    _add_option(args, "--images", images)
    _add_option(args, "--results", results)
    _add_flag(args, "--corrections", corrections)
    args.extend(_split_extra(extra))
    _run_module(context, "patientjournals.validation.cli", args)


@task(name="report")
def validation_report(
    context,
    input_path: str = "validations",
    out: str = "validation_reports",
    min_n: int = 1,
    extra: str = "",
) -> None:
    args: list[str] = []
    _add_option(args, "--input", input_path)
    _add_option(args, "--out", out)
    _add_option(args, "--min-n", min_n)
    args.extend(_split_extra(extra))
    _run_module(context, "patientjournals.validation.analysis", args)


@task(name="path")
def config_path(_context) -> None:
    print(Path(config_module.__file__).resolve())


@task(name="show")
def config_show(_context) -> None:
    payload = asdict(config) if is_dataclass(config) else {}
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


local = Collection("local")
local.add_task(local_run, "run")

data = Collection("data")
data.add_task(data_batch, "batch")

batch = Collection("batch")
batch.add_task(upload)
batch.add_task(submit)
batch.add_task(status)
batch.add_task(retrieve)
batch.add_task(check_models)

validation = Collection("validation")
validation.add_task(validate)
validation.add_task(validation_report)

config_tasks = Collection("config")
config_tasks.add_task(config_path)
config_tasks.add_task(config_show)

ns = Collection()
ns.add_collection(data)
ns.add_collection(local)
ns.add_collection(batch)
ns.add_collection(validation)
ns.add_collection(config_tasks)
