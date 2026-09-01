from patientjournals.tasks import module_command, prepare_ocr, verify_batch


def test_module_command_quotes_arguments() -> None:
    command = module_command(
        "patientjournals.batch.submit",
        ["--run-dir", "runs/submit with space"],
    )

    assert command == "python -m patientjournals.batch.submit --run-dir 'runs/submit with space'"


def test_batch_ocr_task_builds_cloud_preparation_command() -> None:
    commands: list[str] = []

    class Context:
        def run(self, command: str, *, pty: bool) -> None:
            assert pty is True
            commands.append(command)

    prepare_ocr.body(
        Context(),
        workers=4,
        api_batch_size=8,
        force=True,
        limit=12,
        allow_failures=True,
    )

    assert commands == [
        "python -m patientjournals.batch.prepare_ocr "
        "--workers 4 --api-batch-size 8 --force --limit 12 --allow-failures"
    ]


def test_batch_verify_task_builds_submit_and_retrieve_options() -> None:
    commands: list[str] = []

    class Context:
        def run(self, command: str, *, pty: bool) -> None:
            assert pty is True
            commands.append(command)

    verify_batch.body(
        Context(),
        retrieve=True,
        source_run_dir="runs/retrieves/source run",
        run_dir="runs/verifications/verify run",
        model="gemini-3.1-pro-preview",
        thinking_level="high",
        max_output_tokens=8192,
        num_chunks=3,
        wait=True,
    )

    assert commands == [
        "python -m patientjournals.batch.verify --retrieve "
        "--source-run-dir 'runs/retrieves/source run' "
        "--run-dir 'runs/verifications/verify run' "
        "--model gemini-3.1-pro-preview --thinking-level high "
        "--max-output-tokens 8192 --num-chunks 3 --wait"
    ]
