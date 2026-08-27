from patientjournals.tasks import module_command, prepare_ocr


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
        force=True,
        limit=12,
        allow_failures=True,
    )

    assert commands == [
        "python -m patientjournals.batch.prepare_ocr "
        "--workers 4 --force --limit 12 --allow-failures"
    ]
