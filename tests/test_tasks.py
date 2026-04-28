from patientjournals.tasks import module_command


def test_module_command_quotes_arguments() -> None:
    command = module_command(
        "patientjournals.batch.submit",
        ["--run-dir", "runs/submit with space"],
    )

    assert command == "python -m patientjournals.batch.submit --run-dir 'runs/submit with space'"

