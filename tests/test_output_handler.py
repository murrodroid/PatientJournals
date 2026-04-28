from patientjournals.config.schemas import PageLine, TextPage
from patientjournals.shared.output_handler import data_to_rows


def test_text_page_expands_to_one_row_per_line() -> None:
    page = TextPage(
        page_lines=[
            PageLine(text="first", metadata="1/1"),
            PageLine(text="second", metadata=None),
        ]
    )

    rows = data_to_rows(page, "page_0001.png")

    assert rows == [
        {
            "text": "first",
            "metadata": "1/1",
            "page_line_number": 1,
            "file_name": "page_0001.png",
        },
        {
            "text": "second",
            "metadata": None,
            "page_line_number": 2,
            "file_name": "page_0001.png",
        },
    ]


def test_text_page_preserves_field_confidence_by_line() -> None:
    page = TextPage(page_lines=[PageLine(text="first", metadata="1/1")])

    rows = data_to_rows(
        page,
        "page_0001.png",
        field_confidence_by_pointer={
            "/page_lines/0/text": {
                "field_confidence_logprobs": -0.1,
                "field_confidence_ratio": 0.9,
            }
        },
    )

    assert rows[0]["field_confidence"]["text"] == {
        "field_confidence_logprobs": -0.1,
        "field_confidence_ratio": 0.9,
    }

