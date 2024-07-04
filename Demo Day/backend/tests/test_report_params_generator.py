import pytest
from financial_chat.nodes import report_params_generator
from financial_chat.schema import ReportParams


def test_report_params_generator():
    # Test with valid inputs

    response = report_params_generator.chain.invoke(
        {"input": "What is Apple's management discussion summary for 2021?"}
    )

    params = ReportParams(ticker="AAPL", year=2021, form="10-K", item_names=["Item 7"])
    assert response == params

def test_report_params_generator_valid_multiple_items():
    # Test with valid inputs and multiple items

    response = report_params_generator.chain.invoke(
        {"input": "What are the items 1, 2, and 7 in Microsoft's 2020 10-K report?"}
    )

    params = ReportParams(
        ticker="MSFT",
        year=2020,
        form="10-K",
        item_names=["Item 1", "Item 2", "Item 7"],
    )
    assert response == params

def test_report_params_generator_invalid_ticker():
    # Test with an invalid ticker

    with pytest.raises(ValueError):
        report_params_generator.chain.invoke(
            {
                "input": "What is the management discussion summary for INVALID in 2021?"
            }
        )
