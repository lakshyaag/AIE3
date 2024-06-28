import pytest
from financial_chat.fetch_documents import get_company_filing
from financial_chat.types_ import ReportParams


def test_get_company_filing_valid():
    # Test with valid inputs

    params = ReportParams(
        ticker="AAPL", year=2020, form="10-K", item_names=["Item 1", "Item 7", "Item 2"]
    )

    result = get_company_filing(params)
    assert result["ticker"] == "AAPL"
    assert result["year"] == 2020
    assert "Item 1" in result["items"]
    assert "Item 7" in result["items"]
    assert "Item 2" in result["items"]

    assert len(result["items"]) == 3


def test_get_company_filing_invalid_ticker():
    # Test with an invalid ticker
    with pytest.raises(ValueError):
        get_company_filing(
            ReportParams(
                ticker="INVALID", year=2020, form="10-K", item_names=["Item 1"]
            )
        )


def test_get_company_filing_invalid_year():
    # Test with a year that has no filings
    with pytest.raises(ValueError):
        get_company_filing(
            ReportParams(
                ticker="INVALID", year=1900, form="10-K", item_names=["Item 1"]
            )
        )


def test_get_company_filing_invalid_item():
    # Test with an item that does not exist
    with pytest.raises(ValueError):
        get_company_filing(
            ReportParams(
                ticker="AAPL", year=2020, form="10-K", item_names=["Nonexistent Item"]
            )
        )


def test_get_company_filing_invalid_item_type():
    # Test with an invalid item type
    with pytest.raises(ValueError):
        get_company_filing(
            ReportParams(
                ticker="AAPL", year=2020, form="10-K", item_names=["Item 1", 123]
            )
        )
