import pytest
from financial_chat.fetch_documents import get_company_filing
from financial_chat.chunk_load import (
    chunk_items,
    load_documents,
    check_if_documents_exist,
)
from financial_chat.types_ import ReportParams


def test_chunk_valid_filing():
    params = ReportParams(
        ticker="AAPL", year=2020, form="10-K", item_names=["Item 1", "Item 2", "Item 7"]
    )

    result = get_company_filing(params)

    documents = chunk_items(result["items"], params)

    assert len(documents) > 0


def test_chunk_invalid_filing():
    with pytest.raises(ValueError):
        params = ReportParams(
            ticker="INVALID",
            year=2020,
            form="10-K",
            item_names=["Item 1", "Item 2", "Item 7"],
        )
        result = get_company_filing(params)

        documents = chunk_items(result["items"], params)

        assert len(documents) == 0


def test_add_documents():
    params = ReportParams(
        ticker="COST",
        year=2022,
        form="10-K",
        item_names=["Item 1", "Item 2", "Item 7"],
    )

    result = get_company_filing(params)

    documents = chunk_items(result["items"], params)

    assert len(documents) > 0

    load_documents(documents)


def test_add_documents_invalid():
    with pytest.raises(ValueError):
        params = ReportParams(
            ticker="INVALID",
            year=2020,
            form="10-K",
            item_names=["Item 1", "Item 2", "Item 7"],
        )
        result = get_company_filing(params)

        documents = chunk_items(result["items"], params)

        assert len(documents) == 0

        load_documents(documents)


def test_check_documents():
    params = ReportParams(
        ticker="COST",
        year=2022,
        form="10-K",
        item_names=["Item 1", "Item 2", "Item 7"],
    )

    rows = check_if_documents_exist(params)

    assert rows is True


def test_check_documents_2():
    params = ReportParams(
        ticker="COST",
        year=2022,
        form="10-K",
        item_names=["Item 6"],
    )

    rows = check_if_documents_exist(params)

    assert rows is False
