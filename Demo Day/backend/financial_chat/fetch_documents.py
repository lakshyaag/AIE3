import re
from typing import List
from financial_chat.schema import FilingItem, ReportParams
from edgar import Company, CompanyFiling, CompanyFilings, set_identity

set_identity("lazy@lazyg.com")
valid_items = [item.value for item in FilingItem]


def filter_filings(filings: CompanyFilings, year: int) -> CompanyFiling:
    """
    Filter filings by form and year.

    Args:
        filings (CompanyFilings): List of EntityFilings.
        year (int): The year to filter by.

    Returns:
        CompanyFiling: The filtered EntityFiling.

    Raises:
        ValueError: If no filing is found for the specified year.
    """
    for filing in filings:
        year_of_filing = filing.filing_date.year
        if year_of_filing == year:
            return filing

    raise ValueError(f"No filing found for year {year}.")


def get_items(filing: CompanyFiling, item_names: List[FilingItem] = None) -> dict:
    """
    Get the items from a filing. If no items are specified, all items are returned.

    Args:
        filing (CompanyFiling): The filing to get items from.
        item_names (List[FilingItem], optional): The items to get. Defaults to None.

    Returns:
        dict: The items from the filing.

    Raises:
        ValueError: If an item is not found in the filing.
    """
    items = (
        {} if item_names is not None else {item: filing[item] for item in filing.items}
    )

    if item_names:
        for item in item_names:
            matched_items = [
                filing_item
                for filing_item in filing.items
                if filing_item.lower() == item.value.lower()
            ]
            if matched_items:
                items[item.value] = filing[matched_items[0]]
            else:
                raise ValueError(f"Item '{item.value}' not found in filing.")

    # Remove any newline characters, trim outer whitespace, and remove unwanted patterns
    pattern1 = r"-{3,}|\.{3,}"
    pattern2 = r"\+{2,}"

    cleaned_items = {}

    for key, value in items.items():
        cleaned_value = re.sub(
            pattern2, "", re.sub(pattern1, "", value.replace("\n", " ").strip())
        )
        cleaned_items[key] = cleaned_value

    return cleaned_items


def get_company_filing(
    params: ReportParams,
) -> dict:
    """
    Get a company's filing information based on parameters.

    Args:
        params (ReportParams): The parameters to get the filing information for.

    Returns:
        dict: The items from the company's filing.
    """

    if params.item_names and not all(
        isinstance(item, FilingItem) for item in params.item_names
    ):
        raise ValueError("All items must be of type FilingItem.")

    company = Company(params.ticker)

    if company is None:
        raise ValueError(f"Company with ticker '{params.ticker}' not found.")

    filings = company.get_filings(form=params.form)
    filing = filter_filings(filings, params.year).obj()
    items = get_items(filing, params.item_names)

    information = {
        "ticker": params.ticker,
        "year": params.year,
        "form": params.form,
        "items": items,
    }

    return information


if __name__ == "__main__":
    params = ReportParams(
        ticker="AAPL", year=2020, form="10-K", item_names=["Item 1", "Item 7"]
    )

    info = get_company_filing(params)
    print(info)
