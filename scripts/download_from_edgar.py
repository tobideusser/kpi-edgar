import csv
import logging
from datetime import date
from time import sleep

from secedgar import filings, FilingType
from tqdm import tqdm

logger = logging.getLogger(__name__)

USER_AGENT = "Tobias Deusser (tobias.deusser@iais.fraunhofer.de)"
TICKER_FILE = "../data/220120_nasdaq_ticker_above$200B.csv"
COMPANY_TICKER = [
    "aapl",  # Apple
    "msft",  # Microsoft
    "googl",  # Google
    "fb",  # Facebook
    "amzn",  # Amazon
    "tsla",  # Tesla
    "wmt",  # Walmart
    "pfe",  # Pfizer
    "jpm",  # JP Morgan Chase
    "f"  # Ford
]
NUMBER_REPORTS = 20
PATH = "/cluster/edgar_filings_above$200B"
START_DATE = date(2020, 1, 1)


def main():
    if TICKER_FILE:
        company_ticker = []
        with open(TICKER_FILE, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                company_ticker.append(row["Symbol"])
    else:
        company_ticker = COMPANY_TICKER

    for ticker in tqdm(company_ticker, desc="Downloading reports"):
        try:
            filing = filings(
                cik_lookup=ticker,
                start_date=START_DATE,
                filing_type=FilingType.FILING_10K,
                count=NUMBER_REPORTS,
                user_agent=USER_AGENT
            )
            filing.save(PATH)
        except ValueError:
            logger.warning(f"No filings for {ticker} available.")
        sleep(1)


if __name__ == "__main__":
    main()
