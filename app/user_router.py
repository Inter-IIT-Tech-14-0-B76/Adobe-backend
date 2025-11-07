from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pymongo.collection import Collection
from typing import Dict, List, Optional
from app.db import users_collection
from app.helpers import verify_firebase_token, get_uid, get_portfolio_nav_history
from app.bigquery import run_bigquery


user_router = APIRouter()
uid = "0520MubIFDW59HdFnvBcv5Iv0T03"


@user_router.get("/portfolio/summary")
def get_portfolio_summary(uid: str = Depends(get_uid)):
    """
    Fetches every fund in the user's portfolio in one BigQuery call:
      1) initial_nav as of dateAdded
      2) latest_nav overall
      3) schemeName
    Then computes values and returns an aggregated summary.
    """
    # uid = "0520MubIFDW59HdFnvBcv5Iv0T03"
    project_id = "stock-market-462609"
    dataset_id = "stock_data"

    # Table for NAV history
    table_name = "nav_history"
    TABLE_ID = f"{project_id}.{dataset_id}.{table_name}"

    # Table for fund details (like schemeName)
    detailed_table_name = "Mutual_Fund_Detailed"
    DETAILED_TABLE_ID = f"{project_id}.{dataset_id}.{detailed_table_name}"

    # Get user data
    user = users_collection.find_one({"_id": uid})
    if not user:
        return JSONResponse(status_code=404, content={"message": "User not found"})
    funds: List[Dict] = user.get("portfolio", {}).get("funds", [])
    if not funds:
        return {
            "total_value": 0.0,
            "total_gain": 0.0,
            "fund_count": 0,
            "funds": [],
        }

    struct_clauses = ",\n    ".join(
        f"STRUCT('{f['schemeCode']}' AS scheme_code, DATE '{f['date']}' AS date_added)"
        for f in funds
        if f.get("date")
    )

    # 2) One BigQuery call using JOIN + ROW_NUMBER() to de-correlate
    # Also joins with detailed table to get scheme_name_unique
    query = f"""
    WITH fund_list AS (
    SELECT * FROM UNNEST([
        {struct_clauses}
    ]) AS f
    ),

    initial_navs AS (
    SELECT
        f.scheme_code,
        f.date_added,
        nh.nav_value AS initial_nav,
        ROW_NUMBER() OVER(
        PARTITION BY f.scheme_code, f.date_added
        ORDER BY nh.date DESC
        ) AS rn
    FROM fund_list AS f
    JOIN `{TABLE_ID}` AS nh
        ON nh.scheme_code = f.scheme_code
    AND DATE(nh.date) <= f.date_added
    ),

    latest_navs AS (
    SELECT
        f.scheme_code,
        nh.nav_value AS latest_nav,
        ROW_NUMBER() OVER(
        PARTITION BY f.scheme_code
        ORDER BY nh.date DESC
        ) AS rn
    FROM fund_list AS f
    JOIN `{TABLE_ID}` AS nh
        ON nh.scheme_code = f.scheme_code
    )

    SELECT
    f.scheme_code,
    f.date_added,
    i.initial_nav,
    l.latest_nav,
    mfd.scheme_name_unique
    FROM fund_list AS f

    LEFT JOIN initial_navs AS i
    ON f.scheme_code = i.scheme_code
    AND f.date_added = i.date_added
    AND i.rn = 1

    LEFT JOIN latest_navs AS l
    ON f.scheme_code = l.scheme_code
    AND l.rn = 1

    LEFT JOIN `{DETAILED_TABLE_ID}` AS mfd
    ON f.scheme_code = mfd.scheme_code
    """

    print(query)

    rows, err = run_bigquery(query=query)
    if err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"BigQuery error: {err}",
        )

    # Map results by schemeCode
    nav_map = {
        r["scheme_code"]: {
            "initial_nav": float(r["initial_nav"]),
            "latest_nav": float(r["latest_nav"]),
            "scheme_name": r["scheme_name_unique"],
        }
        for r in rows
        if r.get("initial_nav") is not None
        and r.get("latest_nav") is not None
        and r.get("scheme_name_unique")
    }

    total_initial = 0.0
    total_current = 0.0
    detailed: List[Dict] = []

    # Compute per-fund performance
    for fund in funds:
        scheme = fund.get("schemeCode")
        navs = nav_map.get(scheme)
        if not navs:
            continue

        initial_nav = navs["initial_nav"]
        latest_nav = navs["latest_nav"]
        scheme_name = navs["scheme_name"]

        units: Optional[float] = fund.get("units")
        invested_amount: Optional[float] = fund.get("amountInvested")

        # Derive units if not stored
        if units is None and invested_amount is not None:
            units = invested_amount / initial_nav
        if units is None:
            continue

        initial_value = units * initial_nav
        current_value = units * latest_nav
        gain_value = current_value - initial_value
        gain_percent = (gain_value / initial_value) * 100 if initial_value else 0

        total_initial += initial_value
        total_current += current_value

        detailed.append(
            {
                "schemeCode": scheme,
                "schemeName": scheme_name,
                "dateAdded": fund.get("date"),
                "units": round(units, 4),
                "purchaseNav": round(initial_nav, 4),
                "latestNav": round(latest_nav, 4),
                "initialValue": round(initial_value, 2),
                "currentValue": round(current_value, 2),
                "gainValue": round(gain_value, 2),
                "gainPercent": round(gain_percent, 2),
            }
        )

    total_gain = total_current - total_initial

    return {
        "total_value": round(total_current, 2),
        "total_gain": round(total_gain, 2),
        "fund_count": len(detailed),
        "funds": detailed,
    }


def get_uid(token_payload: Dict = Depends(verify_firebase_token)) -> str:
    return token_payload["uid"]


@user_router.post("/portfolio/add")
def add_portfolio_entry(entry: dict, uid: str = Depends(get_uid)):
    existing_user = users_collection.find_one({"_id": uid})

    if not existing_user:
        users_collection.insert_one(
            {"_id": uid, "portfolio": {"funds": [entry]}, "watchlist": []}
        )
        return {"message": "Portfolio created and fund added"}

    existing_funds = existing_user.get("portfolio", {}).get("funds", [])
    if any(fund["schemeCode"] == entry["schemeCode"] for fund in existing_funds):
        print("Scheme Code already added.", existing_funds, entry)
        raise HTTPException(status_code=400, detail="Fund already exists in portfolio")

    try:
        users_collection.update_one({"_id": uid}, {"$push": {"portfolio.funds": entry}})
    except Exception as e:
        print(f"[DB Error] Failed to add fund to portfolio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )
    return {"message": "Fund added to portfolio"}


@user_router.post("/watchlist/add")
def add_watchlist_entry(entry: dict, uid: str = Depends(get_uid)):
    existing_user = users_collection.find_one({"_id": uid})

    if not existing_user:
        users_collection.insert_one(
            {"_id": uid, "portfolio": {"funds": []}, "watchlist": [entry]}
        )
        return {"message": "Watchlist created and fund added"}

    existing_watchlist = existing_user.get("watchlist", [])
    if any(watch["schemeCode"] == entry["schemeCode"] for watch in existing_watchlist):
        raise HTTPException(status_code=400, detail="Fund already exists in watchlist")

    users_collection.update_one({"_id": uid}, {"$push": {"watchlist": entry}})
    return {"message": "Fund added to watchlist"}


@user_router.get("/data")
def get_user_data(uid: str = Depends(get_uid)):
    user = users_collection.find_one({"_id": uid}, {"_id": 0})
    return user or {"portfolio": {"funds": []}, "watchlist": []}


@user_router.get("/portfolio/history", response_model=List[Dict])
async def portfolio_nav_history_route(
    portfolio_history: List[Dict] = Depends(get_portfolio_nav_history),
):
    """
    Returns the historical daily value of the user's entire portfolio.
    Each point in the list is a dictionary containing a date and the
    total portfolio value on that day.
    """
    return portfolio_history
