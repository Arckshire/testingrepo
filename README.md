# FTL In-Transit Time Calculator

This Streamlit app processes a raw FTL tracking file (CSV or Excel) and calculates in-transit time at shipment / lane level.

It:

- Splits shipments into **Tracked** vs **Untracked** using a boolean `Tracked` column.
- Within tracked shipments, identifies **missed milestones** when:
  - Pickup or drop-off timestamps are missing/blank/invalid, or  
  - The calculated transit duration is non-positive (arrival earlier than or equal to departure).
- For valid tracked shipments, computes in-transit time in **days** with custom rounding rules.
- Generates a processed output file (Excel + CSV) with:
  - A **summary table** (tracked, missed milestone, untracked, grand total).
  - A **detailed table** listing only tracked shipments with valid in-transit time.

All important column names are defined as constants at the top of `app.py` so you can easily adjust them.

---

## How to run this app on Streamlit Community Cloud

You don't need any local Python setup. Everything can be done via the browser.

### 1. Put the code on GitHub

1. Go to [GitHub](https://github.com) and sign in (or create an account).
2. Click **New** repository.
3. Name it something like `ftl-intransit-app`.
4. Set **Visibility** to **Public**.
5. Click **Create repository**.
6. In the new repo, click **Add file → Upload files**.
7. Upload these three files:
   - `app.py`
   - `requirements.txt`
   - `README.md`
8. Scroll down and click **Commit changes** to save to the **main** branch.

### 2. Deploy on Streamlit Community Cloud

1. Go to [Streamlit Community Cloud](https://share.streamlit.io) and sign in with your **GitHub** account.
2. Click **New app**.
3. Under **Repository**, choose your `ftl-intransit-app` repo.
4. Set:
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **Deploy**.

After it builds, your app will open in the browser with a shareable URL.

### 3. Updating the app

- To change logic or column names, edit `app.py` directly on GitHub (via the web editor).
- Click **Commit changes** to `main`.
- Streamlit will automatically redeploy the updated app.
