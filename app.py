import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO, StringIO

# ====== CONFIGURABLE COLUMN NAMES (EASY TO CHANGE) ======

# Raw input columns
TRACKED_COL = "Tracked"
PICKUP_TS_COL = "Pickup Departure UTC Timestamp Raw"
DROPOFF_TS_COL = "Dropoff Arrival UTC Timestamp Raw"  # "Dropoff" as one word
RAW_PICKUP_CITY_STATE_COL = "Pickup City State"
RAW_DROPOFF_CITY_STATE_COL = "Dropoff City State"

BILL_OF_LADING_COL = "Bill of lading"
PICKUP_NAME_COL = "Pickup name"
PICKUP_COUNTRY_COL = "Pickup country"
DROPOFF_NAME_COL = "Drop-off name"
DROPOFF_COUNTRY_COL = "Drop-off country"

# Derived/output columns
PICKUP_CITY_COL = "Pickup city"
PICKUP_STATE_COL = "Pickup state"
DROPOFF_CITY_COL = "Drop-off city"
DROPOFF_STATE_COL = "Drop-off state"

IN_TRANSIT_OUTPUT_COL = "In transit time"


# ====== COLUMN STANDARDIZATION (TOLERANT MATCHING) ======

def _normalize_col_name(name: str) -> str:
    """Lowercase, strip, collapse multiple spaces for matching."""
    if name is None:
        return ""
    return " ".join(str(name).strip().lower().split())


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map columns in the uploaded file to our expected constant names
    using a case-insensitive, space-insensitive match.
    If a match is found, rename the column in df to the constant name.
    """
    df = df.copy()

    # Map normalized existing names -> original names
    existing_norm = {_normalize_col_name(c): c for c in df.columns}

    # All required / used raw columns
    expected_cols = [
        TRACKED_COL,
        PICKUP_TS_COL,
        DROPOFF_TS_COL,
        BILL_OF_LADING_COL,
        PICKUP_NAME_COL,
        PICKUP_COUNTRY_COL,
        DROPOFF_NAME_COL,
        DROPOFF_COUNTRY_COL,
        RAW_PICKUP_CITY_STATE_COL,
        RAW_DROPOFF_CITY_STATE_COL,
    ]

    rename_map = {}
    for expected in expected_cols:
        norm_expected = _normalize_col_name(expected)
        if norm_expected in existing_norm:
            original_name = existing_norm[norm_expected]
            rename_map[original_name] = expected

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def validate_columns(df: pd.DataFrame):
    required_cols = [
        TRACKED_COL,
        PICKUP_TS_COL,
        DROPOFF_TS_COL,
        BILL_OF_LADING_COL,
        PICKUP_NAME_COL,
        PICKUP_COUNTRY_COL,
        DROPOFF_NAME_COL,
        DROPOFF_COUNTRY_COL,
        RAW_PICKUP_CITY_STATE_COL,
        RAW_DROPOFF_CITY_STATE_COL,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"The following required columns are missing from the uploaded file: {', '.join(missing)}"
        )


# ====== CORE LOGIC ======

def get_tracked_untracked_masks(df: pd.DataFrame):
    """
    Use the TRACKED_COL to build boolean masks for tracked and untracked.
    Assumes values TRUE/FALSE, but also robust to 'True'/'False' strings and actual booleans.
    """
    tracked_series = df[TRACKED_COL].astype(str).str.upper().str.strip()
    tracked_mask = tracked_series == "TRUE"
    untracked_mask = tracked_series == "FALSE"
    return tracked_mask, untracked_mask


def split_city_state(series: pd.Series):
    """
    Split 'City - State' or 'City-State' into two Series: city, state.
    - Before '-' -> city
    - After '-'  -> state
    - Whitespace is stripped.
    - If no '-' is found, whole string goes to city, state is ''.
    """
    cities = []
    states = []
    for v in series:
        if pd.isna(v):
            cities.append("")
            states.append("")
        else:
            text = str(v)
            parts = text.split("-", 1)
            if len(parts) == 2:
                city = parts[0].strip()
                state = parts[1].strip()
            else:
                city = text.strip()
                state = ""
            cities.append(city)
            states.append(state)
    return pd.Series(cities, index=series.index), pd.Series(states, index=series.index)


def compute_in_transit(df_tracked: pd.DataFrame):
    """
    For tracked shipments:
    - Classify missed milestones (missing timestamps / non-positive duration)
    - Compute rounded in-transit time for valid shipments

    Returns:
      detail_df: dataframe of shipments with VALID in-transit time (one row per shipment)
      missed_milestone_count: count of rows with missing/invalid timestamps or non-positive duration
    """
    df_tracked = df_tracked.copy()

    # Ensure datetime
    df_tracked[PICKUP_TS_COL] = pd.to_datetime(
        df_tracked[PICKUP_TS_COL], errors="coerce", utc=True
    )
    df_tracked[DROPOFF_TS_COL] = pd.to_datetime(
        df_tracked[DROPOFF_TS_COL], errors="coerce", utc=True
    )

    pickup = df_tracked[PICKUP_TS_COL]
    dropoff = df_tracked[DROPOFF_TS_COL]

    has_both = pickup.notna() & dropoff.notna()
    duration_days = (dropoff - pickup).dt.total_seconds() / (24 * 3600)

    # Valid only if both timestamps present AND duration strictly > 0
    positive_duration = duration_days > 0
    valid_mask = has_both & positive_duration

    # Missed milestone = everything else
    missed_mask = ~valid_mask
    missed_milestone_count = int(missed_mask.sum())

    # Keep only valid rows for in-transit calculation
    df_valid = df_tracked[valid_mask].copy()
    df_valid["transit_days_raw"] = duration_days[valid_mask]

    # Rounding logic
    def round_transit_days(x: float) -> float:
        # 0 < x < 0.5 → 0.5
        if 0 < x < 0.5:
            return 0.5
        # 0.5 <= x < 1 → 1
        if 0.5 <= x < 1:
            return 1.0
        # x >= 1 → round to nearest whole number, halves up
        if x >= 1:
            floor = np.floor(x)
            frac = x - floor
            if frac < 0.5:
                return float(floor)
            else:
                return float(floor + 1.0)
        # Any other case (<=0) should not appear given valid_mask, but be defensive
        return np.nan

    df_valid[IN_TRANSIT_OUTPUT_COL] = df_valid["transit_days_raw"].apply(round_transit_days)

    # Prepare the detail table (only the columns required for output)
    detail_cols = [
        BILL_OF_LADING_COL,
        PICKUP_NAME_COL,
        PICKUP_CITY_COL,
        PICKUP_STATE_COL,
        PICKUP_COUNTRY_COL,
        DROPOFF_NAME_COL,
        DROPOFF_CITY_COL,
        DROPOFF_STATE_COL,
        DROPOFF_COUNTRY_COL,
        IN_TRANSIT_OUTPUT_COL,
    ]

    detail_df = df_valid[detail_cols].copy()

    return detail_df, missed_milestone_count


def build_excel_file(
    tracked_count: int,
    untracked_count: int,
    missed_milestone_count: int,
    detail_df: pd.DataFrame,
    original_total_count: int,
) -> bytes:
    # Grand total = total number of rows in original file
    grand_total = original_total_count

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm:ss") as writer:
        workbook = writer.book
        sheet_name = "In-transit summary"
        worksheet = workbook.add_worksheet(sheet_name)
        writer.sheets[sheet_name] = worksheet

        # Summary table (A1:B5, with D1:E1 definition)
        worksheet.write("A1", "Label")
        worksheet.write("B1", "Shipment Count")
        worksheet.write("D1", "Definition of in transit time")
        worksheet.write("E1", "Time taken from departure to arrival")

        worksheet.write("A2", "Tracked")
        worksheet.write("B2", tracked_count)

        worksheet.write("A3", "Missed milestone")
        worksheet.write("B3", missed_milestone_count)

        worksheet.write("A4", "Untracked")
        worksheet.write("B4", untracked_count)

        worksheet.write("A5", "Grand total")
        worksheet.write("B5", grand_total)

        # Blank row 6 (index 5) – intentionally left blank

        # Detailed tracked shipments table header at row 7 (index 6)
        headers = [
            "Bill of lading",
            "Pickup name",
            "Pickup city",
            "Pickup state",
            "Pickup country",
            "Drop-off name",
            "Drop-off city",
            "Drop-off state",
            "Drop-off country",
            "In transit time",
        ]
        header_row_idx = 6
        for col_idx, header in enumerate(headers):
            worksheet.write(header_row_idx, col_idx, header)

        # Data starts at row 8 (index 7)
        start_row_idx = 7
        for row_offset, (_, row) in enumerate(detail_df.iterrows()):
            excel_row = start_row_idx + row_offset
            worksheet.write(excel_row, 0, row[BILL_OF_LADING_COL])
            worksheet.write(excel_row, 1, row[PICKUP_NAME_COL])
            worksheet.write(excel_row, 2, row[PICKUP_CITY_COL])
            worksheet.write(excel_row, 3, row[PICKUP_STATE_COL])
            worksheet.write(excel_row, 4, row[PICKUP_COUNTRY_COL])
            worksheet.write(excel_row, 5, row[DROPOFF_NAME_COL])
            worksheet.write(excel_row, 6, row[DROPOFF_CITY_COL])
            worksheet.write(excel_row, 7, row[DROPOFF_STATE_COL])
            worksheet.write(excel_row, 8, row[DROPOFF_COUNTRY_COL])
            worksheet.write(excel_row, 9, row[IN_TRANSIT_OUTPUT_COL])

    output.seek(0)
    return output.getvalue()


def build_csv_file(
    tracked_count: int,
    untracked_count: int,
    missed_milestone_count: int,
    detail_df: pd.DataFrame,
    original_total_count: int,
) -> str:
    grand_total = original_total_count
    max_cols = 10  # A–J

    def pad_row(values):
        row = list(values)
        if len(row) < max_cols:
            row.extend([""] * (max_cols - len(row)))
        else:
            row = row[:max_cols]
        return row

    rows = []

    # Row 1: headers for summary + definition
    rows.append(
        pad_row(
            [
                "Label",
                "Shipment Count",
                "",
                "Definition of in transit time",
                "Time taken from departure to arrival",
            ]
        )
    )

    # Row 2–5: summary
    rows.append(pad_row(["Tracked", tracked_count]))
    rows.append(pad_row(["Missed milestone", missed_milestone_count]))
    rows.append(pad_row(["Untracked", untracked_count]))
    rows.append(pad_row(["Grand total", grand_total]))

    # Row 6: blank separator
    rows.append(pad_row([""]))

    # Row 7: detail header
    headers = [
        "Bill of lading",
        "Pickup name",
        "Pickup city",
        "Pickup state",
        "Pickup country",
        "Drop-off name",
        "Drop-off city",
        "Drop-off state",
        "Drop-off country",
        "In transit time",
    ]
    rows.append(pad_row(headers))

    # From Row 8: detail data (only valid in-transit shipments)
    for _, r in detail_df.iterrows():
        rows.append(
            pad_row(
                [
                    r[BILL_OF_LADING_COL],
                    r[PICKUP_NAME_COL],
                    r[PICKUP_CITY_COL],
                    r[PICKUP_STATE_COL],
                    r[PICKUP_COUNTRY_COL],
                    r[DROPOFF_NAME_COL],
                    r[DROPOFF_CITY_COL],
                    r[DROPOFF_STATE_COL],
                    r[DROPOFF_COUNTRY_COL],
                    r[IN_TRANSIT_OUTPUT_COL],
                ]
            )
        )

    csv_df = pd.DataFrame(rows)
    buffer = StringIO()
    # No header row, because we already included our own "row 1"
    csv_df.to_csv(buffer, index=False, header=False)
    return buffer.getvalue()


def process_file(df: pd.DataFrame):
    df = df.copy()

    # Validate required raw columns first
    validate_columns(df)

    original_total_count = len(df)

    # Derive city/state columns from "Pickup City State" and "Dropoff City State"
    pickup_cities, pickup_states = split_city_state(df[RAW_PICKUP_CITY_STATE_COL])
    dropoff_cities, dropoff_states = split_city_state(df[RAW_DROPOFF_CITY_STATE_COL])

    df[PICKUP_CITY_COL] = pickup_cities
    df[PICKUP_STATE_COL] = pickup_states
    df[DROPOFF_CITY_COL] = dropoff_cities
    df[DROPOFF_STATE_COL] = dropoff_states

    tracked_mask, untracked_mask = get_tracked_untracked_masks(df)

    df_tracked = df[tracked_mask].copy()
    df_untracked = df[untracked_mask].copy()

    tracked_count = int(len(df_tracked))
    untracked_count = int(len(df_untracked))

    # Compute in-transit & missed milestones within tracked
    detail_df, missed_milestone_count = compute_in_transit(df_tracked)

    actual_tracked_with_valid_transit = len(detail_df)

    # Build files for download
    excel_bytes = build_excel_file(
        tracked_count=tracked_count,
        untracked_count=untracked_count,
        missed_milestone_count=missed_milestone_count,
        detail_df=detail_df,
        original_total_count=original_total_count,
    )
    csv_str = build_csv_file(
        tracked_count=tracked_count,
        untracked_count=untracked_count,
        missed_milestone_count=missed_milestone_count,
        detail_df=detail_df,
        original_total_count=original_total_count,
    )

    summary_counts = {
        "tracked_count": tracked_count,
        "missed_milestone_count": missed_milestone_count,
        "untracked_count": untracked_count,
        "grand_total": original_total_count,
        "actual_tracked_with_valid_transit": actual_tracked_with_valid_transit,
    }

    return summary_counts, detail_df, excel_bytes, csv_str


# ====== STREAMLIT APP ======

st.set_page_config(page_title="FTL In-Transit Time Calculator", layout="wide")

st.title("FTL In-Transit Time Calculator")
st.write(
    "Upload a raw FTL tracking file (CSV or Excel). "
    "The app will calculate in-transit time (in days) for tracked shipments and "
    "mark missed milestones according to your business rules."
)

uploaded_file = st.file_uploader(
    "Upload your raw FTL tracking file",
    type=["csv", "xlsx", "xls"],
    help="The file should contain the required columns such as Tracked, timestamps, and lane details.",
)

if uploaded_file is not None:
    try:
        # Read CSV or Excel into DataFrame
        if uploaded_file.name.lower().endswith(".csv"):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_excel(uploaded_file)

        # Normalize/standardize column names to match our constants
        df_input = standardize_columns(df_input)

        st.subheader("Preview of uploaded data (after column standardization)")
        st.dataframe(df_input.head(20), use_container_width=True)
        # Uncomment this if you want to see exact column names:
        # st.write("Columns after standardization:", list(df_input.columns))

        if st.button("Process file"):
            with st.spinner("Processing shipments..."):
                summary_counts, detail_df, excel_bytes, csv_str = process_file(df_input)

            st.success("Processing complete!")

            # Display summary metrics
            st.subheader("Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Tracked", summary_counts["tracked_count"])
            with col2:
                st.metric("Missed milestone", summary_counts["missed_milestone_count"])
            with col3:
                st.metric("Untracked", summary_counts["untracked_count"])
            with col4:
                st.metric("Grand total", summary_counts["grand_total"])

            # Tabular summary view
            summary_df = pd.DataFrame(
                {
                    "Label": ["Tracked", "Missed milestone", "Untracked", "Grand total"],
                    "Shipment Count": [
                        summary_counts["tracked_count"],
                        summary_counts["missed_milestone_count"],
                        summary_counts["untracked_count"],
                        summary_counts["grand_total"],
                    ],
                }
            )
            st.table(summary_df)

            # Detailed table of valid tracked shipments with in-transit time
            st.subheader("Tracked shipments with valid in-transit time")
            st.caption(
                "Only tracked shipments with both timestamps present and a positive transit duration are shown here."
            )
            st.dataframe(detail_df, use_container_width=True)

            # Download buttons
            st.subheader("Download processed files")

            st.download_button(
                label="⬇️ Download Excel (with summary layout)",
                data=excel_bytes,
                file_name="in_transit_processed.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.download_button(
                label="⬇️ Download CSV (summary + detailed table)",
                data=csv_str.encode("utf-8"),
                file_name="in_transit_processed.csv",
                mime="text/csv",
            )

    except ValueError as ve:
        st.error(str(ve))
    except Exception as e:
        st.error(f"An unexpected error occurred while processing the file: {e}")
else:
    st.info("Please upload a CSV or Excel file to begin.")
