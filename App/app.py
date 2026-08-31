from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Application configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Invoice Intelligence Dashboard",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

MODEL_DIRS = [
    PROJECT_ROOT / "Models",
    APP_DIR / "Models",
    PROJECT_ROOT / "models",
]

DATABASE_PATHS = [
    PROJECT_ROOT / "Dataset" / "inventory.db",
    APP_DIR / "Dataset" / "inventory.db",
]


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: "Inter", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, #e8f0ff 0, transparent 28%),
                linear-gradient(135deg, #f8fafc 0%, #eef2f7 100%);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #1e293b 100%);
            border-right: 1px solid rgba(255,255,255,.08);
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        .hero {
            padding: 2rem 2.25rem;
            border-radius: 24px;
            color: white;
            background: linear-gradient(135deg, #172554 0%, #2563eb 55%, #38bdf8 100%);
            box-shadow: 0 18px 45px rgba(37, 99, 235, .22);
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.35rem;
            font-weight: 800;
        }

        .hero p {
            margin: .55rem 0 0;
            color: #dbeafe;
            font-size: 1rem;
        }

        .card {
            padding: 1.25rem;
            border: 1px solid rgba(148, 163, 184, .2);
            border-radius: 18px;
            background: rgba(255,255,255,.76);
            box-shadow: 0 10px 30px rgba(15, 23, 42, .06);
            margin-bottom: 1rem;
        }

        .metric-card {
            padding: 1.25rem;
            border-radius: 18px;
            background: rgba(255,255,255,.82);
            border: 1px solid #e2e8f0;
            box-shadow: 0 8px 25px rgba(15, 23, 42, .06);
        }

        .metric-label {
            color: #64748b;
            font-size: .82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: .04em;
        }

        .metric-value {
            color: #0f172a;
            font-size: 1.8rem;
            font-weight: 800;
            margin-top: .3rem;
        }

        .status {
            padding: .65rem .8rem;
            border-radius: 10px;
            margin: .4rem 0;
            font-size: .85rem;
        }

        .status-ok {
            color: #166534;
            background: #dcfce7;
        }

        .status-error {
            color: #991b1b;
            background: #fee2e2;
        }

        .result-safe {
            padding: 1.4rem;
            border-radius: 16px;
            color: #166534;
            background: linear-gradient(135deg, #dcfce7, #bbf7d0);
            border: 1px solid #86efac;
        }

        .result-warning {
            padding: 1.4rem;
            border-radius: 16px;
            color: #9a3412;
            background: linear-gradient(135deg, #ffedd5, #fed7aa);
            border: 1px solid #fdba74;
        }

        .result-danger {
            padding: 1.4rem;
            border-radius: 16px;
            color: #991b1b;
            background: linear-gradient(135deg, #fee2e2, #fecaca);
            border: 1px solid #fca5a5;
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,.8);
            border: 1px solid #e2e8f0;
            padding: 1rem;
            border-radius: 16px;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def first_existing(paths: list[Path]) -> Path | None:
    """Return the first existing path."""
    return next((path for path in paths if path.exists()), None)


@st.cache_resource(show_spinner=False)
def load_model(filename: str) -> tuple[Any | None, str | None]:
    """Load a model from the supported model directories."""
    model_path = first_existing([directory / filename for directory in MODEL_DIRS])

    if model_path is None:
        return None, f"Model not found: {filename}"

    try:
        try:
            model = joblib.load(model_path)
        except Exception:
            with model_path.open("rb") as file:
                model = pickle.load(file)

        return model, str(model_path)

    except Exception as exc:
        return None, f"Could not load {filename}: {exc}"


@st.cache_data(show_spinner=False)
def database_tables(database_path: str) -> list[str]:
    """Return SQLite table names."""
    with sqlite3.connect(database_path) as connection:
        query = """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """
        return pd.read_sql_query(query, connection)["name"].tolist()


def read_table(database_path: str, table_name: str, limit: int) -> pd.DataFrame:
    """Read a table safely from SQLite."""
    with sqlite3.connect(database_path) as connection:
        safe_table = '"' + table_name.replace('"', '""') + '"'
        return pd.read_sql_query(
            f"SELECT * FROM {safe_table} LIMIT ?",
            connection,
            params=(limit,),
        )


def model_feature_names(model: Any) -> list[str] | None:
    """Read feature names when the estimator exposes them."""
    names = getattr(model, "feature_names_in_", None)

    if names is not None:
        return [str(name) for name in names]

    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                return [str(name) for name in names]

    return None


def build_model_input(model: Any, values: dict[str, Any]) -> pd.DataFrame:
    """
    Build a model input DataFrame.

    If feature_names_in_ exists, values are matched by exact name or
    case-insensitive name. Otherwise, the standard input order is used.
    """
    feature_names = model_feature_names(model)

    if feature_names:
        normalized = {key.lower().strip(): value for key, value in values.items()}
        row = [
            normalized.get(name.lower().strip(), 0)
            for name in feature_names
        ]
        return pd.DataFrame([row], columns=feature_names)

    return pd.DataFrame([list(values.values())], columns=list(values.keys()))


def prediction_value(prediction: Any) -> Any:
    """Convert NumPy scalar predictions to regular Python values."""
    if hasattr(prediction, "item"):
        return prediction.item()
    return prediction


# -----------------------------------------------------------------------------
# Cached model loading
# -----------------------------------------------------------------------------

freight_model, freight_model_status = load_model("Best_freight_model.pkl")
flagging_model, flagging_model_status = load_model("Flagging Vendor Invoices.pkl")
database_path = first_existing(DATABASE_PATHS)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🧾 Invoice Intelligence")
    st.caption("Smart invoice analytics and prediction")

    page = st.radio(
        "Navigate",
        [
            "📊 Overview & Analytics",
            "🚚 Freight Cost Predictor",
            "🚩 Invoice Flagging & Anomaly Detection",
            "🗄️ Database Viewer",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### System status")

    if freight_model is not None:
        st.markdown(
            '<div class="status status-ok">● Freight model loaded</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status status-error">● Freight model unavailable</div>',
            unsafe_allow_html=True,
        )

    if flagging_model is not None:
        st.markdown(
            '<div class="status status-ok">● Flagging model loaded</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status status-error">● Flagging model unavailable</div>',
            unsafe_allow_html=True,
        )

    if database_path is not None:
        st.markdown(
            '<div class="status status-ok">● Database connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status status-error">● Database unavailable</div>',
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Shared header
# -----------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>Invoice Intelligence Dashboard</h1>
        <p>Transform invoice data into faster, safer, and more informed decisions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------

if page == "📊 Overview & Analytics":
    st.subheader("Overview & Analytics")

    total_rows = 0
    table_count = 0

    if database_path is not None:
        try:
            tables = database_tables(str(database_path))
            table_count = len(tables)

            for table in tables:
                try:
                    with sqlite3.connect(database_path) as connection:
                        result = pd.read_sql_query(
                            f'SELECT COUNT(*) AS count FROM "{table}"',
                            connection,
                        )
                        total_rows += int(result.iloc[0]["count"])
                except Exception:
                    continue
        except Exception as exc:
            st.warning(f"Database analytics could not be loaded: {exc}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Invoice records</div>
                <div class="metric-value">{total_rows:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Database tables</div>
                <div class="metric-value">{table_count:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Prediction models</div>
                <div class="metric-value">{int(freight_model is not None) + int(flagging_model is not None)}/2</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Platform status</div>
                <div class="metric-value">Ready</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown(
        """
        <div class="card">
            <h3>Welcome to Invoice Intelligence</h3>
            <p>
                Use the navigation panel to estimate freight costs, identify
                suspicious invoices, or explore your SQLite database.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if database_path is not None:
        try:
            tables = database_tables(str(database_path))
            if tables:
                st.markdown("### Available data")
                st.dataframe(
                    pd.DataFrame({"Table": tables}),
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception as exc:
            st.info(f"No database summary available: {exc}")



# -----------------------------------------------------------------------------
# Freight predictor
# -----------------------------------------------------------------------------

elif page == "🚚 Freight Cost Predictor":
    st.subheader("Freight Cost Predictor")
    st.caption("Enter the invoice amount and quantity to estimate freight cost.")

    if freight_model is None:
        st.error(freight_model_status)
        st.info("Place the model in the Models folder and refresh the application.")
    else:
        with st.form("freight_prediction_form"):

            col1, col2 = st.columns(2)

            with col1:
                dollars = st.number_input(
                    "Invoice Amount ($)",
                    min_value=0.0,
                    value=1000.0,
                    step=50.0,
                    help="Enter the invoice amount (Dollars).",
                )

            with col2:
                quantity = st.number_input(
                    "Quantity",
                    min_value=0.0,
                    value=10.0,
                    step=1.0,
                    help="Enter the total quantity of items.",
                )

            submitted = st.form_submit_button(
                "Estimate Freight Cost",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            values = {
                "Dollars": dollars,
                "Quantity": quantity,
            }

            try:
                model_input = build_model_input(
                    freight_model,
                    values,
                )

                prediction = prediction_value(
                    freight_model.predict(model_input)[0]
                )

                st.markdown(
                    f"""
                    <div class="result-safe">
                        <h3>Estimated Freight Cost</h3>
                        <h1>${float(prediction):,.2f}</h1>
                        <p>
                            Freight cost estimated from invoice amount
                            and quantity.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("View Model Input"):
                    st.dataframe(
                        model_input,
                        use_container_width=True,
                        hide_index=True,
                    )

            except Exception as exc:
                st.error(
                    "Prediction failed. Please make sure the input features "
                    "match the features used when training the freight model."
                )
                st.exception(exc)


# -----------------------------------------------------------------------------
# Invoice flagging
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Invoice flagging
# -----------------------------------------------------------------------------

elif page == "🚩 Invoice Flagging & Anomaly Detection":
    st.subheader("Invoice Flagging & Anomaly Detection")
    st.caption(
        "Enter invoice details to determine whether the invoice should be flagged."
    )

    if flagging_model is None:
        st.error(flagging_model_status)
        st.info(
            "Place the model in the Models folder and refresh the application."
        )
    else:
        with st.form("invoice_flagging_form"):

            col1, col2 = st.columns(2)

            with col1:
                invoice_quantity = st.number_input(
                    "Invoice Quantity",
                    min_value=0.0,
                    value=10.0,
                    step=1.0,
                )

                invoice_dollars = st.number_input(
                    "Invoice Dollars ($)",
                    min_value=0.0,
                    value=1000.0,
                    step=50.0,
                )

                freight = st.number_input(
                    "Freight ($)",
                    min_value=0.0,
                    value=50.0,
                    step=5.0,
                )

                total_brands = st.number_input(
                    "Total Brands",
                    min_value=0,
                    value=1,
                    step=1,
                )

            with col2:
                total_item_quantity = st.number_input(
                    "Total Item Quantity",
                    min_value=0.0,
                    value=10.0,
                    step=1.0,
                )

                days_po_to_invoice = st.number_input(
                    "Days PO to Invoice",
                    min_value=0.0,
                    value=5.0,
                    step=1.0,
                )

                total_item_dollars = st.number_input(
                    "Total Item Dollars ($)",
                    min_value=0.0,
                    value=1000.0,
                    step=50.0,
                )

            submitted = st.form_submit_button(
                "Analyze Invoice",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            values = {
                "invoice_quantity": invoice_quantity,
                "invoice_dollars": invoice_dollars,
                "Freight": freight,
                "total_brands": total_brands,
                "total_item_quantity": total_item_quantity,
                "days_po_to_invoice": days_po_to_invoice,
                "total_item_dollars": total_item_dollars,
            }

            try:
                model_input = build_model_input(
                    flagging_model,
                    values,
                )

                prediction = prediction_value(
                    flagging_model.predict(model_input)[0]
                )

                # Get probability if the model supports it
                probability = None

                if hasattr(flagging_model, "predict_proba"):
                    probabilities = flagging_model.predict_proba(
                        model_input
                    )[0]

                    probability = float(max(probabilities))

                # Determine whether invoice is flagged
                prediction_text = str(prediction).lower()

                is_flagged = (
                    prediction in (1, True, "1")
                    or any(
                        word in prediction_text
                        for word in (
                            "flag",
                            "risk",
                            "anomaly",
                            "fraud",
                            "suspicious",
                        )
                    )
                )

                if is_flagged:
                    result_class = "result-danger"
                    title = "🚩 Invoice Flagged"
                    message = (
                        "The classification model identified this invoice "
                        "as potentially suspicious and it should be reviewed."
                    )
                else:
                    result_class = "result-safe"
                    title = "✅ Invoice Looks Normal"
                    message = (
                        "The classification model did not identify this "
                        "invoice as suspicious."
                    )

                confidence_text = (
                    f"<p>Model confidence: {probability:.1%}</p>"
                    if probability is not None
                    else ""
                )

                st.markdown(
                    f"""
                    <div class="{result_class}">
                        <h3>{title}</h3>
                        <p>{message}</p>
                        {confidence_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("View Model Input"):
                    st.dataframe(
                        model_input,
                        use_container_width=True,
                        hide_index=True,
                    )

            except Exception as exc:
                st.error(
                    "Classification failed. Please make sure that the "
                    "frontend inputs match the features used during model training."
                )
                st.exception(exc)


# -----------------------------------------------------------------------------
# Database viewer
# -----------------------------------------------------------------------------

elif page == "🗄️ Database Viewer":
    st.subheader("Database Viewer")
    st.caption("Explore invoice records stored in inventory.db.")

    if database_path is None:
        st.error(
            "Database not found. Expected Dataset/inventory.db relative to the "
            "project root."
        )
    else:
        try:
            tables = database_tables(str(database_path))

            if not tables:
                st.warning("The database does not contain any user tables.")
            else:
                selected_table = st.selectbox("Select table", tables)
                row_limit = st.slider(
                    "Rows to display",
                    min_value=10,
                    max_value=5000,
                    value=100,
                    step=10,
                )

                data = read_table(
                    str(database_path),
                    selected_table,
                    row_limit,
                )

                st.dataframe(data, use_container_width=True, hide_index=True)

                csv_data = data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download displayed data as CSV",
                    data=csv_data,
                    file_name=f"{selected_table}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                st.caption(
                    f"Showing {len(data):,} rows from `{selected_table}` "
                    f"using `{database_path}`."
                )

        except Exception as exc:
            st.error(f"Could not query the SQLite database: {exc}")