import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.optimize import newton
from scipy.stats import norm
from datetime import datetime



# ---- Streamlit Page Configuration ----
st.set_page_config(page_title="XIRR & P&L Dashboard", page_icon="📊", layout="wide")

# ---- Title ----
st.title("📊 XIRR & P&L Calculator for Closed Trades")

# ---- File Upload ----
fl = st.file_uploader("📂 Upload Trade Ledger (CSV/XLSX)", type=["csv", "xlsx"])

if fl is not None:
    # ---- Read File ----
    ext = fl.name.split(".")[-1]
    df = pd.read_csv(fl) if ext == "csv" else pd.read_excel(fl)

    # Normalize column names
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

    # Convert 'trade_date' to datetime and sort by date
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df.dropna(subset=["trade_date"], inplace=True)
    df.sort_values(by="trade_date", ascending=True, inplace=True)


    # ---- Streamlit Sidebar for Date Filter ----
    st.sidebar.header("📅 Filter Data by Date Range")
    start_date = st.sidebar.date_input("Start Date", value=df["trade_date"].min().date())
    end_date = st.sidebar.date_input("End Date", value=df["trade_date"].max().date())

    # Convert user inputs to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # ---- Filter Data Based on Selected Date Range ----
    df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
    # ---- Identify ETFs and Stocks ----
    df["product_type"] = df["scrip_name"].apply(lambda x: "ETF" if "ETF" in str(x).upper() else "Stock")

    # ---- Process Closed Trades & Open Positions ----
    closed_trades, open_positions = [], {}

    for _, row in df.iterrows():
        symbol, trade_date = row["symbol"], row["trade_date"]
        buy_qty, buy_rate = row["buy_qty"], row["buy_rate"]
        sell_qty, sell_rate = row["sell_qty"], row["sell_rate"]
        product_type = row["product_type"]

        # Store buy transactions
        if buy_qty > 0:
            open_positions.setdefault(symbol, []).append({"date": trade_date, "qty": buy_qty, "rate": buy_rate, "product_type": product_type})

        # Match sell transactions with buy (FIFO)
        if sell_qty > 0 and symbol in open_positions:
            remaining_sell_qty = sell_qty
            while remaining_sell_qty > 0 and open_positions[symbol]:
                open_buy = open_positions[symbol][0]
                matched_qty = min(open_buy["qty"], remaining_sell_qty)
                pnl = (matched_qty * sell_rate) - (matched_qty * open_buy["rate"])
                pnl_percentage = (pnl / (matched_qty * open_buy["rate"])) * 100 if matched_qty * open_buy["rate"] != 0 else 0

                # Add **BUY** and **SELL** transactions
                closed_trades.append({"trade_date": open_buy["date"], "symbol": symbol, "product_type": product_type,
                                      "closed_qty": matched_qty, "buy_amount": matched_qty * open_buy["rate"],
                                      "sell_amount": 0, "net_pnl": 0, "pnl_percentage": 0,
                                      "cash_flow": -matched_qty * open_buy["rate"], "transaction_type": "BUY"})

                closed_trades.append({"trade_date": trade_date, "symbol": symbol, "product_type": product_type,
                                      "closed_qty": matched_qty, "buy_amount": 0, "sell_amount": matched_qty * sell_rate,
                                      "net_pnl": pnl, "pnl_percentage": pnl_percentage,
                                      "cash_flow": matched_qty * sell_rate, "transaction_type": "SELL"})

                open_buy["qty"] -= matched_qty
                remaining_sell_qty -= matched_qty

                if open_buy["qty"] == 0:
                    open_positions[symbol].pop(0)

    # Convert closed trades to DataFrame
    closed_trades_df = pd.DataFrame(closed_trades).sort_values(by="trade_date", ascending=True)

    # ---- Compute Open Positions ----
    open_positions_list = []
    for symbol, buys in open_positions.items():
        for buy in buys:
            open_positions_list.append({
                "symbol": symbol,
                "product_type": buy["product_type"],
                "open_qty": buy["qty"],
                "avg_buy_price": buy["rate"],
                "total_value": buy["qty"] * buy["rate"],
                "buy_date": buy["date"]
            })

    # Convert open positions to DataFrame
    open_positions_df = pd.DataFrame(open_positions_list)
    
    # Separate ETF & Stock open positions
    etf_open_positions = open_positions_df[open_positions_df["product_type"] == "ETF"]
    stock_open_positions = open_positions_df[open_positions_df["product_type"] == "Stock"]

    # ---- Display Open Positions ----
    st.subheader("📌 Current Open Positions")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 ETF Open Positions")
        st.dataframe(etf_open_positions)
    with col2:
        st.subheader("📈 Stock Open Positions")
        st.dataframe(stock_open_positions)

    # ---- Compute P&L Summary ----
    pnl_summary = closed_trades_df.groupby(["product_type", "symbol"]).agg(
        net_buy_amount=("buy_amount", "sum"), net_sell_amount=("sell_amount", "sum"),
        net_pnl=("net_pnl", "sum"), pnl_percentage=("pnl_percentage", "mean")
    ).reset_index()

    # Compute Total P&L
    net_pnl = pnl_summary["net_pnl"].sum()
    pnl_etf = pnl_summary[pnl_summary["product_type"] == "ETF"]["net_pnl"].sum()
    pnl_stock = pnl_summary[pnl_summary["product_type"] == "Stock"]["net_pnl"].sum()

    # ---- Display Summary Metrics ----
    st.subheader("📊 Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Net P&L", f"{net_pnl:,.2f}")
    col2.metric("📈 ETF P&L", f"{pnl_etf:,.2f}")
    col3.metric("📉 Stock P&L", f"{pnl_stock:,.2f}")

    # ---- Compute XIRR ----
    def compute_xirr(cashflows, cashflow_dates):
        day_counts = [(date - cashflow_dates[0]).days for date in cashflow_dates]
        def npv_equation(rate):
            return sum(cash / ((1 + rate) ** (days / 365)) for cash, days in zip(cashflows, day_counts))
        return newton(npv_equation, x0=0.1)

    if not closed_trades_df.empty:
        cash_flows = closed_trades_df.groupby("trade_date")["cash_flow"].sum().reset_index()
        cash_flow_values, cash_flow_dates = cash_flows["cash_flow"].tolist(), cash_flows["trade_date"].tolist()

        try:
            calculated_xirr = compute_xirr(cash_flow_values, cash_flow_dates) * 100
        except:
            calculated_xirr = None
    else:
        calculated_xirr = None

    # ---- Display XIRR Calculation Result ----
    st.subheader("📊 XIRR Calculation Result (For Closed Trades)")
    st.metric("📈 XIRR (%)", f"{calculated_xirr:.2f}%" if calculated_xirr else "⚠️ XIRR Calculation Failed")

    # ---- Pie Charts ----
    st.subheader("📊 Net P&L & Investment Distribution")
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(px.pie(pnl_summary, names="product_type", values="net_pnl",
                               title="Net P&L Distribution", color_discrete_sequence=["blue", "orange"]), use_container_width=True)

    with col2:
        st.plotly_chart(px.pie(pnl_summary, names="product_type", values="net_buy_amount",
                               title="Net Investment Distribution", color_discrete_sequence=["purple", "green"]), use_container_width=True)

    # ---- Streamlit Header ----
    st.subheader("📊 Net Closed Investment vs. Net Return Comparison")

    # ---- Prepare Data ----
    data = {
        "Category": ["ETF", "ETF", "Stock", "Stock"],
        "Metric": ["Net Investment", "Net Return", "Net Investment", "Net Return"],
        "Value": [
            pnl_summary[pnl_summary["product_type"] == "ETF"]["net_buy_amount"].sum(),
            pnl_summary[pnl_summary["product_type"] == "ETF"]["net_pnl"].sum(),
            pnl_summary[pnl_summary["product_type"] == "Stock"]["net_buy_amount"].sum(),
            pnl_summary[pnl_summary["product_type"] == "Stock"]["net_pnl"].sum()
        ]
    }

    df_bar = pd.DataFrame(data)

    # ---- Create Interactive Bar Chart ----
    fig = px.bar(df_bar, x="Category", y="Value", color="Metric",
                barmode="group", title="Investment vs. Return for ETF & Stocks",
                labels={"Value": "Amount (₹)", "Category": "Investment Type"},
                text_auto=True)

    # ---- Display Chart in Streamlit ----
    st.plotly_chart(fig, use_container_width=True)


        # ---- Aggregate Daily P&L for Line Chart ----
    daily_pnl = closed_trades_df.groupby("trade_date")["net_pnl"].sum().reset_index()

    # Categorize Profit & Loss Separately for Coloring
    daily_pnl["P&L Type"] = np.where(daily_pnl["net_pnl"] >= 0, "Profit", "Loss")

    # ---- Interactive Line Chart for Net P&L Over Time ----
    st.subheader("📊 P&L Trend Over Time (Profits & Losses)")

    fig_line = px.line(
        daily_pnl, x="trade_date", y="net_pnl", 
        title="Net P&L Over Time (Profit & Loss Visualization)",
        labels={"net_pnl": "Net P&L (₹)", "trade_date": "Date"},
        markers=True,
        color="P&L Type",  # Color-code profit vs loss
        color_discrete_map={"Profit": "green", "Loss": "red"}
    )

    # Show Net P&L Values on Hover
    fig_line.update_traces(mode="markers+lines", marker=dict(size=8), text=daily_pnl["net_pnl"], hoverinfo="text+x")

    # ---- Display the Chart in Streamlit ----
    st.plotly_chart(fig_line, use_container_width=True)


     # ---- Normal Distribution Graph for P&L % ----
    st.subheader("📊 Normal Distribution of P&L %")

    def plot_distribution(data, title):
        fig, ax = plt.subplots(figsize=(5, 2.5))  # Adjusted figure size
        sns.histplot(data, bins=15, kde=True, stat="density", color="blue", edgecolor="black", label="Histogram")
        mu, sigma = np.mean(data), np.std(data)
        x_vals = np.linspace(min(data), max(data), 100)
        y_vals = norm.pdf(x_vals, mu, sigma)
        ax.plot(x_vals, y_vals, color="red", label="Normal Distribution")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("P&L %", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        return fig

    # ---- Arrange Charts in Rows ----
    row1_col1, row1_col2 = st.columns(2)  # First row (two charts side by side)
    row2_col1 = st.columns(1)[0]  # Second row (one full-width chart)

    with row1_col1:
        st.pyplot(plot_distribution(pnl_summary["pnl_percentage"], "Overall P&L % Distribution"))

    if not pnl_summary[pnl_summary["product_type"] == "ETF"].empty:
        with row1_col2:
            st.pyplot(plot_distribution(pnl_summary[pnl_summary["product_type"] == "ETF"]["pnl_percentage"], "ETF P&L % Distribution"))

    if not pnl_summary[pnl_summary["product_type"] == "Stock"].empty:
        with row2_col1:
            st.pyplot(plot_distribution(pnl_summary[pnl_summary["product_type"] == "Stock"]["pnl_percentage"], "Stock P&L % Distribution"))


    # ---- Aggregate Daily P&L for Cumulative Growth ----
    daily_pnl = closed_trades_df.groupby("trade_date").agg(
        net_pnl=("net_pnl", "sum"),
        etf_pnl=("net_pnl", lambda x: x[closed_trades_df["product_type"] == "ETF"].sum()),
        stock_pnl=("net_pnl", lambda x: x[closed_trades_df["product_type"] == "Stock"].sum())
    ).reset_index()

    # Fill missing values with 0
    daily_pnl.fillna(0, inplace=True)

    # Compute Cumulative Sum to Track Growth Over Time
    daily_pnl["cumulative_net_pnl"] = daily_pnl["net_pnl"].cumsum()
    daily_pnl["cumulative_etf_pnl"] = daily_pnl["etf_pnl"].cumsum()
    daily_pnl["cumulative_stock_pnl"] = daily_pnl["stock_pnl"].cumsum()

    # ---- Interactive Cumulative P&L Growth Line Chart ----
    st.subheader("📊 Cumulative P&L Growth Over Time")

    fig_cumulative = px.line(
        daily_pnl, x="trade_date",
        y=["cumulative_net_pnl", "cumulative_etf_pnl", "cumulative_stock_pnl"],
        title="Cumulative P&L Growth Over Time",
        labels={"value": "Cumulative P&L (₹)", "trade_date": "Date"},
        markers=True
    )

    # Rename legend labels
    fig_cumulative.for_each_trace(lambda t: t.update(name={
        "cumulative_net_pnl": "Net P&L",
        "cumulative_etf_pnl": "ETF P&L",
        "cumulative_stock_pnl": "Stock P&L"
    }[t.name]))

    # Display the Chart in Streamlit
    st.plotly_chart(fig_cumulative, use_container_width=True)

    # ---- Display DataFrames ----
    st.subheader("📄 Processed DataFrames")
    st.dataframe(closed_trades_df)
    st.dataframe(pnl_summary)

    
