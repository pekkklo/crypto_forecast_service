# frontend/app.py

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
import calendar
import numpy as np

API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Crypto Forecast", layout="wide")

st.markdown(
    '<h1 style="text-align: center;">Crypto Forecast Service</h1>',
    unsafe_allow_html=True
)


if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = "BTC"

# =========================================================
# Сайдбар
# =========================================================

with st.sidebar:

    st.title("Navigation")

    try:
        response = requests.get(f"{API_URL}/cryptocurrencies")

        if response.status_code == 200:
            cryptos = response.json()

            symbols = [c['symbol'] for c in cryptos]

            selected_symbol = st.selectbox(
                "Select Cryptocurrency",
                options=symbols,
                index=symbols.index(
                    st.session_state.selected_symbol
                ) if st.session_state.selected_symbol in symbols else 0
            )

            st.session_state.selected_symbol = selected_symbol

    except Exception as e:
        st.error(f"Cannot connect to backend: {e}")
        selected_symbol = "BTC"

    st.divider()

    horizon = st.radio(
        "Forecast Horizon",
        options=[30, 180],
        format_func=lambda x: f"{x} days",
        horizontal=True
    )

    st.divider()

    # =====================================================
    # Панель админа для обновления прогнозов
    # =====================================================

    with st.expander("Admin Panel"):

        login = st.text_input("Login", key="admin_login")

        password = st.text_input(
            "Password",
            type="password",
            key="admin_password"
        )

        if st.button("Update Forecasts", type="primary"):

            if login and password:

                try:
                    resp = requests.post(
                        f"{API_URL}/admin/update",
                        json={
                            "login": login,
                            "password": password
                        }
                    )

                    if resp.status_code == 200:
                        st.success(
                            "✅ Update completed successfully"
                        )

                    else:
                        st.error("❌ Invalid credentials")

                except Exception as e:
                    st.error(f"Error: {e}")

            else:
                st.warning("Enter credentials")

# =========================================================
# TABS
# =========================================================

tab1, tab2 = st.tabs(["Forecast", "Historical"])

# =========================================================
# Текущий прогноз
# =========================================================

with tab1:

    st.header(f"{selected_symbol} — {horizon}-day Forecast")

    with st.spinner("Loading..."):

        try:

            resp = requests.get(
                f"{API_URL}/forecasts/{selected_symbol}",
                params={"horizon": horizon}
            )

            if resp.status_code == 200:

                data = resp.json()

                forecasts = data.get('forecasts', [])

                if forecasts:

                    df = pd.DataFrame(forecasts)

                    df['target_date'] = pd.to_datetime(
                        df['target_date']
                    )

                    # =================================================
                    # График
                    # =================================================

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=df['target_date'],
                        y=df['upper_bound'],
                        fill=None,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ))

                    fig.add_trace(go.Scatter(
                        x=df['target_date'],
                        y=df['lower_bound'],
                        fill='tonexty',
                        mode='lines',
                        line=dict(width=0),
                        name='Confidence Interval',
                        fillcolor='rgba(247,147,26,0.2)'
                    ))

                    fig.add_trace(go.Scatter(
                        x=df['target_date'],
                        y=df['predicted_price'],
                        mode='lines+markers',
                        name='Predicted Price',
                        line=dict(color='#f7931a', width=2),
                        marker=dict(size=4)
                    ))

                    fig.update_layout(
                        title=f"{selected_symbol} Price Forecast",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        template='plotly_dark',
                        height=450
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    # =================================================
                    # Актуальные метрики моделей
                    # =================================================

                    st.subheader("Current Model Performance")

                    metrics_resp = requests.get(
                        f"{API_URL}/metrics/{selected_symbol}"
                    )

                    if metrics_resp.status_code == 200:

                        metrics = metrics_resp.json()

                        horizon_metrics = [
                            m for m in metrics
                            if m.get('horizon') == horizon
                        ]

                        if horizon_metrics:

                            latest = horizon_metrics[-1]

                            col1, col2 = st.columns(2)

                            with col1:

                                mape = latest.get('mape')

                                st.metric(
                                    "MAPE",
                                    f"{mape:.2f}%"
                                    if mape is not None
                                    else "N/A"
                                )

                            with col2:

                                mae = latest.get('mae')

                                st.metric(
                                    "MAE",
                                    f"${mae:,.2f}"
                                    if mae is not None
                                    else "N/A"
                                )

                        else:
                            st.info("No metrics available")

                    # =================================================
                    # Табличка с прогнозами
                    # =================================================

                    with st.expander("Show forecast table"):

                        display_df = df.copy()

                        display_df['target_date'] = (
                            display_df['target_date']
                            .dt.strftime('%Y-%m-%d')
                        )

                        display_df['predicted_price'] = (
                            display_df['predicted_price']
                            .apply(lambda x: f"${x:,.2f}")
                        )

                        st.dataframe(
                            display_df[[
                                'target_date',
                                'predicted_price'
                            ]],
                            use_container_width=True,
                            hide_index=True
                        )

                else:
                    st.warning(
                        f"No forecasts for {selected_symbol}"
                    )

            else:
                st.warning("Cannot load forecasts")

        except Exception as e:
            st.error(f"Error: {e}")

# ========== ВКЛАДКА 2: ИСТОРИЯ ==========
with tab2:

    st.header(
        f"{selected_symbol} — Historical Backtesting"
    )


    historical_horizon = st.radio(
        "Historical Forecast Horizon",
        options=[30, 180],
        format_func=lambda x: f"{x} days",
        horizontal=True,
        key="historical_horizon"
    )

    # =====================================================
    # Выбор периода (месяцы для 30 дней, полугодия для 180)
    # =====================================================

    today = date.today()

    if historical_horizon == 30:
        # Для 30 дней — месяцы
        periods = []
        for i in range(12):
            d = today.replace(day=1) - timedelta(days=30 * i)
            periods.append({
                "name": d.strftime('%B %Y'),
                "value": f"{d.year}-{d.month:02d}",
                "start_date": date(d.year, d.month, 1),
                "end_date": date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])
            })
        periods = sorted(periods, key=lambda x: x["start_date"], reverse=True)
        
        selected_period = st.selectbox(
            "Select month",
            options=[p["value"] for p in periods],
            format_func=lambda x: next(p["name"] for p in periods if p["value"] == x)
        )
        
        selected_period_data = next(p for p in periods if p["value"] == selected_period)
        start_date = selected_period_data["start_date"]
        end_date = selected_period_data["end_date"]
        
        st.write(f"**Period:** {selected_period_data['name']} ({start_date} → {end_date})")
        
    else:  # 180 дней — полугодовые периоды

        semesters = []

        # =====================================================
        # Создаем периоды
        # =====================================================

        for year in range(2024, today.year + 1):

            for half, months_range in [
                (1, (1, 6)),
                (2, (7, 12))
            ]:

                start = date(year, months_range[0], 1)

                end = date(
                    year,
                    months_range[1],
                    calendar.monthrange(
                        year,
                        months_range[1]
                    )[1]
                )

                # =============================================
                # Только периоды,
                # для которых уже доступны реальные данные
                # =============================================

                if end <= today - timedelta(days=180):

                    semesters.append({
                        "name": (
                            f"{year} "
                            f"H{half} "
                            f"({months_range[0]:02d}-"
                            f"{months_range[1]:02d})"
                        ),
                        "value": f"{year}-H{half}",
                        "start_date": start,
                        "end_date": end
                    })

        semesters = sorted(
            semesters,
            key=lambda x: x["start_date"],
            reverse=True
        )

        if not semesters:

            st.warning(
                "No semesters available "
                "for 180-day backtesting yet"
            )

            st.stop()

        selected_period = st.selectbox(
            "Select half-year period",
            options=[s["value"] for s in semesters],
            format_func=lambda x: next(
                s["name"]
                for s in semesters
                if s["value"] == x
            )
        )

        selected_period_data = next(
            s for s in semesters
            if s["value"] == selected_period
        )

        start_date = selected_period_data["start_date"]
        end_date = selected_period_data["end_date"]

        st.write(
            f"**Period:** "
            f"{selected_period_data['name']} "
            f"({start_date} → {end_date})"
        )
        
    with st.spinner("Loading historical backtesting..."):
        try:
            # =====================================================
            # Загрузка цены
            # =====================================================

            hist_resp = requests.get(
                f"{API_URL}/historical/{selected_symbol}",
                params={"days": 365}
            )

            if hist_resp.status_code != 200:
                st.error("Failed to load historical data")
                st.stop()

            historical = hist_resp.json()

            df_hist = pd.DataFrame(historical)

            if df_hist.empty:
                st.warning("No historical data available")
                st.stop()

            df_hist['date'] = pd.to_datetime(df_hist['date'])

            df_hist = df_hist[
                (df_hist['date'] >= pd.Timestamp(start_date)) &
                (df_hist['date'] <= pd.Timestamp(end_date))
            ]

            if df_hist.empty:
                st.warning(
                    f"No historical data for selected period"
                )
                st.stop()

            # =====================================================
            # Загрузка исторических данных
            # =====================================================

            forecasts_resp = requests.get(
                f"{API_URL}/forecast-history/{selected_symbol}",
                params={"horizon": historical_horizon}
            )

            if forecasts_resp.status_code != 200:
                st.warning("No historical forecasts found")
                st.stop()

            all_forecasts = forecasts_resp.json()

            if not all_forecasts:
                st.warning("Historical forecasts are empty")
                st.stop()

            # =====================================================
            # Фильтруем по выбранному периоду
            # =====================================================

            predictions = []

            for f in all_forecasts:
                target_date = pd.to_datetime(f["target_date"])
                
                if not (start_date <= target_date.date() <= end_date):
                    continue
                
                predictions.append({
                    "forecast_date": pd.to_datetime(f["forecast_date"]),
                    "target_date": target_date,
                    "predicted_price": f["predicted_price"]
                })

            if not predictions:
                st.warning(
                    f"No {historical_horizon}-day forecasts "
                    f"for selected period"
                )
                st.stop()

            df_pred = pd.DataFrame(predictions)

            # =====================================================
            # Берем свежий прогноз для нужной даты
            # =====================================================

            df_pred = (
                df_pred
                .sort_values("forecast_date")
                .groupby("target_date")
                .tail(1)
            )

            # =====================================================
            # Таблица сравнений
            # =====================================================

            comparison = []

            for _, pred in df_pred.iterrows():
                actual = df_hist[df_hist['date'] == pred['target_date']]
                
                if actual.empty:
                    continue
                
                actual_price = actual['price'].iloc[0]
                predicted_price = pred['predicted_price']
                
                if predicted_price is None:
                    continue
                
                error_abs = abs(actual_price - predicted_price)
                error_pct = (error_abs / actual_price) * 100
                
                comparison.append({
                    "Date": pred['target_date'],
                    "Actual": actual_price,
                    "Predicted": predicted_price,
                    "ErrorAbs": error_abs,
                    "ErrorPct": error_pct
                })

            if not comparison:
                st.warning("No matching actual values found")
                st.stop()

            df_comp = pd.DataFrame(comparison)

            # =====================================================
            # График
            # =====================================================

            fig_hist = go.Figure()

            fig_hist.add_trace(go.Scatter(
                x=df_hist['date'],
                y=df_hist['price'],
                mode='lines+markers',
                name='Actual Price',
                line=dict(color='#17BECF', width=2),
                marker=dict(size=5)
            ))

            fig_hist.add_trace(go.Scatter(
                x=df_comp['Date'],
                y=df_comp['Predicted'],
                mode='lines+markers',
                name=f'{historical_horizon}-day Forecast',
                line=dict(color='#f7931a', width=2),
                marker=dict(size=5)
            ))

            fig_hist.update_layout(
                title=(
                    f"{selected_symbol} "
                    f"{historical_horizon}-Day "
                    f"Historical Forecast Accuracy"
                ),
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template='plotly_dark',
                height=500
            )

            st.plotly_chart(fig_hist, use_container_width=True)

            # =====================================================
            # Метрики
            # =====================================================

            avg_mape = df_comp['ErrorPct'].mean()
            avg_mae = df_comp['ErrorAbs'].mean()

            col1, col2, col3 = st.columns(3)

            col1.metric("Average MAPE", f"{avg_mape:.2f}%")
            col2.metric("Average MAE", f"${avg_mae:,.2f}")
            col3.metric("Forecast Samples", len(df_comp))


            st.subheader("Forecast vs Actual")

            display_df = df_comp.copy()

            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            display_df['Actual'] = display_df['Actual'].apply(lambda x: f"${x:,.2f}")
            display_df['Predicted'] = display_df['Predicted'].apply(lambda x: f"${x:,.2f}")
            display_df['ErrorAbs'] = display_df['ErrorAbs'].apply(lambda x: f"${x:,.2f}")
            display_df['ErrorPct'] = display_df['ErrorPct'].apply(lambda x: f"{x:.2f}%")

            st.dataframe(
                display_df.rename(columns={
                    "ErrorAbs": "Absolute Error",
                    "ErrorPct": "Error %"
                }),
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:
            st.error(f"Error: {e}")