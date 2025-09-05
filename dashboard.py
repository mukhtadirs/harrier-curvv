#!/usr/bin/env python3
"""
TATA EV Configurator Analytics Dashboard

Interactive dashboard for monitoring loader drop-off rates across different models,
devices, and time periods with automatic comparison functionality.
"""

import os
import json
import tempfile
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, Any, List
from dotenv import load_dotenv

# Import our existing BigQuery logic
from alert_loader_dropoff import LoaderDropoffMonitor

# Load environment variables
load_dotenv()

# Attempt to load GCP creds from Streamlit secrets (preferred on hosted environments)
try:
    if hasattr(st, "secrets") and "gcp_service_account" in st.secrets and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        sa_dict = dict(st.secrets["gcp_service_account"])  # Streamlit secrets to dict
        fd, sa_path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(sa_dict, f)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
except Exception:
    # Fall back to GOOGLE_APPLICATION_CREDENTIALS env or ADC without raising here
    pass

# Page configuration
st.set_page_config(
    page_title="TATA EV Analytics Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Color scheme (aligned with dashboard primary blue)
PRIMARY = "#1f77b4"   # Blue (Streamlit primaryColor)
SUCCESS = "#2ca02c"   # Green
ACCENT  = "#ff7f0e"   # Orange
GRAY    = "#6b7280"   # Neutral gray

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    background: linear-gradient(90deg, #1f77b4, #ff7f0e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #ddd;
    text-align: center;
}
.alert-critical { border-left: 5px solid #ff4444; }
.alert-warning { border-left: 5px solid #ffaa00; }
.alert-good { border-left: 5px solid #44ff44; }
</style>
""", unsafe_allow_html=True)

def get_model_info():
    """Get available models and their dataset IDs"""
    return {
        "Harrier": {"dataset": "analytics_490128245", "icon": "🦌"},
        "Curvv": {"dataset": "analytics_452739489", "icon": "🏎️"}
    }

def _combine_dashboard_data(dashboards: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    """Combine multiple dashboard data dicts into one aggregated view."""
    if not dashboards:
        return {}

    # Use first date range and days
    first = dashboards[0]
    date_range = first["current_period"]["date_range"]
    days = first["current_period"]["days"]

    # Sum overall starts/ends
    total_starts = sum(d["current_period"]["overall"]["starts"] for d in dashboards)
    total_ends = sum(d["current_period"]["overall"]["ends"] for d in dashboards)
    dropoff_pct = round(((total_starts - total_ends) / total_starts * 100) if total_starts else 0, 2)

    # Aggregate devices
    device_key_to_stats: Dict[str, Dict[str, float]] = {}
    for d in dashboards:
        for rec in d["current_period"]["devices"]:
            key = rec["device"].lower()
            stats = device_key_to_stats.setdefault(key, {"starts": 0, "ends": 0})
            stats["starts"] += rec.get("starts", 0)
            stats["ends"] += rec.get("ends", 0)
    # Recompute device metrics and traffic share
    combined_devices: List[Dict[str, Any]] = []
    for key, stats in device_key_to_stats.items():
        starts = int(stats["starts"])
        ends = int(stats["ends"])
        combined_devices.append({
            "device": key,
            "starts": starts,
            "ends": ends,
            "dropoff_pct": round(((starts - ends) / starts * 100) if starts else 0, 2),
            # traffic_share computed after we know total_starts
        })
    for rec in combined_devices:
        rec["traffic_share"] = round((rec["starts"] / total_starts * 100) if total_starts else 0, 1)

    # Aggregate OS breakdown per device/os
    os_agg: Dict[tuple, Dict[str, float]] = {}
    for d in dashboards:
        for r in d["current_period"].get("os_breakdown", []):
            device_cat = (r.get("device_category") or "").lower()
            os_name = (r.get("os") or "").lower()
            key = (device_cat, os_name)
            entry = os_agg.setdefault(key, {"starts": 0, "ends": 0})
            entry["starts"] += r.get("loader_starts", 0)
            entry["ends"] += r.get("loader_ends", 0)

    # Compute totals per device category for pct_of_total_starts
    totals_per_device: Dict[str, float] = {}
    for (device_cat, _), vals in os_agg.items():
        totals_per_device[device_cat] = totals_per_device.get(device_cat, 0) + vals["starts"]

    combined_os: List[Dict[str, Any]] = []
    for (device_cat, os_name), vals in os_agg.items():
        starts = int(vals["starts"])  
        ends = int(vals["ends"])  
        combined_os.append({
            "device_category": device_cat,
            "os": os_name,
            "loader_starts": starts,
            "loader_ends": ends,
            "dropoff_pct": round(((starts - ends) / starts * 100) if starts else 0, 2),
            "pct_of_total_starts": round((starts / totals_per_device.get(device_cat, 1) * 100) if totals_per_device.get(device_cat, 0) else 0, 1),
        })

    # Combine previous period overall
    prev_starts = sum(d["previous_period"]["overall"]["starts"] for d in dashboards)
    prev_ends = sum(d["previous_period"]["overall"]["ends"] for d in dashboards)
    prev_drop = round(((prev_starts - prev_ends) / prev_starts * 100) if prev_starts else 0, 2)

    combined = {
        "current_period": {
            "date_range": date_range,
            "days": days,
            "overall": {
                "starts": total_starts,
                "ends": total_ends,
                "dropoff_count": total_starts - total_ends,
                "dropoff_pct": dropoff_pct,
            },
            "devices": combined_devices,
            "os_breakdown": combined_os,
        },
        "previous_period": {
            "overall": {
                "starts": prev_starts,
                "ends": prev_ends,
                "dropoff_pct": prev_drop,
            }
        },
        "threshold": threshold,
        "trends": {
            "overall": dropoff_pct - prev_drop,
        },
    }
    return combined

def get_dashboard_data(model_dataset: str, start_date: datetime, end_date: datetime, threshold: float) -> Dict[str, Any]:
    """Get comprehensive dashboard data for a model and date range"""
    
    # Calculate days for the period
    days_back = (end_date - start_date).days + 1
    
    # Set up environment for the monitor
    os.environ['BQ_PROJECT'] = 'tata-new-experience'
    os.environ['BQ_TABLE'] = model_dataset
    os.environ['THRESHOLD_PCT'] = str(threshold)
    os.environ['MOCK_MATTERMOST'] = 'true'
    # Credentials are loaded from Streamlit secrets above or from GOOGLE_APPLICATION_CREDENTIALS env
    
    try:
        # Get current period data
        monitor = LoaderDropoffMonitor()
        current_metrics = monitor.get_loader_metrics(days_back)

        # Normalize device breakdown keys to a consistent schema used by the dashboard
        raw_devices = current_metrics.get("device_breakdown", [])
        devices_mapped = []
        for d in raw_devices:
            device_name = (d.get("device_category") or d.get("device") or "unknown").lower()
            devices_mapped.append({
                "device": device_name,
                "starts": int(d.get("loader_starts", d.get("starts", 0)) or 0),
                "ends": int(d.get("loader_ends", d.get("ends", 0)) or 0),
                "dropoff_pct": float(d.get("dropoff_pct", 0) or 0),
                "traffic_share": float(d.get("pct_of_total_starts", d.get("traffic_share", 0)) or 0),
            })

        # Calculate previous period (same duration, immediately before current period)
        prev_end_date = start_date - timedelta(days=1)
        prev_start_date = prev_end_date - timedelta(days=days_back-1)
        prev_days_back = (prev_end_date - prev_start_date).days + 1
        
        # Get previous period data (we'll need to modify the date calculation in the monitor)
        # For now, let's use the 7-day data as previous period comparison
        previous_metrics = monitor.get_loader_metrics(days_back * 2)  # Approximate previous period
        
        # Structure the data for dashboard
        dashboard_data = {
            "current_period": {
                "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                "days": days_back,
                "overall": {
                    "starts": current_metrics["loader_starts"],
                    "ends": current_metrics["loader_ends"], 
                    "dropoff_count": current_metrics["loader_starts"] - current_metrics["loader_ends"],
                    "dropoff_pct": current_metrics["dropoff_pct"]
                },
                "devices": devices_mapped,
                "os_breakdown": current_metrics.get("os_breakdown", [])
            },
            "previous_period": {
                "overall": {
                    "starts": previous_metrics["loader_starts"],
                    "ends": previous_metrics["loader_ends"],
                    "dropoff_pct": previous_metrics["dropoff_pct"]
                }
            },
            "threshold": threshold
        }
        
        # Calculate trends
        dashboard_data["trends"] = {
            "overall": current_metrics["dropoff_pct"] - previous_metrics["dropoff_pct"]
        }
        
        return dashboard_data
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def create_hero_metrics(data: Dict[str, Any]):
    """Create the hero metrics section"""
    st.subheader("📊 Key Performance Indicators")
    
    current = data["current_period"]
    threshold = data["threshold"]
    overall_trend = data["trends"]["overall"]
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Overall metric
    with col1:
        overall_pct = current["overall"]["dropoff_pct"]
        status = "🚨" if overall_pct > threshold else "✅"
        st.metric(
            label=f"{status} Overall Drop-off",
            value=f"{overall_pct}%",
            delta=f"{overall_trend:+.1f}%",
            delta_color="inverse",
            help=f"Threshold: {threshold}% · Delta vs previous period"
        )
    
    # Device metrics
    devices = current["devices"]
    device_cols = [col2, col3, col4]
    device_names = ["mobile", "desktop", "tablet"]
    device_icons = ["📱", "💻", "📟"]
    
    for i, device_name in enumerate(device_names):
        device_data = next((d for d in devices if d["device"].lower() == device_name), None)
        
        with device_cols[i]:
            if device_data:
                device_pct = device_data["dropoff_pct"]
                if device_pct > threshold:
                    status = "🚨"
                elif device_pct > threshold * 0.8:
                    status = "⚠️"
                else:
                    status = "✅"
                
                st.metric(
                    label=f"{device_icons[i]} {device_name.title()}",
                    value=f"{device_pct}%",
                    delta=f"{device_data['starts']} starts",
                    help=f"Traffic share: {device_data.get('traffic_share', 0):.1f}%"
                )
            else:
                st.metric(
                    label=f"{device_icons[i]} {device_name.title()}",
                    value="No data",
                    help="No data available for this device"
                )

def create_traffic_funnel(data: Dict[str, Any]):
    """Create traffic flow funnel visualization"""
    st.subheader("🔄 Traffic Flow Analysis")
    
    current = data["current_period"]
    devices = current["devices"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        
        # Create funnel chart for overall traffic
        fig = go.Figure()
        
        # Add overall funnel
        fig.add_trace(go.Funnel(
            y=["Loader Starts", "Loader Ends"],
            x=[current["overall"]["starts"], current["overall"]["ends"]],
            texttemplate="%{x}",
            textposition="inside",
            name="Overall Traffic",
            marker=dict(color=[PRIMARY, SUCCESS]),
        ))
        
        fig.update_layout(
            title="Overall Traffic Funnel",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        
        # Create device comparison chart
        device_df = pd.DataFrame(devices)
        
        fig = px.bar(
            device_df,
            x="device",
            y=["starts", "ends"],
            title="Starts vs Ends by Device",
            barmode="group",
            color_discrete_map={"starts": PRIMARY, "ends": SUCCESS}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

def create_comparison_chart(data: Dict[str, Any]):
    """Create current vs previous period comparison"""
    st.subheader("📈 Period Comparison")
    
    current = data["current_period"]["overall"]["dropoff_pct"]
    previous = data["previous_period"]["overall"]["dropoff_pct"]
    
    comparison_data = pd.DataFrame({
        "Period": ["Current Period", "Previous Period"],
        "Drop-off Rate": [current, previous],
        "Status": ["Current", "Previous"]
    })
    
    fig = px.bar(
        comparison_data,
        x="Period",
        y="Drop-off Rate",
        color="Status",
        title=f"Drop-off Rate Comparison ({data['current_period']['date_range']})",
        color_discrete_map={"Current": PRIMARY, "Previous": ACCENT}
    )
    
    # Add threshold line
    fig.add_hline(y=data["threshold"], line_dash="dash", line_color="red", 
                  annotation_text=f"Threshold: {data['threshold']}%")
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def _build_figures_for_pdf(data: Dict[str, Any]):
    """Create Plotly figures (funnel, device bar, comparison) for PDF export."""
    current = data["current_period"]

    # Funnel
    funnel_fig = go.Figure()
    funnel_fig.add_trace(go.Funnel(
        y=["Loader Starts", "Loader Ends"],
        x=[current["overall"]["starts"], current["overall"]["ends"]],
        texttemplate="%{x}",
        textposition="inside",
        name="Overall Traffic",
    ))
    funnel_fig.update_layout(title="Overall Traffic Funnel", height=400, showlegend=False)

    # Device bar
    device_df = pd.DataFrame(current["devices"])
    device_bar = px.bar(
        device_df,
        x="device",
        y=["starts", "ends"],
        title="Starts vs Ends by Device",
        barmode="group",
        color_discrete_map={"starts": "#1f77b4", "ends": "#2ca02c"},
    )
    device_bar.update_layout(height=400)

    # Comparison bar
    comp_df = pd.DataFrame({
        "Period": ["Current Period", "Previous Period"],
        "Drop-off Rate": [current["overall"]["dropoff_pct"], data["previous_period"]["overall"]["dropoff_pct"]],
        "Status": ["Current", "Previous"],
    })
    comp_bar = px.bar(
        comp_df, x="Period", y="Drop-off Rate", color="Status",
        title=f"Drop-off Rate Comparison ({data['current_period']['date_range']})",
        color_discrete_map={"Current": "#ff7f0e", "Previous": "#1f77b4"},
    )
    comp_bar.add_hline(y=data["threshold"], line_dash="dash", line_color="red", annotation_text=f"Threshold: {data['threshold']}%")
    comp_bar.update_layout(height=400)

    return funnel_fig, device_bar, comp_bar

def build_pdf_report(data: Dict[str, Any], model_name: str, threshold: float) -> BytesIO:
    """Generate a consolidated PDF report containing charts and tables."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.units import inch

    # Build figures and render to images
    funnel_fig, device_bar, comp_bar = _build_figures_for_pdf(data)
    funnel_png = funnel_fig.to_image(format="png", width=1200, height=500, scale=2)
    device_png = device_bar.to_image(format="png", width=1200, height=500, scale=2)
    comp_png = comp_bar.to_image(format="png", width=1200, height=500, scale=2)

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    elements = []

    title = f"TATA EV Analytics Report — {model_name}"
    date_range = data["current_period"]["date_range"]
    elements.append(Paragraph(title, styles["Title"]))
    elements.append(Paragraph(f"Period: {date_range}", styles["Normal"]))
    elements.append(Paragraph(f"Threshold: {threshold}%", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # Charts
    elements.append(Paragraph("Overall Traffic Funnel", styles["Heading2"]))
    elements.append(RLImage(BytesIO(funnel_png), width=7.2*inch, height=3.0*inch))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Starts vs Ends by Device", styles["Heading2"]))
    elements.append(RLImage(BytesIO(device_png), width=7.2*inch, height=3.0*inch))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Drop-off Rate Comparison", styles["Heading2"]))
    elements.append(RLImage(BytesIO(comp_png), width=7.2*inch, height=3.0*inch))
    elements.append(Spacer(1, 18))

    # Device breakdown table
    elements.append(Paragraph("Device Breakdown", styles["Heading2"]))
    dev_rows = [["Device", "Starts", "Ends", "Drop-off %", "Traffic Share"]]
    for d in data["current_period"]["devices"]:
        dev_rows.append([
            d["device"].title(), d["starts"], d["ends"], f"{d['dropoff_pct']:.1f}%", f"{d.get('traffic_share', 0):.1f}%",
        ])
    dev_table = Table(dev_rows, repeatRows=1)
    dev_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
    ]))
    elements.append(dev_table)
    elements.append(Spacer(1, 18))

    # OS breakdown tables
    os_rows = data["current_period"].get("os_breakdown", [])
    if os_rows:
        elements.append(Paragraph("OS Breakdown — Mobile", styles["Heading2"]))
        mob_rows = [["OS", "Starts", "Ends", "Drop-off %", "Traffic Share"]]
        for r in os_rows:
            if (r.get("device_category") or "").lower() == "mobile":
                mob_rows.append([r.get("os",""), r.get("loader_starts",0), r.get("loader_ends",0), f"{float(r.get('dropoff_pct',0)):.1f}%", f"{float(r.get('pct_of_total_starts',0)):.1f}%"]) 
        if len(mob_rows) > 1:
            t = Table(mob_rows, repeatRows=1)
            t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.25,colors.grey),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("ALIGN",(1,1),(-1,-1),"RIGHT")]))
            elements.append(t)
            elements.append(Spacer(1, 12))

        elements.append(Paragraph("OS Breakdown — Desktop", styles["Heading2"]))
        desk_rows = [["OS", "Starts", "Ends", "Drop-off %", "Traffic Share"]]
        for r in os_rows:
            if (r.get("device_category") or "").lower() == "desktop":
                desk_rows.append([r.get("os",""), r.get("loader_starts",0), r.get("loader_ends",0), f"{float(r.get('dropoff_pct',0)):.1f}%", f"{float(r.get('pct_of_total_starts',0)):.1f}%"]) 
        if len(desk_rows) > 1:
            t = Table(desk_rows, repeatRows=1)
            t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.25,colors.grey),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("ALIGN",(1,1),(-1,-1),"RIGHT")]))
            elements.append(t)

    doc.build(elements)
    buffer.seek(0)
    return buffer
def create_detailed_table(data: Dict[str, Any]):
    """Create interactive detailed breakdown table"""
    st.subheader("📊 Detailed Device Breakdown")
    
    devices = data["current_period"]["devices"]
    threshold = data["threshold"]
    
    # Prepare data for table
    table_data = []
    total_starts = data["current_period"]["overall"]["starts"]
    
    for device in devices:
        traffic_share = (device["starts"] / total_starts * 100) if total_starts > 0 else 0
        
        # Status indicator
        if device["dropoff_pct"] > threshold:
            status = "🚨 Critical"
        elif device["dropoff_pct"] > threshold * 0.8:
            status = "⚠️ Warning"
        else:
            status = "✅ Good"
        
        table_data.append({
            "Device": device["device"].title(),
            "Starts": device["starts"],
            "Ends": device["ends"],
            "Drop-off Count": device["starts"] - device["ends"],
            "Drop-off %": f"{device['dropoff_pct']:.1f}%",
            "Traffic Share": f"{traffic_share:.1f}%",
            "Status": status
        })
    
    # Add overall row
    overall = data["current_period"]["overall"]
    table_data.append({
        "Device": "Overall",
        "Starts": overall["starts"],
        "Ends": overall["ends"], 
        "Drop-off Count": overall["dropoff_count"],
        "Drop-off %": f"{overall['dropoff_pct']:.1f}%",
        "Traffic Share": "100.0%",
        "Status": "🚨 Critical" if overall["dropoff_pct"] > threshold else "✅ Good"
    })
    
    df = pd.DataFrame(table_data)
    
    # Display interactive table
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
    )

    # OS breakdown per device category
    st.subheader("🧩 OS Breakdown (per device)")
    os_rows = data["current_period"].get("os_breakdown", [])
    if os_rows:
        # Normalize for dashboard table
        normalized = []
        for r in os_rows:
            device = (r.get("device_category") or r.get("device") or "unknown").title()
            normalized.append({
                "Device": device,
                "OS": r.get("os", "unknown"),
                "Starts": int(r.get("loader_starts", 0)),
                "Ends": int(r.get("loader_ends", 0)),
                "Drop-off %": f"{float(r.get('dropoff_pct', 0)):.1f}%",
                "Traffic Share": f"{float(r.get('pct_of_total_starts', 0)):.1f}%",
            })

        os_df = pd.DataFrame(normalized)
        # Split into Mobile and Desktop tables for clarity
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Mobile**")
            st.dataframe(os_df[os_df["Device"] == "Mobile"], hide_index=True, width="stretch")
        with col2:
            st.write("**Desktop**")
            st.dataframe(os_df[os_df["Device"] == "Desktop"], hide_index=True, width="stretch")
    else:
        st.info("No OS breakdown available for this period.")

def create_insights_section(data: Dict[str, Any]):
    """Create insights and recommendations section"""
    st.subheader("💡 Key Insights & Recommendations")
    
    devices = data["current_period"]["devices"]
    threshold = data["threshold"]
    overall_pct = data["current_period"]["overall"]["dropoff_pct"]
    
    insights = []
    
    # Overall status
    if overall_pct > threshold:
        insights.append(f"🚨 **Critical**: Overall drop-off rate ({overall_pct:.1f}%) exceeds threshold ({threshold}%)")
    else:
        insights.append(f"✅ **Good**: Overall drop-off rate ({overall_pct:.1f}%) is within acceptable limits")
    
    # Device-specific insights
    sorted_devices = sorted(devices, key=lambda x: x["dropoff_pct"], reverse=True)
    
    if sorted_devices:
        highest = sorted_devices[0]
        lowest = sorted_devices[-1]
        
        insights.append(f"📱 **Highest drop-off**: {highest['device'].title()} at {highest['dropoff_pct']:.1f}%")
        insights.append(f"🎯 **Lowest drop-off**: {lowest['device'].title()} at {lowest['dropoff_pct']:.1f}%")
        
        # Most used device
        most_used = max(devices, key=lambda x: x["starts"])
        total_starts = data["current_period"]["overall"]["starts"]
        usage_pct = (most_used["starts"] / total_starts * 100) if total_starts > 0 else 0
        insights.append(f"📊 **Most used device**: {most_used['device'].title()} ({usage_pct:.1f}% of traffic)")
    
    # Display insights
    for insight in insights:
        st.write(insight)
    
    # Recommendations
    st.write("**🎯 Recommendations:**")
    for device in sorted_devices:
        if device["dropoff_pct"] > threshold:
            st.write(f"• **{device['device'].title()} optimization needed**: {device['dropoff_pct']:.1f}% drop-off is above threshold")

# Main dashboard
def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 TATA EV Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        # Model selection
        with st.expander("🚗 Model Selection", expanded=True):
            model_options = get_model_info()
            selected_models = st.multiselect(
                "Select Vehicle Model(s)",
                options=list(model_options.keys()),
                default=list(model_options.keys())[:1],
                format_func=lambda x: f"{model_options[x]['icon']} {x}",
                help="Choose one or more models to analyze"
            )
        
        # Date range selection
        with st.expander("📅 Time Period", expanded=True):
            preset_ranges = {
                "Last 4 days": 4,
                "Last 7 days": 7, 
                "Last 30 days": 30,
                "Custom range": None
            }
            
            selected_preset = st.selectbox(
                "Quick Select",
                options=list(preset_ranges.keys()),
                index=0  # Default to "Last 4 days"
            )
            
            if preset_ranges[selected_preset]:
                days_back = preset_ranges[selected_preset]
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back-1)
                st.info(f"📅 {selected_preset}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=7))
                with col2:
                    end_date = st.date_input("End Date", value=datetime.now())
        
        # Analysis settings
        with st.expander("Set Threshold value"):
            threshold = st.slider(
                "Alert Threshold (%)",
                min_value=5.0,
                max_value=50.0,
                value=15.0,
                step=0.5,
                help="Drop-off percentage threshold for alerts"
            )
    
    # Generate report button
    if st.button("🔄 Generate Analytics Report", type="primary", width="stretch"):
        # Show loading and compute
        with st.spinner("Analyzing selected model(s) data..."):
            dashboards: List[Dict[str, Any]] = []
            for m in (selected_models or []):
                dataset_id = model_options[m]["dataset"]
                dashboards.append(get_dashboard_data(dataset_id, start_date, end_date, threshold))
            # If multi-selected, combine; if single, use directly
            dashboards = [d for d in dashboards if d]
            if len(dashboards) == 1:
                dashboard_data = dashboards[0]
                combined_label = (selected_models or ["model"])[0]
            else:
                dashboard_data = _combine_dashboard_data(dashboards, threshold)
                combined_label = "+".join(selected_models) if selected_models else "models"

        if dashboard_data:
            # Persist for subsequent interactions (e.g., PDF download)
            st.session_state["dashboard_data"] = dashboard_data
            st.session_state["selected_model"] = combined_label
            st.session_state["threshold"] = threshold

    # Display report if available in session (either just computed or from prior run)
    dashboard_data = st.session_state.get("dashboard_data")
    if dashboard_data:
        create_hero_metrics(dashboard_data)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            create_traffic_funnel(dashboard_data)
        with col2:
            create_comparison_chart(dashboard_data)
        st.markdown("---")
        create_detailed_table(dashboard_data)
        st.markdown("---")
        try:
            pdf_bytes = build_pdf_report(
                dashboard_data,
                st.session_state.get("selected_model", "model"),
                st.session_state.get("threshold", threshold),
            ).getvalue()
            # Prominent PDF button styled to match primary color
            st.markdown(
                f"""
                <style>
                .pdf-btn > button {{
                  background-color: {PRIMARY} !important;
                  color: white !important;
                  border: 1px solid {PRIMARY} !important;
                  width: 100% !important;
                  height: 3rem !important;
                  font-weight: 600 !important;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.download_button(
                label="📄 Download as PDF",
                data=pdf_bytes,
                file_name=f"analytics_{st.session_state.get('selected_model','model')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                key="download_pdf_btn",
                help="Download the full report as PDF",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Failed to generate PDF: {e}")
    else:
        st.info("👆 Select your model and date range, then click 'Generate Analytics Report' to view the dashboard.")

if __name__ == "__main__":
    main()
