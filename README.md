# Loader Drop-off Alert — README

This project sets up a simple **weekly alert** for the 3D configurator loader drop-off using **BigQuery** + **Mattermost**.

The alert is triggered if **loader drop-off > 15%**.
Drop-off = `(loader_starts - loader_ends) / loader_starts * 100`

---

## Requirements

* Python 3.10+
* Google Cloud service account with **read-only access** to BigQuery dataset
* Mattermost channel with **incoming webhook URL**

---

## Setup

1. **Folder structure**

   ```
   .
   ├─ alert_loader_dropoff.py
   ├─ .env.example
   └─ README.md
   ```

2. **Install dependencies**

   ```bash
   pip install google-cloud-bigquery requests python-dotenv
   ```

3. **Create service account key** in Google Cloud Console

   * Grant it `BigQuery Data Viewer` role
   * Download JSON file (e.g., `tata_mcp_bq_server.json`)

4. **Prepare environment variables**
   Create a `.env` file (not committed) or set environment variables:

   ```bash
   # Google Cloud BigQuery Configuration
   # Prefer secret stores. For local dev only, keep this JSON outside the repo
   # GOOGLE_APPLICATION_CREDENTIALS=/Users/you/secrets/sa.json
   export BQ_PROJECT=tata-new-experience
   
   # Dataset ID - choose one:
   # For Harrier: analytics_490128245
   # For Curvv: analytics_452739489
   export BQ_TABLE=analytics_490128245
   
   # Alert Configuration
   export THRESHOLD_PCT=15
   
   # Mattermost Integration (leave empty for mock mode)
   export MATTERMOST_WEBHOOK=
   
   # Mock Mode (set to 'true' to print alerts instead of sending)
   export MOCK_MATTERMOST=true
   ```

5. **Security & secrets**

   - Do not commit credentials. `.gitignore` already ignores common secret patterns and `.env`.
   - Streamlit (hosted): set `gcp_service_account` in Secrets and `BQ_PROJECT`.
   - Render: create a Secret File with the SA JSON (e.g., `/opt/render/project/src/sa.json`) and set `GOOGLE_APPLICATION_CREDENTIALS` to that path.
   - GitHub Actions: store `BQ_SERVICE_ACCOUNT_JSON` and `MATTERMOST_WEBHOOK` in repo Secrets.
   - If a key is ever committed, revoke and rotate in GCP immediately and purge history.

---

## Python Script Responsibilities

The script (`alert_loader_dropoff.py`) automatically:

1. **Authenticates** with BigQuery using `GOOGLE_APPLICATION_CREDENTIALS`.
2. **Detects dataset type** (Harrier vs Curvv) and runs appropriate SQL:

   * **Harrier** (`analytics_490128245`): Uses `loader` → `load_time_in_sec` events
   * **Curvv** (`analytics_452739489`): Uses `loader` → loader completion params
   * Calculates drop-off percentage per unique user over the last 7 days.
3. **Compares** drop-off against `THRESHOLD_PCT`.
4. **If `dropoff_pct > THRESHOLD_PCT`**:

   * In **mock mode**: Prints alert to console (great for testing!)
   * In **live mode**: Sends JSON POST to `MATTERMOST_WEBHOOK`
   * Example message:

     ```
     ⚠️ Loader drop-off alert: 24.48% (>15.0%)
     Starts: 678  Ends: 512
     Window: last 7 days (2025-08-23 to 2025-08-30)
     ```
5. **Always** prints a summary to console for logs.

---

## Testing Locally

1. **Set up virtual environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the script** with environment variables:

   ```bash
   # Test with mock mode (safe for development)
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json \
   BQ_PROJECT=tata-new-experience \
   BQ_TABLE=analytics_490128245 \
   THRESHOLD_PCT=15 \
   MOCK_MATTERMOST=true \
   python3 alert_loader_dropoff.py
   ```

3. **Example output**:

   ```
   🔍 Starting loader drop-off check...
   Querying loader metrics for the last 7 days...
   📊 Loader metrics: 24.48% drop-off (678 starts, 512 ends) over 7 days
   🚨 Threshold exceeded! 24.48% > 15.0%
   🔧 MOCK MODE - Would send to Mattermost:
      ⚠️ Loader drop-off alert: 24.48% (>15.0%)
   Starts: 678  Ends: 512
   Window: last 7 days (2025-08-23 to 2025-08-30)
   ✅ Alert process completed successfully
   ```

4. **For production**: Set `MOCK_MATTERMOST=false` and provide `MATTERMOST_WEBHOOK` URL.

---

## Automation with GitHub Actions

To run once a week automatically:

1. Create file: `.github/workflows/loader_alert.yml`

   ```yaml
   name: Loader Drop-off Alert

   on:
     schedule:
       # Every Monday at 10:00 UTC
       - cron: "0 10 * * 1"

   jobs:
     check-dropoff:
       runs-on: ubuntu-latest
       steps:
         - name: Checkout repo
           uses: actions/checkout@v4

         - name: Set up Python
           uses: actions/setup-python@v5
           with:
             python-version: '3.11'

         - name: Install dependencies
           run: pip install -r requirements.txt

         - name: Run alert script
           env:
             GOOGLE_APPLICATION_CREDENTIALS: ${{ secrets.BQ_SERVICE_ACCOUNT_JSON }}
             BQ_PROJECT: "tata-new-experience"
             BQ_TABLE: "analytics_490128245"  # or analytics_452739489 for Curvv
             THRESHOLD_PCT: "15"
             MATTERMOST_WEBHOOK: ${{ secrets.MATTERMOST_WEBHOOK }}
             MOCK_MATTERMOST: "false"  # Use real webhook in production
           run: python3 alert_loader_dropoff.py
   ```

2. Store secrets in GitHub:

   * `BQ_SERVICE_ACCOUNT_JSON` → contents of your service account JSON
   * `MATTERMOST_WEBHOOK` → your Mattermost webhook URL

---

## Summary

* ✅ **Script tested and working** with Tata EV Analytics BigQuery data
* 🔧 **Mock mode** allows safe testing without Mattermost webhook
* 📊 **Supports both Harrier and Curvv** datasets automatically
* 🚨 **Real data example**: Currently showing 24.48% drop-off (678 starts, 512 ends)
* ⏰ **Weekly automation** via GitHub Actions when webhook is available
* 🛡️ **Error handling** and comprehensive logging included

## Current Status

The Python script is **fully functional** and ready for production use. You can:

1. **Test immediately** in mock mode (no webhook required)
2. **Deploy to production** once Mattermost webhook is available
3. **Monitor both Harrier and Curvv** by changing the `BQ_TABLE` environment variable

## Next Steps

1. Get Mattermost webhook URL from admin
2. Set `MOCK_MATTERMOST=false` and `MATTERMOST_WEBHOOK=<your-url>`
3. Set up GitHub Actions for weekly automation
