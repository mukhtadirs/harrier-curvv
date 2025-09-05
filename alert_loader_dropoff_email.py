#!/usr/bin/env python3
"""
Loader Drop-off Email Alert Script

This script monitors 3D configurator loader drop-off rates using BigQuery
and sends email alerts when the drop-off exceeds the configured threshold.

Drop-off = (loader_starts - loader_ends) / loader_starts * 100
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account

# Load environment variables
load_dotenv()

class LoaderDropoffEmailMonitor:
    def __init__(self):
        # BigQuery configuration
        self.project_id = os.getenv('BQ_PROJECT')
        self.table_id = os.getenv('BQ_TABLE')
        self.threshold_pct = float(os.getenv('THRESHOLD_PCT', 15))
        
        # Email configuration
        self.from_email = os.getenv('FROM_EMAIL')
        self.to_email = os.getenv('TO_EMAIL')
        self.email_password = os.getenv('EMAIL_PASSWORD')  # App password for Gmail
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.mock_email = os.getenv('MOCK_EMAIL', 'false').lower() == 'true'
        
        # Initialize BigQuery client
        self.client = self._init_bigquery_client()
        
        # Validate required configuration
        if not self.project_id or not self.table_id:
            raise ValueError("BQ_PROJECT and BQ_TABLE environment variables are required")
        
        if not self.mock_email and (not self.from_email or not self.to_email or not self.email_password):
            raise ValueError("FROM_EMAIL, TO_EMAIL, and EMAIL_PASSWORD are required for email alerts")
    
    def _init_bigquery_client(self) -> bigquery.Client:
        """Initialize BigQuery client with service account credentials."""
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        if credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            return bigquery.Client(credentials=credentials, project=self.project_id)
        else:
            # Fallback to default credentials
            return bigquery.Client(project=self.project_id)
    
    def get_loader_metrics(self, days_back: int = 7) -> Dict[str, Any]:
        """Get loader metrics including device breakdown."""
        overall_metrics = self._get_overall_metrics(days_back)
        device_breakdown = self._get_device_breakdown(days_back)
        overall_metrics['device_breakdown'] = device_breakdown
        return overall_metrics
    
    def _get_overall_metrics(self, days_back: int = 7) -> Dict[str, Any]:
        """Query BigQuery for overall loader metrics."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        start_suffix = start_date.strftime('%Y%m%d')
        
        dataset_id = self.table_id.split('.')[0] if '.' in self.table_id else self.table_id
        
        if dataset_id == 'analytics_490128245':
            # Harrier query
            query = f"""
            WITH per_user AS (
                SELECT 
                    user_pseudo_id,
                    MAX(CASE WHEN LOWER(event_name) = 'loader' THEN 1 ELSE 0 END) AS has_loader_start,
                    MAX(CASE WHEN LOWER(event_name) = 'load_time_in_sec' THEN 1 ELSE 0 END) AS has_loader_end
                FROM `tata-new-experience.{dataset_id}.events_*`
                WHERE _TABLE_SUFFIX >= '{start_suffix}'
                    AND LOWER(event_name) IN ('loader', 'load_time_in_sec')
                GROUP BY user_pseudo_id
            )
            SELECT 
                SUM(has_loader_start) AS total_loader_starts,
                SUM(has_loader_end) AS total_loader_ends,
                ROUND(((SUM(has_loader_start) - SUM(has_loader_end)) / NULLIF(SUM(has_loader_start),0) * 100), 2) AS dropoff_pct
            FROM per_user
            WHERE has_loader_start = 1
            """
        else:
            # Curvv query
            query = f"""
            WITH loader_users AS (
                SELECT DISTINCT user_pseudo_id
                FROM `tata-new-experience.{dataset_id}.events_*`
                WHERE _TABLE_SUFFIX >= '{start_suffix}'
                    AND LOWER(event_name) = 'loader'
            ),
            loader_end_users AS (
                SELECT DISTINCT user_pseudo_id
                FROM `tata-new-experience.{dataset_id}.events_*`, UNNEST(event_params) ep
                WHERE _TABLE_SUFFIX >= '{start_suffix}'
                    AND LOWER(event_name) = 'loader'
                    AND (
                        LOWER(ep.value.string_value) IN ('end','completed','done') OR
                        (LOWER(ep.key) IN ('click','state','status','phase') AND LOWER(ep.value.string_value) IN ('end','completed','done'))
                    )
            )
            SELECT 
                (SELECT COUNT(*) FROM loader_users) AS total_loader_starts,
                (SELECT COUNT(*) FROM loader_end_users) AS total_loader_ends,
                ROUND((((SELECT COUNT(*) FROM loader_users) - (SELECT COUNT(*) FROM loader_end_users)) / NULLIF((SELECT COUNT(*) FROM loader_users), 0) * 100), 2) AS dropoff_pct
            """
        
        try:
            print(f"📊 Querying loader metrics for the last {days_back} days...")
            query_job = self.client.query(query)
            results = query_job.result()
            
            for row in results:
                return {
                    'loader_starts': int(row.total_loader_starts or 0),
                    'loader_ends': int(row.total_loader_ends or 0),
                    'dropoff_pct': float(row.dropoff_pct or 0),
                    'days_window': days_back,
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d')
                }
            
            return {
                'loader_starts': 0, 'loader_ends': 0, 'dropoff_pct': 0,
                'days_window': days_back,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            }
            
        except Exception as e:
            print(f"❌ Error querying BigQuery: {e}")
            raise
    
    def _get_device_breakdown(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """Query BigQuery for device breakdown."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        start_suffix = start_date.strftime('%Y%m%d')
        dataset_id = self.table_id.split('.')[0] if '.' in self.table_id else self.table_id
        
        if dataset_id == 'analytics_490128245':
            # Harrier device breakdown
            query = f"""
            WITH per_user AS (
              SELECT user_pseudo_id,
                     COALESCE(device.category, 'unknown') AS device_category,
                     MAX(CASE WHEN LOWER(event_name)='loader' THEN 1 ELSE 0 END) AS has_loader_start,
                     MAX(CASE WHEN LOWER(event_name)='load_time_in_sec' THEN 1 ELSE 0 END) AS has_loader_end
              FROM `tata-new-experience.{dataset_id}.events_*`
              WHERE _TABLE_SUFFIX >= '{start_suffix}'
                AND LOWER(event_name) IN ('loader','load_time_in_sec')
              GROUP BY user_pseudo_id, device_category
            )
            SELECT 
              device_category,
              SUM(has_loader_start) AS total_loader_starts,
              SUM(has_loader_end) AS total_loader_ends,
              ROUND(((SUM(has_loader_start) - SUM(has_loader_end)) / NULLIF(SUM(has_loader_start),0) * 100), 2) AS dropoff_pct,
              ROUND((SUM(has_loader_start) / (SELECT SUM(has_loader_start) FROM per_user WHERE has_loader_start = 1) * 100), 1) AS pct_of_total_starts
            FROM per_user
            WHERE has_loader_start = 1
            GROUP BY device_category
            ORDER BY total_loader_starts DESC
            """
        else:
            # Curvv device breakdown
            query = f"""
            WITH loader_users AS (
               SELECT user_pseudo_id, COALESCE(device.category, 'unknown') AS device_category
               FROM `tata-new-experience.{dataset_id}.events_*`
               WHERE _TABLE_SUFFIX >= '{start_suffix}' AND LOWER(event_name)='loader'
               GROUP BY user_pseudo_id, device_category
            ),
            loader_end_users AS (
               SELECT user_pseudo_id, COALESCE(device.category, 'unknown') AS device_category
               FROM `tata-new-experience.{dataset_id}.events_*`, UNNEST(event_params) ep
               WHERE _TABLE_SUFFIX >= '{start_suffix}' AND LOWER(event_name)='loader'
                 AND (LOWER(ep.value.string_value) IN ('end','completed','done') OR
                      (LOWER(ep.key) IN ('click','state','status','phase') AND LOWER(ep.value.string_value) IN ('end','completed','done')))
               GROUP BY user_pseudo_id, device_category
            )
            SELECT 
              device_category,
              (SELECT COUNT(*) FROM loader_users lu WHERE lu.device_category = l.device_category) AS total_loader_starts,
              (SELECT COUNT(*) FROM loader_end_users leu WHERE leu.device_category = l.device_category) AS total_loader_ends,
              ROUND(((SELECT COUNT(*) FROM loader_users lu WHERE lu.device_category = l.device_category) - 
                     (SELECT COUNT(*) FROM loader_end_users leu WHERE leu.device_category = l.device_category)) / 
                     NULLIF((SELECT COUNT(*) FROM loader_users lu WHERE lu.device_category = l.device_category), 0) * 100, 2) AS dropoff_pct,
              ROUND(((SELECT COUNT(*) FROM loader_users lu WHERE lu.device_category = l.device_category) / 
                     (SELECT COUNT(*) FROM loader_users) * 100), 1) AS pct_of_total_starts
            FROM (SELECT DISTINCT device_category FROM loader_users) l
            ORDER BY total_loader_starts DESC
            """
        
        try:
            print(f"📱 Querying device breakdown...")
            query_job = self.client.query(query)
            results = query_job.result()
            
            device_breakdown = []
            for row in results:
                device_breakdown.append({
                    'device_category': row.device_category,
                    'loader_starts': int(row.total_loader_starts or 0),
                    'loader_ends': int(row.total_loader_ends or 0),
                    'dropoff_pct': float(row.dropoff_pct or 0),
                    'pct_of_total_starts': float(row.pct_of_total_starts or 0)
                })
            
            return device_breakdown
            
        except Exception as e:
            print(f"❌ Error querying device breakdown: {e}")
            return []
    
    def send_email_alert(self, metrics: Dict[str, Any]) -> bool:
        """Send email alert with loader drop-off metrics."""
        
        if self.mock_email:
            print("📧 MOCK EMAIL MODE - Would send email:")
            print(f"   From: {self.from_email}")
            print(f"   To: {self.to_email}")
            print(f"   Subject: Loader Drop-off Alert: {metrics['dropoff_pct']}%")
            print(f"   Content: HTML email with device breakdown table")
            return True
        
        try:
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"🚨 Loader Drop-off Alert: {metrics['dropoff_pct']}% (>{self.threshold_pct}%)"
            msg['From'] = self.from_email
            msg['To'] = self.to_email
            
            # Create HTML content
            html_content = self._create_html_email(metrics)
            
            # Create plain text version
            text_content = self._create_text_email(metrics)
            
            # Attach both versions
            part_text = MIMEText(text_content, 'plain')
            part_html = MIMEText(html_content, 'html')
            
            msg.attach(part_text)
            msg.attach(part_html)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.from_email, self.email_password)
                server.send_message(msg)
            
            print(f"✅ Email alert sent successfully to {self.to_email}")
            return True
            
        except Exception as e:
            print(f"❌ Error sending email: {e}")
            return False
    
    def _create_html_email(self, metrics: Dict[str, Any]) -> str:
        """Create HTML email content with styled table."""
        
        # Get dataset name for title
        dataset_id = self.table_id.split('.')[0] if '.' in self.table_id else self.table_id
        dataset_name = "Harrier" if dataset_id == 'analytics_490128245' else "Curvv"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background: #ff6b6b; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .metrics-table th {{ background: #f2f2f2; font-weight: bold; }}
                .metrics-table .overall-row {{ background: #f9f9f9; font-weight: bold; }}
                .insights {{ background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .warning {{ color: #ff6b6b; font-weight: bold; }}
                .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🚨 {dataset_name} Loader Drop-off Alert</h2>
                <p><strong>{metrics['dropoff_pct']}% drop-off</strong> exceeds {self.threshold_pct}% threshold</p>
                <p>📅 Period: {metrics['start_date']} to {metrics['end_date']} ({metrics['days_window']} days)</p>
            </div>
        """
        
        # Add device breakdown table
        if 'device_breakdown' in metrics and metrics['device_breakdown']:
            device_data = sorted(metrics['device_breakdown'], 
                               key=lambda x: {'mobile': 0, 'desktop': 1}.get(x['device_category'].lower(), 2))
            
            html += """
            <h3>📱 Device Breakdown</h3>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Device</th>
                        <th>Starts</th>
                        <th>Ends</th>
                        <th>Drop-off %</th>
                        <th>% of Total</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for device in device_data:
                html += f"""
                    <tr>
                        <td>{device['device_category'].title()}</td>
                        <td>{device['loader_starts']:,}</td>
                        <td>{device['loader_ends']:,}</td>
                        <td>{device['dropoff_pct']:.1f}%</td>
                        <td>{device['pct_of_total_starts']:.1f}%</td>
                    </tr>
                """
            
            # Add overall row
            html += f"""
                    <tr class="overall-row">
                        <td>Overall</td>
                        <td>{metrics['loader_starts']:,}</td>
                        <td>{metrics['loader_ends']:,}</td>
                        <td>{metrics['dropoff_pct']:.1f}%</td>
                        <td>100.0%</td>
                    </tr>
                </tbody>
            </table>
            """
            
            # Add insights
            insights = self._generate_insights(device_data)
            if insights:
                html += f'<div class="insights"><h3>🔍 Key Insights</h3>{insights}</div>'
        
        html += f"""
            <div class="footer">
                <p>This is an automated alert from the Loader Drop-off Monitoring System.</p>
                <p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_text_email(self, metrics: Dict[str, Any]) -> str:
        """Create plain text email content."""
        dataset_id = self.table_id.split('.')[0] if '.' in self.table_id else self.table_id
        dataset_name = "Harrier" if dataset_id == 'analytics_490128245' else "Curvv"
        
        text = f"""
�� {dataset_name} Loader Drop-off Alert

Drop-off Rate: {metrics['dropoff_pct']}% (exceeds {self.threshold_pct}% threshold)
Period: {metrics['start_date']} to {metrics['end_date']} ({metrics['days_window']} days)

DEVICE BREAKDOWN:
"""
        
        if 'device_breakdown' in metrics and metrics['device_breakdown']:
            device_data = sorted(metrics['device_breakdown'], 
                               key=lambda x: {'mobile': 0, 'desktop': 1}.get(x['device_category'].lower(), 2))
            
            text += f"{'Device':<10} | {'Starts':<8} | {'Ends':<8} | {'Drop-off':<8} | {'% Total':<8}\n"
            text += "-" * 60 + "\n"
            
            for device in device_data:
                text += f"{device['device_category'].title():<10} | {device['loader_starts']:<8,} | {device['loader_ends']:<8,} | {device['dropoff_pct']:<8.1f}% | {device['pct_of_total_starts']:<8.1f}%\n"
            
            text += "-" * 60 + "\n"
            text += f"{'Overall':<10} | {metrics['loader_starts']:<8,} | {metrics['loader_ends']:<8,} | {metrics['dropoff_pct']:<8.1f}% | {'100.0':<8}%\n"
        
        text += f"\n---\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        
        return text
    
    def _generate_insights(self, device_data: List[Dict[str, Any]]) -> str:
        """Generate HTML insights from device breakdown."""
        if not device_data:
            return ""
        
        insights = ""
        
        # Find highest and lowest drop-off rates
        sorted_by_dropoff = sorted(device_data, key=lambda x: x['dropoff_pct'], reverse=True)
        highest_dropoff = sorted_by_dropoff[0]
        lowest_dropoff = sorted_by_dropoff[-1]
        most_used = max(device_data, key=lambda x: x['pct_of_total_starts'])
        
        insights += f"<ul>"
        insights += f"<li><strong>Highest drop-off:</strong> {highest_dropoff['device_category'].title()} ({highest_dropoff['dropoff_pct']:.1f}%)</li>"
        insights += f"<li><strong>Lowest drop-off:</strong> {lowest_dropoff['device_category'].title()} ({lowest_dropoff['dropoff_pct']:.1f}%)</li>"
        insights += f"<li><strong>Most used device:</strong> {most_used['device_category'].title()} ({most_used['pct_of_total_starts']:.1f}% of sessions)</li>"
        
        # Add mobile warning if needed
        mobile_device = next((d for d in device_data if d['device_category'].lower() == 'mobile'), None)
        if mobile_device and mobile_device['dropoff_pct'] > 25:
            insights += f'<li class="warning">⚠️ <strong>Mobile optimization needed:</strong> {mobile_device["dropoff_pct"]:.1f}% mobile drop-off is high</li>'
        
        insights += "</ul>"
        return insights
    
    def run_check(self, days_back: int = 7) -> None:
        """Run the complete loader drop-off check with email alerts."""
        try:
            print(f"🔍 Starting loader drop-off email check...")
            
            # Get metrics from BigQuery
            metrics = self.get_loader_metrics(days_back)
            
            # Log summary
            print(f"📊 Loader metrics: {metrics['dropoff_pct']}% drop-off ({metrics['loader_starts']:,} starts, {metrics['loader_ends']:,} ends)")
            
            # Log device breakdown
            if 'device_breakdown' in metrics and metrics['device_breakdown']:
                print(f"📱 Device breakdown:")
                for device in metrics['device_breakdown']:
                    print(f"   • {device['device_category'].title()}: {device['dropoff_pct']}% drop-off ({device['loader_starts']:,} starts, {device['pct_of_total_starts']:.1f}% of total)")
            
            # Check threshold and send email if needed
            if metrics['dropoff_pct'] > self.threshold_pct:
                print(f"🚨 Threshold exceeded! {metrics['dropoff_pct']}% > {self.threshold_pct}%")
                
                if self.send_email_alert(metrics):
                    print(f"✅ Email alert process completed successfully")
                else:
                    print(f"❌ Email alert process failed")
            else:
                print(f"✅ Drop-off within acceptable range ({metrics['dropoff_pct']}% <= {self.threshold_pct}%)")
            
        except Exception as e:
            print(f"❌ Error during email check: {e}")
            raise


def main():
    """Main function to run the email monitor."""
    try:
        monitor = LoaderDropoffEmailMonitor()
        monitor.run_check()
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
