#!/usr/bin/env python3
"""
Loader Drop-off Alert Script

This script monitors 3D configurator loader drop-off rates using BigQuery
and sends alerts when the drop-off exceeds the configured threshold.

Drop-off = (loader_starts - loader_ends) / loader_starts * 100
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account

# Load environment variables
load_dotenv()

class LoaderDropoffMonitor:
    def __init__(self):
        self.project_id = os.getenv('BQ_PROJECT')
        self.table_id = os.getenv('BQ_TABLE')
        self.threshold_pct = float(os.getenv('THRESHOLD_PCT', 15))
        self.webhook_url = os.getenv('MATTERMOST_WEBHOOK')
        self.mock_mode = os.getenv('MOCK_MATTERMOST', 'false').lower() == 'true'
        
        # Initialize BigQuery client
        self.client = self._init_bigquery_client()
        
        if not self.project_id or not self.table_id:
            raise ValueError("BQ_PROJECT and BQ_TABLE environment variables are required")
    
    def _init_bigquery_client(self) -> bigquery.Client:
        """Initialize BigQuery client with service account credentials."""
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        if credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            return bigquery.Client(credentials=credentials, project=self.project_id)
        else:
            # Fallback to default credentials (useful for local development)
            return bigquery.Client(project=self.project_id)
    
    def get_loader_metrics(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Query BigQuery for loader start/end metrics over the specified time window.
        Uses Tata EV Analytics GA4 export schema.
        
        Args:
            days_back: Number of days to look back for metrics
            
        Returns:
            Dictionary containing loader_starts, loader_ends, dropoff_pct, and device_breakdown
        """
        # Get overall metrics
        overall_metrics = self._get_overall_metrics(days_back)
        
        # Get device breakdown
        device_breakdown = self._get_device_breakdown(days_back)

        # Get OS breakdown (per device category)
        os_breakdown = self._get_os_breakdown(days_back)
        
        # Combine results
        overall_metrics['device_breakdown'] = device_breakdown
        overall_metrics['os_breakdown'] = os_breakdown
        return overall_metrics
    
    def _get_overall_metrics(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Query BigQuery for loader start/end metrics over the specified time window.
        Uses Tata EV Analytics GA4 export schema.
        
        Args:
            days_back: Number of days to look back for metrics
            
        Returns:
            Dictionary containing loader_starts, loader_ends, and dropoff_pct
        """
        # Calculate date range for _TABLE_SUFFIX filtering
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        start_suffix = start_date.strftime('%Y%m%d')
        
        # Determine if this is Harrier or Curvv based on table_id
        dataset_id = self.table_id.split('.')[0] if '.' in self.table_id else self.table_id
        
        if dataset_id == 'analytics_490128245':
            # Harrier: loader event + load_time_in_sec event
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
                CASE 
                    WHEN SUM(has_loader_start) > 0 
                    THEN ROUND(((SUM(has_loader_start) - SUM(has_loader_end)) / SUM(has_loader_start) * 100), 2)
                    ELSE 0 
                END AS dropoff_pct
            FROM per_user
            WHERE has_loader_start = 1  -- Only count users who started loading
            """
        else:
            # Curvv: loader event + loader with completion params
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
                CASE 
                    WHEN (SELECT COUNT(*) FROM loader_users) > 0 
                    THEN ROUND((((SELECT COUNT(*) FROM loader_users) - (SELECT COUNT(*) FROM loader_end_users)) / (SELECT COUNT(*) FROM loader_users) * 100), 2)
                    ELSE 0 
                END AS dropoff_pct
            """
        
        try:
            print(f"Querying loader metrics for the last {days_back} days...")
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
            
            # If no results, return zeros
            return {
                'loader_starts': 0,
                'loader_ends': 0,
                'dropoff_pct': 0,
                'days_window': days_back,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            }
            
        except Exception as e:
            print(f"Error querying BigQuery: {e}")
            raise
    
    def _get_device_breakdown(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Query BigQuery for device breakdown of loader metrics.
        
        Args:
            days_back: Number of days to look back for metrics
            
        Returns:
            List of dictionaries containing device-specific metrics
        """
        # Calculate date range for _TABLE_SUFFIX filtering
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        start_suffix = start_date.strftime('%Y%m%d')
        
        # Determine if this is Harrier or Curvv based on table_id
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
               SELECT user_pseudo_id,
                      COALESCE(device.category, 'unknown') AS device_category
               FROM `tata-new-experience.{dataset_id}.events_*`
               WHERE _TABLE_SUFFIX >= '{start_suffix}'
                 AND LOWER(event_name)='loader'
               GROUP BY user_pseudo_id, device_category
            ),
            loader_end_users AS (
               SELECT user_pseudo_id,
                      COALESCE(device.category, 'unknown') AS device_category
               FROM `tata-new-experience.{dataset_id}.events_*`, UNNEST(event_params) ep
               WHERE _TABLE_SUFFIX >= '{start_suffix}'
                 AND LOWER(event_name)='loader'
                 AND (
                   LOWER(ep.value.string_value) IN ('end','completed','done') OR
                   (LOWER(ep.key) IN ('click','state','status','phase') AND LOWER(ep.value.string_value) IN ('end','completed','done'))
                 )
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
            print(f"Querying device breakdown for the last {days_back} days...")
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
            print(f"Error querying device breakdown: {e}")
            return []

    def _get_os_breakdown(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """Return OS breakdown by device category (mobile and desktop)."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        start_suffix = start_date.strftime('%Y%m%d')

        dataset_id = self.table_id.split('.')[0] if '.' in self.table_id else self.table_id

        if dataset_id == 'analytics_490128245':
            # Harrier OS breakdown using loader + load_time_in_sec
            query = f"""
            WITH per_user AS (
              SELECT user_pseudo_id,
                     COALESCE(device.category, 'unknown') AS device_category,
                     COALESCE(device.operating_system, 'unknown') AS os_name,
                     MAX(CASE WHEN LOWER(event_name)='loader' THEN 1 ELSE 0 END) AS has_loader_start,
                     MAX(CASE WHEN LOWER(event_name)='load_time_in_sec' THEN 1 ELSE 0 END) AS has_loader_end
              FROM `tata-new-experience.{dataset_id}.events_*`
              WHERE _TABLE_SUFFIX >= '{start_suffix}'
                AND LOWER(event_name) IN ('loader','load_time_in_sec')
              GROUP BY user_pseudo_id, device_category, os_name
            ),
            starts AS (
              SELECT device_category, os_name, SUM(has_loader_start) AS starts
              FROM per_user
              WHERE has_loader_start = 1
              GROUP BY device_category, os_name
            ),
            ends AS (
              SELECT device_category, os_name, SUM(has_loader_end) AS ends
              FROM per_user
              WHERE has_loader_start = 1
              GROUP BY device_category, os_name
            ),
            totals AS (
              SELECT s.device_category, SUM(s.starts) AS device_category_total
              FROM starts s
              GROUP BY s.device_category
            )
            SELECT 
              s.device_category,
              s.os_name,
              s.starts AS total_loader_starts,
              COALESCE(e.ends, 0) AS total_loader_ends,
              ROUND(((s.starts - COALESCE(e.ends,0)) / NULLIF(s.starts,0) * 100), 2) AS dropoff_pct,
              ROUND(s.starts * 100 / NULLIF(t.device_category_total, 0), 1) AS pct_of_total_starts
            FROM starts s
            LEFT JOIN ends e ON e.device_category = s.device_category AND e.os_name = s.os_name
            LEFT JOIN totals t ON t.device_category = s.device_category
            ORDER BY s.device_category, total_loader_starts DESC
            """
        else:
            # Curvv OS breakdown using loader + completion params
            query = f"""
            WITH loader_users AS (
               SELECT user_pseudo_id,
                      COALESCE(device.category, 'unknown') AS device_category,
                      COALESCE(device.operating_system, 'unknown') AS os_name
               FROM `tata-new-experience.{dataset_id}.events_*`
               WHERE _TABLE_SUFFIX >= '{start_suffix}'
                 AND LOWER(event_name)='loader'
               GROUP BY user_pseudo_id, device_category, os_name
            ),
            loader_end_users AS (
               SELECT user_pseudo_id,
                      COALESCE(device.category, 'unknown') AS device_category,
                      COALESCE(device.operating_system, 'unknown') AS os_name
               FROM `tata-new-experience.{dataset_id}.events_*`, UNNEST(event_params) ep
               WHERE _TABLE_SUFFIX >= '{start_suffix}'
                 AND LOWER(event_name)='loader'
                 AND (
                   LOWER(ep.value.string_value) IN ('end','completed','done') OR
                   (LOWER(ep.key) IN ('click','state','status','phase') AND LOWER(ep.value.string_value) IN ('end','completed','done'))
                 )
               GROUP BY user_pseudo_id, device_category, os_name
            ),
            starts AS (
               SELECT device_category, os_name, COUNT(DISTINCT user_pseudo_id) AS starts
               FROM loader_users
               GROUP BY device_category, os_name
            ),
            ends AS (
               SELECT device_category, os_name, COUNT(DISTINCT user_pseudo_id) AS ends
               FROM loader_end_users
               GROUP BY device_category, os_name
            ),
            totals AS (
               SELECT s.device_category, SUM(s.starts) AS device_category_total
               FROM starts s
               GROUP BY s.device_category
            )
            SELECT s.device_category,
                   s.os_name,
                   s.starts AS total_loader_starts,
                   COALESCE(e.ends,0) AS total_loader_ends,
                   ROUND(((s.starts - COALESCE(e.ends,0)) / NULLIF(s.starts,0) * 100), 2) AS dropoff_pct,
                   ROUND(s.starts * 100 / NULLIF(t.device_category_total, 0), 1) AS pct_of_total_starts
            FROM starts s
            LEFT JOIN ends e ON e.device_category = s.device_category AND e.os_name = s.os_name
            LEFT JOIN totals t ON t.device_category = s.device_category
            ORDER BY s.device_category, total_loader_starts DESC
            """

        try:
            query_job = self.client.query(query)
            results = query_job.result()

            os_breakdown: List[Dict[str, Any]] = []
            for row in results:
                # Only include mobile and desktop as requested
                if str(row.device_category).lower() in ('mobile', 'desktop'):
                    os_breakdown.append({
                        'device_category': row.device_category,
                        'os': row.os_name,
                        'loader_starts': int(row.total_loader_starts or 0),
                        'loader_ends': int(row.total_loader_ends or 0),
                        'dropoff_pct': float(row.dropoff_pct or 0),
                        'pct_of_total_starts': float(row.pct_of_total_starts or 0)
                    })

            return os_breakdown
        except Exception as e:
            print(f"Error querying OS breakdown: {e}")
            return []
    
    def send_alert(self, metrics: Dict[str, Any]) -> bool:
        """
        Send alert to Mattermost or print to console in mock mode.
        
        Args:
            metrics: Dictionary containing loader metrics
            
        Returns:
            True if alert was sent successfully, False otherwise
        """
        message = self._format_alert_message(metrics)
        
        if self.mock_mode or not self.webhook_url:
            print("🔧 MOCK MODE - Would send to Mattermost:")
            print(f"   {message}")
            return True
        
        try:
            payload = {
                "text": message,
                "username": "Loader Alert Bot",
                "icon_emoji": ":warning:"
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"✅ Alert sent to Mattermost successfully")
                return True
            else:
                print(f"❌ Failed to send alert: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error sending alert to Mattermost: {e}")
            return False
    
    def _get_vehicle_info(self) -> Dict[str, str]:
        """Get OEM and Model information based on dataset."""
        dataset_id = self.table_id.split('.')[0] if '.' in self.table_id else self.table_id
        
        if dataset_id == 'analytics_490128245':
            return {'oem': 'TATA', 'model': 'Harrier'}
        elif dataset_id == 'analytics_452739489':
            return {'oem': 'TATA', 'model': 'Curvv'}
        else:
            return {'oem': 'Unknown', 'model': 'Unknown'}
    
    def _format_alert_message(self, metrics: Dict[str, Any]) -> str:
        """Format the alert message for Mattermost with nice table layout."""
        vehicle_info = self._get_vehicle_info()
        
        header = (
            f"🚨 **{vehicle_info['oem']} {vehicle_info['model']} - Loader Drop-off Alert**\n"
            f"⚠️ **{metrics['dropoff_pct']}% drop-off** exceeds {self.threshold_pct}% threshold\n"
            f"🏭 **OEM**: {vehicle_info['oem']} | �� **Model**: {vehicle_info['model']}\n"
            f"📅 **Window**: {metrics['days_window']} days ({metrics['start_date']} to {metrics['end_date']})\n"
        )
        
        # Create device breakdown table
        if 'device_breakdown' in metrics and metrics['device_breakdown']:
            # Sort devices: mobile, desktop, others, then add overall
            device_data = sorted(metrics['device_breakdown'], 
                               key=lambda x: {'mobile': 0, 'desktop': 1}.get(x['device_category'].lower(), 2))
            
            # Add overall row
            overall_row = {
                'device_category': 'Overall',
                'loader_starts': metrics['loader_starts'],
                'loader_ends': metrics['loader_ends'],
                'dropoff_pct': metrics['dropoff_pct'],
                'pct_of_total_starts': 100.0
            }
            device_data.append(overall_row)
            
            # Calculate column widths for alignment
            max_device_width = max(len(device['device_category']) for device in device_data)
            max_starts_width = max(len(f"{device['loader_starts']:,}") for device in device_data)
            max_ends_width = max(len(f"{device['loader_ends']:,}") for device in device_data)
            
            # Create table header
            table = "\n```\n"
            table += f"{'Device':<{max_device_width}} │ {'Starts':>{max_starts_width}} │ {'Ends':>{max_ends_width}} │ Drop-off │ % Total\n"
            table += "─" * (max_device_width + max_starts_width + max_ends_width + 24) + "\n"
            
            # Add device rows
            for device in device_data:
                device_name = device['device_category'].title()
                starts = f"{device['loader_starts']:,}"
                ends = f"{device['loader_ends']:,}"
                dropoff = f"{device['dropoff_pct']:5.1f}%"
                pct_total = f"{device['pct_of_total_starts']:5.1f}%"
                
                # Add separator before Overall row
                if device['device_category'] == 'Overall':
                    table += "─" * (max_device_width + max_starts_width + max_ends_width + 24) + "\n"
                
                table += f"{device_name:<{max_device_width}} │ {starts:>{max_starts_width}} │ {ends:>{max_ends_width}} │ {dropoff:>8} │ {pct_total:>7}\n"
            
            table += "```"
            
            # Add insights
            insights = self._generate_insights(device_data[:-1])  # Exclude overall row
            
            return header + table + insights
        else:
            # Fallback for no device data
            return (
                header + 
                f"📊 **Overall**: {metrics['loader_starts']:,} starts → {metrics['loader_ends']:,} ends"
            )
    
    def _generate_insights(self, device_data: List[Dict[str, Any]]) -> str:
        """Generate insights from device breakdown data."""
        if not device_data:
            return ""
        
        insights = "\n\n🔍 **Key Insights:**\n"
        
        # Find highest and lowest drop-off rates
        sorted_by_dropoff = sorted(device_data, key=lambda x: x['dropoff_pct'], reverse=True)
        highest_dropoff = sorted_by_dropoff[0]
        lowest_dropoff = sorted_by_dropoff[-1]
        
        # Find most used device
        most_used = max(device_data, key=lambda x: x['pct_of_total_starts'])
        
        insights += f"• **Highest drop-off**: {highest_dropoff['device_category'].title()} ({highest_dropoff['dropoff_pct']:.1f}%)\n"
        insights += f"• **Lowest drop-off**: {lowest_dropoff['device_category'].title()} ({lowest_dropoff['dropoff_pct']:.1f}%)\n"
        insights += f"• **Most used device**: {most_used['device_category'].title()} ({most_used['pct_of_total_starts']:.1f}% of sessions)\n"
        
        # Add recommendation if mobile has high drop-off
        mobile_device = next((d for d in device_data if d['device_category'].lower() == 'mobile'), None)
        if mobile_device and mobile_device['dropoff_pct'] > 25:
            insights += f"• ⚠️ **Mobile optimization needed**: {mobile_device['dropoff_pct']:.1f}% mobile drop-off is high\n"
        
        return insights
    
    def run_check(self, days_back: int = 7) -> None:
        """
        Run the complete loader drop-off check.
        
        Args:
            days_back: Number of days to analyze
        """
        try:
            print(f"🔍 Starting loader drop-off check...")
            
            # Get metrics from BigQuery
            metrics = self.get_loader_metrics(days_back)
            
            # Log summary
            summary = (
                f"Loader metrics: {metrics['dropoff_pct']}% drop-off "
                f"({metrics['loader_starts']:,} starts, {metrics['loader_ends']:,} ends) "
                f"over {metrics['days_window']} days"
            )
            print(f"📊 {summary}")
            
            # Log device breakdown
            if 'device_breakdown' in metrics and metrics['device_breakdown']:
                print(f"📱 Device breakdown:")
                for device in metrics['device_breakdown']:
                    print(f"   • {device['device_category'].title()}: "
                          f"{device['dropoff_pct']}% drop-off "
                          f"({device['loader_starts']:,} starts, {device['loader_ends']:,} ends, "
                          f"{device['pct_of_total_starts']}% of total)")
                print()
            
            # Check if alert threshold is exceeded
            if metrics['dropoff_pct'] > self.threshold_pct:
                print(f"🚨 Threshold exceeded! {metrics['dropoff_pct']}% > {self.threshold_pct}%")
                alert_sent = self.send_alert(metrics)
                
                if alert_sent:
                    print(f"✅ Alert process completed successfully")
                else:
                    print(f"❌ Alert process failed")
            else:
                print(f"✅ Drop-off within acceptable range ({metrics['dropoff_pct']}% <= {self.threshold_pct}%)")
            
        except Exception as e:
            error_msg = f"❌ Error during loader drop-off check: {e}"
            print(error_msg)
            raise


def main():
    """Main function to run the loader drop-off monitor."""
    try:
        monitor = LoaderDropoffMonitor()
        monitor.run_check()
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
