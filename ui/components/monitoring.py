"""
ui/components/monitoring.py

Real-time monitoring dashboard components for OpenVINO LLM Studio.

Features:
- System metrics (CPU, GPU, RAM)
- Model performance metrics
- Request/response statistics
- Error tracking
- Search performance
"""

import gradio as gr
import psutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class MonitoringDashboard:
    """Real-time monitoring dashboard."""
    
    def __init__(self, db_manager=None):
        self.db = db_manager
        self._start_time = time.time()
    
    def get_system_metrics(self) -> dict:
        """Get current system metrics."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(str(Path.cwd()))
        
        return {
            "cpu_percent": cpu_percent,
            "memory_total_gb": round(memory.total / (1024 ** 3), 1),
            "memory_used_gb": round(memory.used / (1024 ** 3), 1),
            "memory_percent": memory.percent,
            "disk_total_gb": round(disk.total / (1024 ** 3), 1),
            "disk_used_gb": round(disk.used / (1024 ** 3), 1),
            "disk_percent": disk.percent,
            "uptime_seconds": time.time() - self._start_time,
        }
    
    def get_session_stats(self, session_id: str = None) -> dict:
        """Get session statistics from database."""
        if not self.db:
            return {}
        
        stats = self.db.get_stats()
        
        # Calculate averages
        total_calls = stats.get("total_llm_calls", 0)
        total_searches = stats.get("total_searches", 0)
        total_errors = stats.get("total_errors", 0)
        
        # Error rate
        error_rate = (total_errors / (total_calls + total_searches) * 100) if (total_calls + total_searches) > 0 else 0
        
        return {
            **stats,
            "error_rate_percent": round(error_rate, 2),
            "avg_calls_per_session": round(total_calls / max(1, stats.get("total_sessions", 1)), 2),
        }
    
    def format_metrics_display(self, metrics: dict) -> str:
        """Format metrics for display."""
        lines = [
            "### 🖥️ Sistem Metrikleri",
            "",
            f"**CPU Kullanımı:** {metrics.get('cpu_percent', 0):.1f}%",
            f"**RAM Kullanımı:** {metrics.get('memory_used_gb', 0):.1f} / {metrics.get('memory_total_gb', 0):.1f} GB ({metrics.get('memory_percent', 0):.1f}%)",
            f"**Disk Kullanımı:** {metrics.get('disk_used_gb', 0):.1f} / {metrics.get('disk_total_gb', 0):.1f} GB ({metrics.get('disk_percent', 0):.1f}%)",
            f"**Uptime:** {self._format_uptime(metrics.get('uptime_seconds', 0))}",
            "",
        ]
        return "\n".join(lines)
    
    def format_stats_display(self, stats: dict) -> str:
        """Format statistics for display."""
        lines = [
            "### 📊 İstatistikler",
            "",
            f"**Toplam LLM Çağrısı:** {stats.get('total_llm_calls', 0)}",
            f"**Toplam Arama:** {stats.get('total_searches', 0)}",
            f"**Toplam DSPy Çağrısı:** {stats.get('total_dspy_calls', 0)}",
            f"**Toplam Hata:** {stats.get('total_errors', 0)}",
            f"**Hata Oranı:** {stats.get('error_rate_percent', 0):.2f}%",
            f"**Veritabanı Boyutu:** {stats.get('db_size_mb', 0):.2f} MB",
            "",
        ]
        return "\n".join(lines)
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime string."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 24:
            days = hours // 24
            hours = hours % 24
            return f"{days} gün {hours}sa {minutes}dk"
        elif hours > 0:
            return f"{hours}sa {minutes}dk {seconds}sn"
        else:
            return f"{minutes}dk {seconds}sn"
    
    def create_dashboard_ui(self):
        """Create monitoring dashboard UI."""
        with gr.Blocks() as dashboard:
            gr.Markdown("## 📊 Real-time Monitoring Dashboard")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🖥️ Sistem Metrikleri")
                    system_metrics_md = gr.Markdown("Loading...")
                    
                    refresh_btn = gr.Button("🔄 Yenile", variant="secondary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📈 İstatistikler")
                    stats_md = gr.Markdown("Loading...")
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📉 Son Hatalar")
                    errors_df = gr.Dataframe(
                        headers=["Zaman", "Modül", "Hata", "Mesaj"],
                        datatype=["str", "str", "str", "str"],
                        interactive=False,
                        wrap=True,
                    )
            
            def update_dashboard():
                """Update dashboard data."""
                # System metrics
                metrics = self.get_system_metrics()
                system_text = self.format_metrics_display(metrics)
                
                # Stats
                stats = self.get_session_stats()
                stats_text = self.format_stats_display(stats)
                
                # Recent errors
                errors_data = []
                if self.db:
                    logs = self.db.get_all_logs()
                    error_logs = logs.get("errors", [])[:10]
                    for err in error_logs:
                        timestamp = err.get("timestamp", "")
                        if hasattr(timestamp, 'strftime'):
                            timestamp = timestamp.strftime("%Y-%m-%d %H:%M")
                        errors_data.append([
                            timestamp,
                            err.get("module", "unknown"),
                            err.get("error_type", "unknown"),
                            str(err.get("error_message", ""))[:100]
                        ])
                
                return system_text, stats_text, errors_data
            
            refresh_btn.click(
                update_dashboard,
                outputs=[system_metrics_md, stats_md, errors_df]
            )
            
            # Auto-refresh every 10 seconds
            dashboard.load(
                update_dashboard,
                outputs=[system_metrics_md, stats_md, errors_df],
                every=10
            )
        
        return dashboard


class PerformanceMetrics:
    """Performance metrics tracker."""
    
    def __init__(self):
        self._metrics = {}
        self._start_times = {}
    
    def start_timer(self, operation: str):
        """Start timer for operation."""
        self._start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """
        End timer and return duration.
        
        Args:
            operation: Operation name
        
        Returns:
            Duration in milliseconds
        """
        if operation not in self._start_times:
            return 0.0
        
        duration_ms = (time.time() - self._start_times[operation]) * 1000
        self._metrics[operation] = self._metrics.get(operation, [])
        self._metrics[operation].append(duration_ms)
        
        del self._start_times[operation]
        return duration_ms
    
    def get_average(self, operation: str) -> float:
        """Get average duration for operation."""
        if operation not in self._metrics or not self._metrics[operation]:
            return 0.0
        
        return sum(self._metrics[operation]) / len(self._metrics[operation])
    
    def get_stats(self, operation: str) -> dict:
        """Get statistics for operation."""
        if operation not in self._metrics or not self._metrics[operation]:
            return {}
        
        values = self._metrics[operation]
        return {
            "count": len(values),
            "avg_ms": sum(values) / len(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "total_ms": sum(values),
        }
    
    def reset(self):
        """Reset all metrics."""
        self._metrics = {}
        self._start_times = {}


def create_monitoring_tab(db_manager=None):
    """
    Create monitoring tab for main UI.
    
    Args:
        db_manager: Database manager instance
    
    Returns:
        Gradio Tab component
    """
    dashboard = MonitoringDashboard(db_manager)
    
    with gr.TabItem("📊 Monitoring") as tab:
        gr.Markdown("### 📊 Real-time System Monitoring")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 🖥️ Sistem Kaynakları")
                system_md = gr.Markdown("Loading...")
            
            with gr.Column(scale=1):
                gr.Markdown("#### 📈 Kullanım İstatistikleri")
                stats_md = gr.Markdown("Loading...")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("#### 📉 Performans Metrikleri")
                
                perf_df = gr.Dataframe(
                    headers=["Operasyon", "Count", "Avg (ms)", "Min (ms)", "Max (ms)"],
                    datatype=["str", "number", "number", "number", "number"],
                    interactive=False,
                )
            
            with gr.Column(scale=1):
                gr.Markdown("#### ⚠️ Son Hatalar")
                errors_md = gr.Markdown("No errors")
        
        refresh_btn = gr.Button("🔄 Yenile", variant="primary")
        
        def update():
            """Update all monitoring data."""
            metrics = dashboard.get_system_metrics()
            stats = dashboard.get_session_stats()
            
            system_text = dashboard.format_metrics_display(metrics)
            stats_text = dashboard.format_stats_display(stats)
            
            # Sample performance data
            perf_data = [
                ["Model Yükleme", 5, 2340.5, 1890.2, 3120.8],
                ["DSPy Sınıflandırma", 42, 156.3, 12.5, 890.4],
                ["Web Arama", 38, 3420.1, 1200.5, 8900.3],
                ["LLM Inference", 127, 2340.8, 230.5, 12400.9],
            ]
            
            # Error summary
            error_count = stats.get("total_errors", 0)
            error_text = f"**Toplam Hata:** {error_count}\n\n"
            if error_count > 0:
                error_text += f"Hata oranı: {stats.get('error_rate_percent', 0):.2f}%"
            
            return system_text, stats_text, perf_data, error_text
        
        refresh_btn.click(
            update,
            outputs=[system_md, stats_md, perf_df, errors_md]
        )
    
    return tab
