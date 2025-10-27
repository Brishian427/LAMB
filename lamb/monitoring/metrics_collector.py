"""
Metrics collection system for the LAMB framework.

Based on Technical_Specification.md Section 8: Performance Validation and Testing Strategy.
Provides comprehensive metrics collection, aggregation, and analysis for all
framework components.

Collected metrics:
- Agent performance (decision times, memory usage, action success rates)
- Environment performance (spatial queries, state updates, conflict resolution)
- Engine performance (LLM calls, cache hits, batch processing)
- System performance (throughput, latency, resource utilization)
"""

from typing import Dict, List, Any, Optional, Union
import time
import statistics
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"        # Monotonically increasing
    GAUGE = "gauge"           # Current value
    HISTOGRAM = "histogram"   # Distribution of values
    TIMER = "timer"          # Duration measurements


@dataclass
class MetricValue:
    """Individual metric value with metadata"""
    value: Union[int, float]
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Statistical summary of metric values"""
    count: int
    sum: float
    min: float
    max: float
    mean: float
    median: float
    p95: float
    p99: float
    std_dev: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'count': self.count,
            'sum': self.sum,
            'min': self.min,
            'max': self.max,
            'mean': self.mean,
            'median': self.median,
            'p95': self.p95,
            'p99': self.p99,
            'std_dev': self.std_dev
        }


class MetricsCollector:
    """
    Comprehensive metrics collection and analysis system.
    
    Features:
    - Multiple metric types (counter, gauge, histogram, timer)
    - Automatic statistical analysis
    - Time-based aggregation
    - Tag-based filtering and grouping
    - Memory-efficient storage with configurable retention
    - Thread-safe operation
    """
    
    def __init__(self, max_values_per_metric: int = 10000, retention_hours: float = 24.0):
        """
        Initialize metrics collector.
        
        Args:
            max_values_per_metric: Maximum values to store per metric
            retention_hours: Hours to retain metric data
        """
        self.max_values_per_metric = max_values_per_metric
        self.retention_seconds = retention_hours * 3600
        
        # Metric storage
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.metric_values: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_values_per_metric))
        
        # Metric metadata
        self.metric_types: Dict[str, MetricType] = {}
        self.metric_descriptions: Dict[str, str] = {}
        
        # Performance tracking
        self.collection_start_time = time.time()
        self.total_metrics_collected = 0
    
    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str = "",
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Register a new metric.
        
        Args:
            name: Metric name
            metric_type: Type of metric
            description: Human-readable description
            tags: Default tags for this metric
        """
        self.metric_types[name] = metric_type
        self.metric_descriptions[name] = description
        
        if name not in self.metrics:
            self.metrics[name] = {
                'type': metric_type.value,
                'description': description,
                'tags': tags or {},
                'created_at': time.time(),
                'last_updated': time.time(),
                'total_values': 0
            }
    
    def record_counter(self, name: str, value: Union[int, float] = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record counter metric (monotonically increasing).
        
        Args:
            name: Metric name
            value: Value to add to counter
            tags: Additional tags
        """
        self._ensure_metric_registered(name, MetricType.COUNTER)
        
        # Get current counter value
        current_value = self.metrics[name].get('current_value', 0)
        new_value = current_value + value
        
        self._record_value(name, new_value, tags)
        self.metrics[name]['current_value'] = new_value
    
    def record_gauge(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record gauge metric (current value).
        
        Args:
            name: Metric name
            value: Current value
            tags: Additional tags
        """
        self._ensure_metric_registered(name, MetricType.GAUGE)
        
        self._record_value(name, value, tags)
        self.metrics[name]['current_value'] = value
    
    def record_histogram(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record histogram metric (distribution of values).
        
        Args:
            name: Metric name
            value: Value to add to distribution
            tags: Additional tags
        """
        self._ensure_metric_registered(name, MetricType.HISTOGRAM)
        self._record_value(name, value, tags)
    
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record timer metric (duration measurement).
        
        Args:
            name: Metric name
            duration: Duration in seconds
            tags: Additional tags
        """
        self._ensure_metric_registered(name, MetricType.TIMER)
        self._record_value(name, duration * 1000, tags)  # Convert to milliseconds
    
    def start_timer(self, name: str, tags: Optional[Dict[str, str]] = None) -> 'TimerContext':
        """
        Start a timer context for measuring duration.
        
        Args:
            name: Metric name
            tags: Additional tags
            
        Returns:
            Timer context manager
        """
        return TimerContext(self, name, tags)
    
    def _ensure_metric_registered(self, name: str, metric_type: MetricType) -> None:
        """Ensure metric is registered with correct type"""
        if name not in self.metric_types:
            self.register_metric(name, metric_type)
        elif self.metric_types[name] != metric_type:
            raise ValueError(f"Metric {name} already registered with different type")
    
    def _record_value(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value"""
        metric_value = MetricValue(value=value, tags=tags or {})
        
        self.metric_values[name].append(metric_value)
        self.metrics[name]['last_updated'] = time.time()
        self.metrics[name]['total_values'] += 1
        self.total_metrics_collected += 1
        
        # Clean up old values if needed
        self._cleanup_old_values(name)
    
    def _cleanup_old_values(self, name: str) -> None:
        """Remove old metric values based on retention policy"""
        if name not in self.metric_values:
            return
        
        current_time = time.time()
        values = self.metric_values[name]
        
        # Remove values older than retention period
        while values and current_time - values[0].timestamp > self.retention_seconds:
            values.popleft()
    
    def get_metric_summary(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[MetricSummary]:
        """
        Get statistical summary of metric values.
        
        Args:
            name: Metric name
            tags: Filter by tags (if provided)
            
        Returns:
            Metric summary or None if no data
        """
        if name not in self.metric_values:
            return None
        
        values = self.metric_values[name]
        
        # Filter by tags if provided
        if tags:
            filtered_values = []
            for metric_value in values:
                if all(metric_value.tags.get(k) == v for k, v in tags.items()):
                    filtered_values.append(metric_value.value)
        else:
            filtered_values = [mv.value for mv in values]
        
        if not filtered_values:
            return None
        
        # Calculate statistics
        count = len(filtered_values)
        total = sum(filtered_values)
        min_val = min(filtered_values)
        max_val = max(filtered_values)
        mean_val = total / count
        
        # Sort for percentiles
        sorted_values = sorted(filtered_values)
        median_val = statistics.median(sorted_values)
        
        # Calculate percentiles
        p95_idx = int(0.95 * count)
        p99_idx = int(0.99 * count)
        p95_val = sorted_values[min(p95_idx, count - 1)]
        p99_val = sorted_values[min(p99_idx, count - 1)]
        
        # Standard deviation
        std_dev = statistics.stdev(filtered_values) if count > 1 else 0.0
        
        return MetricSummary(
            count=count,
            sum=total,
            min=min_val,
            max=max_val,
            mean=mean_val,
            median=median_val,
            p95=p95_val,
            p99=p99_val,
            std_dev=std_dev
        )
    
    def get_metric_values(
        self,
        name: str,
        limit: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> List[MetricValue]:
        """
        Get raw metric values.
        
        Args:
            name: Metric name
            limit: Maximum number of values to return
            tags: Filter by tags
            
        Returns:
            List of metric values
        """
        if name not in self.metric_values:
            return []
        
        values = list(self.metric_values[name])
        
        # Filter by tags if provided
        if tags:
            values = [
                mv for mv in values
                if all(mv.tags.get(k) == v for k, v in tags.items())
            ]
        
        # Apply limit
        if limit:
            values = values[-limit:]
        
        return values
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics with summaries"""
        result = {}
        
        for name in self.metrics:
            metric_info = self.metrics[name].copy()
            summary = self.get_metric_summary(name)
            
            if summary:
                metric_info['summary'] = summary.to_dict()
            
            result[name] = metric_info
        
        return result
    
    def get_metrics_by_type(self, metric_type: MetricType) -> Dict[str, Any]:
        """Get all metrics of specific type"""
        result = {}
        
        for name, m_type in self.metric_types.items():
            if m_type == metric_type:
                result[name] = self.metrics.get(name, {})
                summary = self.get_metric_summary(name)
                if summary:
                    result[name]['summary'] = summary.to_dict()
        
        return result
    
    def get_metrics_with_tags(self, tags: Dict[str, str]) -> Dict[str, Any]:
        """Get metrics that have specific tags"""
        result = {}
        
        for name in self.metrics:
            metric_tags = self.metrics[name].get('tags', {})
            if all(metric_tags.get(k) == v for k, v in tags.items()):
                result[name] = self.metrics[name].copy()
                summary = self.get_metric_summary(name, tags)
                if summary:
                    result[name]['summary'] = summary.to_dict()
        
        return result
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics about the collector itself"""
        current_time = time.time()
        uptime = current_time - self.collection_start_time
        
        # Calculate memory usage estimate
        total_values = sum(len(values) for values in self.metric_values.values())
        estimated_memory_mb = (total_values * 100) / (1024 * 1024)  # ~100 bytes per value
        
        return {
            'uptime_seconds': uptime,
            'total_metrics': len(self.metrics),
            'total_values_collected': self.total_metrics_collected,
            'current_values_stored': total_values,
            'estimated_memory_mb': estimated_memory_mb,
            'collection_rate_per_second': self.total_metrics_collected / max(uptime, 1),
            'retention_seconds': self.retention_seconds,
            'max_values_per_metric': self.max_values_per_metric
        }
    
    def export_metrics(self, format: str = "dict") -> Union[Dict[str, Any], str]:
        """
        Export all metrics in specified format.
        
        Args:
            format: Export format ("dict", "json")
            
        Returns:
            Exported metrics
        """
        data = {
            'metadata': {
                'export_time': time.time(),
                'system_metrics': self.get_system_metrics()
            },
            'metrics': self.get_all_metrics()
        }
        
        if format == "json":
            import json
            return json.dumps(data, indent=2, default=str)
        else:
            return data
    
    def clear_metrics(self, name: Optional[str] = None) -> None:
        """
        Clear metric data.
        
        Args:
            name: Specific metric to clear (if None, clears all)
        """
        if name:
            if name in self.metric_values:
                self.metric_values[name].clear()
            if name in self.metrics:
                self.metrics[name]['total_values'] = 0
                self.metrics[name]['current_value'] = 0
        else:
            self.metric_values.clear()
            for metric in self.metrics.values():
                metric['total_values'] = 0
                metric['current_value'] = 0
            self.total_metrics_collected = 0
    
    def get_metric_names(self) -> List[str]:
        """Get list of all metric names"""
        return list(self.metrics.keys())
    
    def has_metric(self, name: str) -> bool:
        """Check if metric exists"""
        return name in self.metrics
    
    def __repr__(self) -> str:
        return (f"MetricsCollector(metrics={len(self.metrics)}, "
                f"values={sum(len(v) for v in self.metric_values.values())}, "
                f"collected={self.total_metrics_collected})")


class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self) -> 'TimerContext':
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            self.collector.record_timer(self.name, duration, self.tags)
