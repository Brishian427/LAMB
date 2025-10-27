"""
Performance monitoring and metrics collection for the LAMB framework.

This module provides comprehensive monitoring capabilities for all framework
components, including real-time performance tracking, alerting, and
optimization recommendations.

Based on Technical_Specification.md Section 8: Performance Validation and Testing Strategy.

Key components:
- PerformanceMonitor: Real-time performance tracking with alerting
- MetricsCollector: Comprehensive metrics collection and analysis
- Automatic performance optimization recommendations
- Thread-safe operation for concurrent simulations

Performance targets (validated from reconnaissance):
- Agent throughput: >10 agents/second (LLM mode with batching)
- Memory usage: <1KB per agent
- Decision latency: <0.456s per agent (LLM mode)
- Cache hit rate: >60% (response caching)
"""

from .performance_monitor import PerformanceMonitor, PerformanceAlert, PerformanceSnapshot, AlertLevel
from .metrics_collector import MetricsCollector, MetricType, MetricValue, MetricSummary, TimerContext

__all__ = [
    "PerformanceMonitor",
    "PerformanceAlert",
    "PerformanceSnapshot", 
    "AlertLevel",
    "MetricsCollector",
    "MetricType",
    "MetricValue",
    "MetricSummary",
    "TimerContext"
]
