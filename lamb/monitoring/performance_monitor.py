"""
Performance monitoring system for the LAMB framework.

Based on Technical_Specification.md Section 8: Performance Validation and Testing Strategy.
Provides real-time performance tracking, alerting, and optimization recommendations
based on validated performance targets from reconnaissance data.

Performance targets (from Technical_Specification.md):
- Agent throughput: >10 agents/second (LLM mode with batching)
- Memory usage: <1KB per agent
- Decision latency: <0.456s per agent (LLM mode)
- Cache hit rate: >60% (response caching)
- Scaling limit: 10,000+ agents
"""

from typing import Dict, List, Any, Optional, Callable
import time
import threading
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from ..config.simulation_config import PerformanceConfig


class AlertLevel(Enum):
    """Performance alert levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceAlert:
    """Performance alert data"""
    level: AlertLevel
    metric: str
    current_value: float
    threshold: float
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot"""
    timestamp: float
    step: int
    agent_count: int
    agent_throughput: float
    memory_usage_mb: float
    decision_latency_ms: float
    cache_hit_rate: float
    error_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'step': self.step,
            'agent_count': self.agent_count,
            'agent_throughput': self.agent_throughput,
            'memory_usage_mb': self.memory_usage_mb,
            'decision_latency_ms': self.decision_latency_ms,
            'cache_hit_rate': self.cache_hit_rate,
            'error_rate': self.error_rate
        }


class PerformanceMonitor:
    """
    Real-time performance monitoring and alerting system.
    
    Features:
    - Continuous performance tracking
    - Threshold-based alerting
    - Performance trend analysis
    - Optimization recommendations
    - Thread-safe operation
    - Configurable monitoring intervals
    """
    
    def __init__(self, config: PerformanceConfig):
        """
        Initialize performance monitor.
        
        Args:
            config: Performance monitoring configuration
        """
        self.config = config
        
        # Performance history
        self.snapshots: deque[PerformanceSnapshot] = deque(maxlen=1000)
        self.alerts: List[PerformanceAlert] = []
        
        # Current metrics
        self.current_metrics = {
            'agent_throughput': 0.0,
            'memory_usage_mb': 0.0,
            'decision_latency_ms': 0.0,
            'cache_hit_rate': 0.0,
            'error_rate': 0.0
        }
        
        # Monitoring state
        self.is_monitoring = False
        self.last_collection_time = 0.0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # Performance targets (from Technical_Specification.md)
        self.targets = {
            'min_agent_throughput': config.target_agent_throughput,
            'max_memory_per_agent': config.target_memory_per_agent / (1024 * 1024),  # Convert to MB
            'max_decision_time': config.max_decision_time * 1000,  # Convert to ms
            'min_cache_hit_rate': 0.6,  # 60% target
            'max_error_rate': 0.05  # 5% maximum
        }
    
    def start_monitoring(self) -> None:
        """Start performance monitoring"""
        with self._lock:
            self.is_monitoring = True
            self.last_collection_time = time.time()
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring"""
        with self._lock:
            self.is_monitoring = False
    
    def record_step(self, step_data: Dict[str, Any]) -> None:
        """
        Record performance data for a simulation step.
        
        Args:
            step_data: Step performance data
        """
        if not self.is_monitoring:
            return
        
        current_time = time.time()
        
        # Check if it's time to collect metrics
        if current_time - self.last_collection_time < self.config.metrics_collection_interval:
            return
        
        with self._lock:
            # Extract metrics from step data
            agent_count = step_data.get('num_agents', 0)
            step_time = step_data.get('step_time', 0.0)
            
            # Calculate throughput
            agent_throughput = agent_count / step_time if step_time > 0 else 0.0
            
            # Extract engine metrics
            engine_metrics = step_data.get('engine_metrics', {})
            decision_latency_ms = engine_metrics.get('avg_decision_time', 0.0) * 1000
            cache_hit_rate = engine_metrics.get('cache_hit_rate', 0.0)
            error_rate = 1.0 - engine_metrics.get('success_rate', 1.0)
            
            # Extract environment metrics
            env_metrics = step_data.get('environment_metrics', {})
            
            # Estimate memory usage (simplified)
            memory_usage_mb = self._estimate_memory_usage(agent_count, env_metrics)
            
            # Create performance snapshot
            snapshot = PerformanceSnapshot(
                timestamp=current_time,
                step=step_data.get('step', 0),
                agent_count=agent_count,
                agent_throughput=agent_throughput,
                memory_usage_mb=memory_usage_mb,
                decision_latency_ms=decision_latency_ms,
                cache_hit_rate=cache_hit_rate,
                error_rate=error_rate
            )
            
            # Store snapshot
            self.snapshots.append(snapshot)
            
            # Update current metrics
            self.current_metrics.update({
                'agent_throughput': agent_throughput,
                'memory_usage_mb': memory_usage_mb,
                'decision_latency_ms': decision_latency_ms,
                'cache_hit_rate': cache_hit_rate,
                'error_rate': error_rate
            })
            
            # Check for alerts
            if self.config.performance_alerts:
                self._check_alerts(snapshot)
            
            self.last_collection_time = current_time
    
    def _estimate_memory_usage(self, agent_count: int, env_metrics: Dict[str, Any]) -> float:
        """Estimate total memory usage in MB"""
        # Base memory per agent (from Technical_Specification.md)
        base_memory_per_agent = self.config.target_memory_per_agent / 1024  # Convert to KB
        
        # Agent memory
        agent_memory_kb = agent_count * base_memory_per_agent
        
        # Environment memory (rough estimate)
        env_memory_kb = agent_count * 0.1  # 100 bytes per agent for environment overhead
        
        # Convert to MB
        total_memory_mb = (agent_memory_kb + env_memory_kb) / 1024
        
        return total_memory_mb
    
    def _check_alerts(self, snapshot: PerformanceSnapshot) -> None:
        """Check performance thresholds and generate alerts"""
        alerts = []
        
        # Agent throughput alert
        if snapshot.agent_throughput < self.targets['min_agent_throughput']:
            alerts.append(PerformanceAlert(
                level=AlertLevel.WARNING,
                metric='agent_throughput',
                current_value=snapshot.agent_throughput,
                threshold=self.targets['min_agent_throughput'],
                message=f"Agent throughput ({snapshot.agent_throughput:.2f} agents/s) below target ({self.targets['min_agent_throughput']:.2f} agents/s)"
            ))
        
        # Memory usage alert
        memory_per_agent_mb = snapshot.memory_usage_mb / max(1, snapshot.agent_count)
        if memory_per_agent_mb > self.targets['max_memory_per_agent']:
            alerts.append(PerformanceAlert(
                level=AlertLevel.WARNING,
                metric='memory_usage',
                current_value=memory_per_agent_mb,
                threshold=self.targets['max_memory_per_agent'],
                message=f"Memory per agent ({memory_per_agent_mb:.3f} MB) exceeds target ({self.targets['max_memory_per_agent']:.3f} MB)"
            ))
        
        # Decision latency alert
        if snapshot.decision_latency_ms > self.targets['max_decision_time']:
            level = AlertLevel.CRITICAL if snapshot.decision_latency_ms > self.targets['max_decision_time'] * 2 else AlertLevel.WARNING
            alerts.append(PerformanceAlert(
                level=level,
                metric='decision_latency',
                current_value=snapshot.decision_latency_ms,
                threshold=self.targets['max_decision_time'],
                message=f"Decision latency ({snapshot.decision_latency_ms:.1f} ms) exceeds target ({self.targets['max_decision_time']:.1f} ms)"
            ))
        
        # Cache hit rate alert
        if snapshot.cache_hit_rate < self.targets['min_cache_hit_rate']:
            alerts.append(PerformanceAlert(
                level=AlertLevel.INFO,
                metric='cache_hit_rate',
                current_value=snapshot.cache_hit_rate,
                threshold=self.targets['min_cache_hit_rate'],
                message=f"Cache hit rate ({snapshot.cache_hit_rate:.2f}) below target ({self.targets['min_cache_hit_rate']:.2f})"
            ))
        
        # Error rate alert
        if snapshot.error_rate > self.targets['max_error_rate']:
            level = AlertLevel.CRITICAL if snapshot.error_rate > 0.1 else AlertLevel.WARNING
            alerts.append(PerformanceAlert(
                level=level,
                metric='error_rate',
                current_value=snapshot.error_rate,
                threshold=self.targets['max_error_rate'],
                message=f"Error rate ({snapshot.error_rate:.3f}) exceeds target ({self.targets['max_error_rate']:.3f})"
            ))
        
        # Store and notify alerts
        for alert in alerts:
            self.alerts.append(alert)
            self._notify_alert(alert)
    
    def _notify_alert(self, alert: PerformanceAlert) -> None:
        """Notify alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Alert callback error: {e}")
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        with self._lock:
            return self.current_metrics.copy()
    
    def get_recent_snapshots(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent performance snapshots"""
        with self._lock:
            recent = list(self.snapshots)[-count:]
            return [snapshot.to_dict() for snapshot in recent]
    
    def get_alerts(self, level: Optional[AlertLevel] = None) -> List[PerformanceAlert]:
        """Get performance alerts, optionally filtered by level"""
        with self._lock:
            if level is None:
                return self.alerts.copy()
            else:
                return [alert for alert in self.alerts if alert.level == level]
    
    def clear_alerts(self) -> None:
        """Clear all alerts"""
        with self._lock:
            self.alerts.clear()
    
    def get_performance_trends(self, window_size: int = 50) -> Dict[str, Any]:
        """Analyze performance trends over recent snapshots"""
        with self._lock:
            if len(self.snapshots) < 2:
                return {'status': 'insufficient_data'}
            
            recent = list(self.snapshots)[-window_size:]
            
            if len(recent) < 2:
                return {'status': 'insufficient_data'}
            
            # Calculate trends
            trends = {}
            
            for metric in ['agent_throughput', 'memory_usage_mb', 'decision_latency_ms', 'cache_hit_rate', 'error_rate']:
                values = [getattr(snapshot, metric) for snapshot in recent]
                
                if len(values) >= 2:
                    # Simple linear trend (positive = improving, negative = degrading)
                    trend = (values[-1] - values[0]) / len(values)
                    
                    # For some metrics, positive trend is bad
                    if metric in ['memory_usage_mb', 'decision_latency_ms', 'error_rate']:
                        trend = -trend
                    
                    trends[metric] = {
                        'trend': trend,
                        'current': values[-1],
                        'average': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values)
                    }
            
            return {
                'status': 'success',
                'window_size': len(recent),
                'trends': trends
            }
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current performance"""
        recommendations = []
        
        with self._lock:
            if not self.snapshots:
                return ["Insufficient performance data for recommendations"]
            
            latest = self.snapshots[-1]
            
            # Throughput recommendations
            if latest.agent_throughput < self.targets['min_agent_throughput']:
                recommendations.append("Consider increasing batch size for LLM processing")
                recommendations.append("Enable response caching if not already active")
                recommendations.append("Consider using faster LLM model (e.g., gpt-3.5-turbo instead of gpt-4)")
            
            # Memory recommendations
            memory_per_agent = latest.memory_usage_mb / max(1, latest.agent_count) * 1024  # Convert to KB
            if memory_per_agent > self.targets['max_memory_per_agent'] / 1024:
                recommendations.append("Reduce agent history buffer size")
                recommendations.append("Consider using more efficient spatial indexing")
                recommendations.append("Optimize agent metadata storage")
            
            # Latency recommendations
            if latest.decision_latency_ms > self.targets['max_decision_time']:
                recommendations.append("Reduce LLM max_tokens parameter")
                recommendations.append("Increase batch processing size")
                recommendations.append("Consider implementing request timeout optimization")
            
            # Cache recommendations
            if latest.cache_hit_rate < self.targets['min_cache_hit_rate']:
                recommendations.append("Increase cache size")
                recommendations.append("Optimize cache key generation for better hit rates")
                recommendations.append("Consider longer cache TTL for stable scenarios")
            
            # Error rate recommendations
            if latest.error_rate > self.targets['max_error_rate']:
                recommendations.append("Check LLM API connectivity and rate limits")
                recommendations.append("Implement more robust error handling")
                recommendations.append("Consider reducing concurrent request load")
        
        return recommendations if recommendations else ["Performance is within acceptable ranges"]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self._lock:
            if not self.snapshots:
                return {'status': 'no_data'}
            
            latest = self.snapshots[-1]
            
            # Calculate overall performance status
            status_scores = []
            
            # Throughput score
            throughput_score = min(1.0, latest.agent_throughput / self.targets['min_agent_throughput'])
            status_scores.append(throughput_score)
            
            # Memory score (inverted - lower is better)
            memory_per_agent = latest.memory_usage_mb / max(1, latest.agent_count) * 1024
            memory_score = max(0.0, 1.0 - (memory_per_agent / (self.targets['max_memory_per_agent'] / 1024)))
            status_scores.append(memory_score)
            
            # Latency score (inverted - lower is better)
            latency_score = max(0.0, 1.0 - (latest.decision_latency_ms / self.targets['max_decision_time']))
            status_scores.append(latency_score)
            
            # Cache score
            cache_score = latest.cache_hit_rate / self.targets['min_cache_hit_rate']
            status_scores.append(cache_score)
            
            # Error score (inverted - lower is better)
            error_score = max(0.0, 1.0 - (latest.error_rate / self.targets['max_error_rate']))
            status_scores.append(error_score)
            
            # Overall score
            overall_score = sum(status_scores) / len(status_scores)
            
            # Determine status
            if overall_score >= 0.9:
                status = "excellent"
            elif overall_score >= 0.7:
                status = "good"
            elif overall_score >= 0.5:
                status = "fair"
            else:
                status = "poor"
            
            return {
                'status': status,
                'overall_score': overall_score,
                'current_metrics': latest.to_dict(),
                'targets': self.targets,
                'total_snapshots': len(self.snapshots),
                'active_alerts': len([a for a in self.alerts if time.time() - a.timestamp < 300]),  # Last 5 minutes
                'recommendations': self.get_optimization_recommendations()
            }
    
    def reset(self) -> None:
        """Reset all monitoring data"""
        with self._lock:
            self.snapshots.clear()
            self.alerts.clear()
            self.current_metrics = {
                'agent_throughput': 0.0,
                'memory_usage_mb': 0.0,
                'decision_latency_ms': 0.0,
                'cache_hit_rate': 0.0,
                'error_rate': 0.0
            }
    
    def __repr__(self) -> str:
        return (f"PerformanceMonitor(snapshots={len(self.snapshots)}, "
                f"alerts={len(self.alerts)}, monitoring={self.is_monitoring})")
