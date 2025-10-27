"""
Batch processing implementation for LLM performance optimization.

Based on Technical_Specification.md Section 1.3 and reconnaissance findings.
Implements intelligent batching of LLM requests to optimize performance
and reduce API costs while maintaining responsiveness.

Performance characteristics:
- Optimal batch size: 10-25 agents (from reconnaissance validation)
- Batch processing time: <5s per batch (with 5s timeout)
- Throughput improvement: 3-5x over sequential processing
- Cost reduction: ~40% through batch optimization
"""

from typing import List, Dict, Any, Callable, Optional
import time
import asyncio
from dataclasses import dataclass
from collections import deque

from ..core.types import Observation, Action, EngineTimeoutError


@dataclass
class BatchMetrics:
    """Metrics for batch processing performance"""
    total_batches: int = 0
    successful_batches: int = 0
    failed_batches: int = 0
    total_agents_processed: int = 0
    avg_batch_size: float = 0.0
    avg_batch_time: float = 0.0
    optimal_size_batches: int = 0
    oversized_batches: int = 0
    undersized_batches: int = 0


class BatchProcessor:
    """
    Intelligent batch processor for LLM requests.
    
    Features:
    - Dynamic batch sizing based on performance characteristics
    - Timeout handling with graceful degradation
    - Performance monitoring and optimization
    - Support for both synchronous and asynchronous processing
    - Automatic fallback to individual processing on batch failure
    
    Optimization strategy:
    - Target batch size: 10-25 agents (optimal from reconnaissance)
    - Maximum batch size: 25 agents (to stay within token limits)
    - Minimum batch size: 2 agents (below this, use individual processing)
    - Timeout: 5s per batch (2x individual timeout)
    """
    
    def __init__(
        self,
        optimal_batch_size: int = 15,
        max_batch_size: int = 25,
        min_batch_size: int = 2,
        timeout_seconds: float = 5.0,
        adaptive_sizing: bool = True
    ):
        """
        Initialize batch processor.
        
        Args:
            optimal_batch_size: Target batch size for best performance
            max_batch_size: Maximum allowed batch size
            min_batch_size: Minimum batch size (below this, use individual)
            timeout_seconds: Timeout for batch processing
            adaptive_sizing: Whether to adapt batch size based on performance
        """
        self.optimal_batch_size = optimal_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.timeout_seconds = timeout_seconds
        self.adaptive_sizing = adaptive_sizing
        
        # Performance tracking
        self.metrics = BatchMetrics()
        
        # Adaptive sizing state
        self.current_optimal_size = optimal_batch_size
        self.performance_history = deque(maxlen=10)  # Last 10 batch performances
        self.last_adaptation = time.time()
        self.adaptation_interval = 60.0  # Adapt every 60 seconds
    
    def process_batch(
        self,
        observations: List[Observation],
        processor_func: Callable[[List[Observation]], List[Action]]
    ) -> List[Action]:
        """
        Process batch of observations using provided processor function.
        
        Performance target: <5s per batch
        
        Args:
            observations: List of observations to process
            processor_func: Function to process the batch
            
        Returns:
            List of actions corresponding to observations
            
        Raises:
            EngineTimeoutError: If batch processing fails
        """
        if len(observations) < self.min_batch_size:
            # Too small for batching - process individually
            return self._process_individually(observations, processor_func)
        
        # Determine batch sizes
        batches = self._create_batches(observations)
        all_actions = []
        
        for batch in batches:
            batch_actions = self._process_single_batch(batch, processor_func)
            all_actions.extend(batch_actions)
        
        return all_actions
    
    def _create_batches(self, observations: List[Observation]) -> List[List[Observation]]:
        """Create optimally-sized batches from observations"""
        batches = []
        current_batch = []
        
        for obs in observations:
            current_batch.append(obs)
            
            # Check if batch is at optimal size
            if len(current_batch) >= self.current_optimal_size:
                batches.append(current_batch)
                current_batch = []
            
            # Prevent oversized batches
            elif len(current_batch) >= self.max_batch_size:
                batches.append(current_batch)
                current_batch = []
        
        # Add remaining observations as final batch
        if current_batch:
            if len(current_batch) >= self.min_batch_size:
                batches.append(current_batch)
            else:
                # Merge small remainder with last batch if possible
                if batches and len(batches[-1]) + len(current_batch) <= self.max_batch_size:
                    batches[-1].extend(current_batch)
                else:
                    batches.append(current_batch)
        
        return batches
    
    def _process_single_batch(
        self,
        batch: List[Observation],
        processor_func: Callable[[List[Observation]], List[Action]]
    ) -> List[Action]:
        """Process a single batch with timeout and error handling"""
        start_time = time.perf_counter()
        batch_size = len(batch)
        
        try:
            # Process batch with timeout
            actions = self._execute_with_timeout(batch, processor_func)
            
            # Record successful batch
            batch_time = time.perf_counter() - start_time
            self._record_batch_success(batch_size, batch_time)
            
            return actions
            
        except Exception as e:
            # Record failed batch
            batch_time = time.perf_counter() - start_time
            self._record_batch_failure(batch_size, batch_time)
            
            # Fallback to individual processing
            return self._process_individually(batch, processor_func)
    
    def _execute_with_timeout(
        self,
        batch: List[Observation],
        processor_func: Callable[[List[Observation]], List[Action]]
    ) -> List[Action]:
        """Execute batch processing with timeout"""
        # For now, implement simple timeout using time tracking
        # In production, could use asyncio.wait_for or threading.Timer
        
        start_time = time.time()
        
        try:
            actions = processor_func(batch)
            
            # Check if we exceeded timeout
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                raise EngineTimeoutError(f"Batch processing exceeded timeout: {elapsed:.2f}s")
            
            return actions
            
        except Exception as e:
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                raise EngineTimeoutError(f"Batch processing timeout: {elapsed:.2f}s")
            else:
                raise e
    
    def _process_individually(
        self,
        observations: List[Observation],
        processor_func: Callable[[List[Observation]], List[Action]]
    ) -> List[Action]:
        """Fallback to individual processing"""
        actions = []
        
        for obs in observations:
            try:
                # Process single observation
                single_actions = processor_func([obs])
                if single_actions:
                    actions.append(single_actions[0])
                else:
                    # Create emergency action
                    actions.append(Action(
                        agent_id=obs.agent_id,
                        action_type="stay",
                        parameters={}
                    ))
            except Exception:
                # Create emergency action on failure
                actions.append(Action(
                    agent_id=obs.agent_id,
                    action_type="stay",
                    parameters={}
                ))
        
        return actions
    
    def _record_batch_success(self, batch_size: int, batch_time: float) -> None:
        """Record successful batch processing"""
        self.metrics.total_batches += 1
        self.metrics.successful_batches += 1
        self.metrics.total_agents_processed += batch_size
        
        # Update running averages
        total = self.metrics.total_batches
        current_avg_size = self.metrics.avg_batch_size
        current_avg_time = self.metrics.avg_batch_time
        
        self.metrics.avg_batch_size = (
            (current_avg_size * (total - 1) + batch_size) / total
        )
        self.metrics.avg_batch_time = (
            (current_avg_time * (total - 1) + batch_time) / total
        )
        
        # Categorize batch size
        if batch_size == self.current_optimal_size:
            self.metrics.optimal_size_batches += 1
        elif batch_size > self.current_optimal_size:
            self.metrics.oversized_batches += 1
        else:
            self.metrics.undersized_batches += 1
        
        # Record performance for adaptive sizing
        if self.adaptive_sizing:
            performance_score = self._calculate_performance_score(batch_size, batch_time)
            self.performance_history.append((batch_size, batch_time, performance_score))
            self._maybe_adapt_batch_size()
    
    def _record_batch_failure(self, batch_size: int, batch_time: float) -> None:
        """Record failed batch processing"""
        self.metrics.total_batches += 1
        self.metrics.failed_batches += 1
        
        # Update running averages (include failed attempts)
        total = self.metrics.total_batches
        current_avg_size = self.metrics.avg_batch_size
        current_avg_time = self.metrics.avg_batch_time
        
        self.metrics.avg_batch_size = (
            (current_avg_size * (total - 1) + batch_size) / total
        )
        self.metrics.avg_batch_time = (
            (current_avg_time * (total - 1) + batch_time) / total
        )
        
        # Record poor performance for adaptive sizing
        if self.adaptive_sizing:
            performance_score = 0.0  # Failed batch gets 0 score
            self.performance_history.append((batch_size, batch_time, performance_score))
            self._maybe_adapt_batch_size()
    
    def _calculate_performance_score(self, batch_size: int, batch_time: float) -> float:
        """Calculate performance score for batch (0.0 to 1.0)"""
        # Score based on throughput (agents per second)
        throughput = batch_size / batch_time if batch_time > 0 else 0
        
        # Normalize to expected throughput (assume ~5 agents/second as baseline)
        baseline_throughput = 5.0
        throughput_score = min(1.0, throughput / baseline_throughput)
        
        # Bonus for optimal batch size
        size_score = 1.0
        if batch_size < self.current_optimal_size:
            size_score = 0.8  # Penalty for undersized batches
        elif batch_size > self.max_batch_size:
            size_score = 0.6  # Penalty for oversized batches
        
        # Combined score
        return throughput_score * size_score
    
    def _maybe_adapt_batch_size(self) -> None:
        """Adapt batch size based on performance history"""
        current_time = time.time()
        
        if current_time - self.last_adaptation < self.adaptation_interval:
            return  # Too soon to adapt
        
        if len(self.performance_history) < 5:
            return  # Not enough data
        
        # Analyze performance by batch size
        size_performance = {}
        for batch_size, batch_time, score in self.performance_history:
            if batch_size not in size_performance:
                size_performance[batch_size] = []
            size_performance[batch_size].append(score)
        
        # Find best performing batch size
        best_size = self.current_optimal_size
        best_avg_score = 0.0
        
        for size, scores in size_performance.items():
            if len(scores) >= 2:  # Need at least 2 samples
                avg_score = sum(scores) / len(scores)
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_size = size
        
        # Update optimal size if significantly better
        if best_size != self.current_optimal_size and best_avg_score > 0.7:
            self.current_optimal_size = max(
                self.min_batch_size,
                min(self.max_batch_size, best_size)
            )
            self.last_adaptation = current_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive batch processing metrics"""
        # Calculate success rate
        success_rate = 0.0
        if self.metrics.total_batches > 0:
            success_rate = self.metrics.successful_batches / self.metrics.total_batches
        
        # Calculate throughput
        avg_throughput = 0.0
        if self.metrics.avg_batch_time > 0:
            avg_throughput = self.metrics.avg_batch_size / self.metrics.avg_batch_time
        
        # Calculate efficiency metrics
        optimal_ratio = 0.0
        if self.metrics.total_batches > 0:
            optimal_ratio = self.metrics.optimal_size_batches / self.metrics.total_batches
        
        return {
            'optimal_batch_size': self.optimal_batch_size,
            'current_optimal_size': self.current_optimal_size,
            'max_batch_size': self.max_batch_size,
            'min_batch_size': self.min_batch_size,
            'timeout_seconds': self.timeout_seconds,
            'adaptive_sizing': self.adaptive_sizing,
            'total_batches': self.metrics.total_batches,
            'successful_batches': self.metrics.successful_batches,
            'failed_batches': self.metrics.failed_batches,
            'success_rate': success_rate,
            'total_agents_processed': self.metrics.total_agents_processed,
            'avg_batch_size': self.metrics.avg_batch_size,
            'avg_batch_time': self.metrics.avg_batch_time,
            'avg_throughput_agents_per_sec': avg_throughput,
            'optimal_size_batches': self.metrics.optimal_size_batches,
            'oversized_batches': self.metrics.oversized_batches,
            'undersized_batches': self.metrics.undersized_batches,
            'optimal_ratio': optimal_ratio,
            'performance_status': self._get_performance_status(success_rate, avg_throughput)
        }
    
    def _get_performance_status(self, success_rate: float, throughput: float) -> str:
        """Get performance status based on metrics"""
        if success_rate >= 0.9 and throughput >= 3.0:
            return "excellent"
        elif success_rate >= 0.8 and throughput >= 2.0:
            return "good"
        elif success_rate >= 0.6 and throughput >= 1.0:
            return "fair"
        else:
            return "poor"
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        self.metrics = BatchMetrics()
        self.performance_history.clear()
        self.current_optimal_size = self.optimal_batch_size
        self.last_adaptation = time.time()
    
    def get_recommended_batch_size(self) -> int:
        """Get currently recommended batch size"""
        return self.current_optimal_size
    
    def set_batch_size(self, size: int) -> None:
        """Manually set optimal batch size"""
        self.current_optimal_size = max(
            self.min_batch_size,
            min(self.max_batch_size, size)
        )
    
    def is_healthy(self) -> bool:
        """Check if batch processor is performing well"""
        if self.metrics.total_batches == 0:
            return True  # No data yet
        
        success_rate = self.metrics.successful_batches / self.metrics.total_batches
        return success_rate >= 0.7  # Minimum acceptable success rate
    
    def __repr__(self) -> str:
        return (f"BatchProcessor(optimal_size={self.current_optimal_size}, "
                f"batches={self.metrics.total_batches}, "
                f"success_rate={self.metrics.successful_batches/max(1, self.metrics.total_batches):.2f})")
