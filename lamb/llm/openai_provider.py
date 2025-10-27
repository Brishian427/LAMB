"""
OpenAI API provider implementation for LLM integration.

Based on Technical_Specification.md Section 1.3 and reconnaissance findings.
Provides robust OpenAI API integration with error handling, retry logic,
and performance monitoring.

Performance characteristics:
- Single request: <0.456s average (validated from reconnaissance)
- Batch request: <5s for 10-25 agents
- Timeout handling: 5s per request with exponential backoff
- Error recovery: Circuit breaker pattern integration
"""

from typing import Dict, List, Any, Optional
import time
import asyncio
import json
import os
from dataclasses import dataclass

# OpenAI import with fallback for development
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None
    OpenAI = None

from ..core.types import LAMBError, EngineTimeoutError


@dataclass
class APIMetrics:
    """Metrics for API performance tracking"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    avg_response_time: float = 0.0
    timeout_count: int = 0
    rate_limit_count: int = 0


class OpenAIProvider:
    """
    OpenAI API provider with comprehensive error handling and monitoring.
    
    Features:
    - Multiple model support (GPT-3.5, GPT-4, etc.)
    - Automatic retry with exponential backoff
    - Rate limiting and timeout handling
    - Token usage and cost tracking
    - Performance metrics collection
    - Graceful degradation for API issues
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 150,
        temperature: float = 0.7,
        timeout_seconds: float = 5.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            model: Model to use for completions
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature (0.0 to 1.0)
            timeout_seconds: Request timeout
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries (exponential backoff)
        """
        if not OPENAI_AVAILABLE:
            raise LAMBError("OpenAI library not available. Install with: pip install openai")
        
        # Get API key
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise LAMBError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Configuration
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Performance metrics
        self.metrics = APIMetrics()
        
        # Model pricing (tokens per USD, approximate)
        self.model_pricing = {
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},  # per 1K tokens
            'gpt-3.5-turbo-16k': {'input': 0.003, 'output': 0.004},
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-4-32k': {'input': 0.06, 'output': 0.12},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
            'gpt-4o': {'input': 0.005, 'output': 0.015},
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006}
        }
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate single response from OpenAI API.
        
        Performance target: <0.456s average
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            Generated response text
            
        Raises:
            EngineTimeoutError: If request times out or fails after retries
        """
        start_time = time.perf_counter()
        
        for attempt in range(self.max_retries + 1):
            try:
                # Make API call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.timeout_seconds
                )
                
                # Extract response text
                response_text = response.choices[0].message.content
                
                # Update metrics
                response_time = time.perf_counter() - start_time
                self._update_metrics(response, response_time, True)
                
                return response_text
                
            except openai.RateLimitError as e:
                self.metrics.rate_limit_count += 1
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    self._update_metrics(None, time.perf_counter() - start_time, False)
                    raise EngineTimeoutError(f"Rate limit exceeded: {e}")
            
            except openai.APITimeoutError as e:
                self.metrics.timeout_count += 1
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    self._update_metrics(None, time.perf_counter() - start_time, False)
                    raise EngineTimeoutError(f"API timeout: {e}")
            
            except openai.APIError as e:
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    self._update_metrics(None, time.perf_counter() - start_time, False)
                    raise EngineTimeoutError(f"API error: {e}")
            
            except Exception as e:
                self._update_metrics(None, time.perf_counter() - start_time, False)
                raise EngineTimeoutError(f"Unexpected error: {e}")
        
        # Should not reach here
        self._update_metrics(None, time.perf_counter() - start_time, False)
        raise EngineTimeoutError("Max retries exceeded")
    
    def generate_batch_response(self, batch_prompt: str, expected_count: int) -> str:
        """
        Generate batch response for multiple agents.
        
        Performance target: <5s for 10-25 agents
        
        Args:
            batch_prompt: Batch prompt containing multiple agent contexts
            expected_count: Expected number of responses
            
        Returns:
            Generated batch response text
        """
        start_time = time.perf_counter()
        
        # Adjust max_tokens for batch processing
        batch_max_tokens = min(self.max_tokens * expected_count, 4000)
        
        for attempt in range(self.max_retries + 1):
            try:
                # Make API call with increased token limit
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": batch_prompt}],
                    max_tokens=batch_max_tokens,
                    temperature=self.temperature,
                    timeout=self.timeout_seconds * 2  # Double timeout for batch
                )
                
                # Extract response text
                response_text = response.choices[0].message.content
                
                # Update metrics
                response_time = time.perf_counter() - start_time
                self._update_metrics(response, response_time, True)
                
                return response_text
                
            except openai.RateLimitError as e:
                self.metrics.rate_limit_count += 1
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    self._update_metrics(None, time.perf_counter() - start_time, False)
                    raise EngineTimeoutError(f"Batch rate limit exceeded: {e}")
            
            except openai.APITimeoutError as e:
                self.metrics.timeout_count += 1
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    self._update_metrics(None, time.perf_counter() - start_time, False)
                    raise EngineTimeoutError(f"Batch API timeout: {e}")
            
            except Exception as e:
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    self._update_metrics(None, time.perf_counter() - start_time, False)
                    raise EngineTimeoutError(f"Batch API error: {e}")
        
        # Should not reach here
        self._update_metrics(None, time.perf_counter() - start_time, False)
        raise EngineTimeoutError("Batch max retries exceeded")
    
    async def generate_response_async(self, prompt: str) -> str:
        """
        Generate response asynchronously.
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            Generated response text
        """
        # For now, run synchronous version in thread pool
        # In production, would use async OpenAI client
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_response, prompt)
    
    def _update_metrics(self, response: Any, response_time: float, success: bool) -> None:
        """Update performance metrics"""
        self.metrics.total_requests += 1
        
        if success:
            self.metrics.successful_requests += 1
            
            # Update token usage and cost
            if response and hasattr(response, 'usage'):
                usage = response.usage
                self.metrics.total_tokens_used += usage.total_tokens
                
                # Calculate cost
                if self.model in self.model_pricing:
                    pricing = self.model_pricing[self.model]
                    input_cost = (usage.prompt_tokens / 1000) * pricing['input']
                    output_cost = (usage.completion_tokens / 1000) * pricing['output']
                    self.metrics.total_cost_usd += input_cost + output_cost
        else:
            self.metrics.failed_requests += 1
        
        # Update average response time
        total = self.metrics.total_requests
        current_avg = self.metrics.avg_response_time
        self.metrics.avg_response_time = (
            (current_avg * (total - 1) + response_time) / total
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive API metrics"""
        success_rate = 0.0
        if self.metrics.total_requests > 0:
            success_rate = self.metrics.successful_requests / self.metrics.total_requests
        
        return {
            'model': self.model,
            'total_requests': self.metrics.total_requests,
            'successful_requests': self.metrics.successful_requests,
            'failed_requests': self.metrics.failed_requests,
            'success_rate': success_rate,
            'total_tokens_used': self.metrics.total_tokens_used,
            'total_cost_usd': round(self.metrics.total_cost_usd, 4),
            'avg_response_time': self.metrics.avg_response_time,
            'timeout_count': self.metrics.timeout_count,
            'rate_limit_count': self.metrics.rate_limit_count,
            'avg_tokens_per_request': (
                self.metrics.total_tokens_used / max(1, self.metrics.successful_requests)
            ),
            'cost_per_request': (
                self.metrics.total_cost_usd / max(1, self.metrics.successful_requests)
            )
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        self.metrics = APIMetrics()
    
    def validate_connection(self) -> bool:
        """
        Validate OpenAI API connection.
        
        Returns:
            True if connection is working, False otherwise
        """
        try:
            # Make a simple test request
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1,
                temperature=0.0,
                timeout=5.0
            )
            return True
        except Exception:
            return False
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List of model names
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models.data if 'gpt' in model.id]
        except Exception:
            # Return common models as fallback
            return ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini']
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate cost for given token usage.
        
        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        if self.model not in self.model_pricing:
            return 0.0
        
        pricing = self.model_pricing[self.model]
        input_cost = (prompt_tokens / 1000) * pricing['input']
        output_cost = (completion_tokens / 1000) * pricing['output']
        
        return input_cost + output_cost
    
    def __repr__(self) -> str:
        return (f"OpenAIProvider(model={self.model}, "
                f"requests={self.metrics.total_requests}, "
                f"success_rate={self.metrics.successful_requests/max(1, self.metrics.total_requests):.2f})")
