"""
Simulation configuration system using Pydantic for validation.

Based on Technical_Specification.md Section 9: Configuration Schema and API Specification.
Provides comprehensive configuration management for all LAMB framework components
with validation, type checking, and documentation.

Configuration hierarchy:
- SimulationConfig: Top-level configuration
- ParadigmConfig: Paradigm-specific settings (Grid, Physics, Network)
- EngineConfig: Engine-specific settings (LLM, Rule, Hybrid)
- PerformanceConfig: Performance monitoring and optimization settings
"""

from typing import Dict, List, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class ParadigmType(str, Enum):
    """Supported paradigm types"""
    GRID = "grid"
    PHYSICS = "physics"
    NETWORK = "network"


class EngineType(str, Enum):
    """Supported engine types"""
    LLM = "llm"
    RULE = "rule"
    HYBRID = "hybrid"
    MOCK = "mock"


class BoundaryCondition(str, Enum):
    """Boundary condition types"""
    WRAP = "wrap"
    WALL = "wall"
    REFLECT = "reflect"
    ABSORB = "absorb"
    INFINITE = "infinite"


# Paradigm-specific configurations

class GridConfig(BaseModel):
    """Configuration for Grid paradigm"""
    dimensions: tuple[int, int] = Field(
        default=(100, 100),
        description="Grid dimensions (width, height)"
    )
    boundary_condition: BoundaryCondition = Field(
        default=BoundaryCondition.WRAP,
        description="How to handle grid boundaries"
    )
    cell_size: float = Field(
        default=1.0,
        gt=0,
        description="Size of each grid cell"
    )
    max_agents_per_cell: int = Field(
        default=1,
        ge=1,
        description="Maximum agents allowed per cell"
    )
    
    @field_validator('dimensions')
    @classmethod
    def validate_dimensions(cls, v):
        if len(v) != 2 or v[0] <= 0 or v[1] <= 0:
            raise ValueError("Dimensions must be (width, height) with positive values")
        return v


class PhysicsConfig(BaseModel):
    """Configuration for Physics paradigm"""
    world_bounds: tuple[tuple[float, float], tuple[float, float]] = Field(
        default=((-100.0, -100.0), (100.0, 100.0)),
        description="World boundaries ((min_x, min_y), (max_x, max_y))"
    )
    boundary_condition: BoundaryCondition = Field(
        default=BoundaryCondition.REFLECT,
        description="How to handle world boundaries"
    )
    dt: float = Field(
        default=0.1,
        gt=0,
        le=1.0,
        description="Physics simulation time step"
    )
    enable_collisions: bool = Field(
        default=True,
        description="Whether to enable collision detection"
    )
    collision_damping: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Energy loss factor in collisions"
    )
    
    @field_validator('world_bounds')
    @classmethod
    def validate_world_bounds(cls, v):
        (min_x, min_y), (max_x, max_y) = v
        if min_x >= max_x or min_y >= max_y:
            raise ValueError("World bounds must have min < max for both dimensions")
        return v


class NetworkConfig(BaseModel):
    """Configuration for Network paradigm"""
    is_directed: bool = Field(
        default=False,
        description="Whether graph is directed"
    )
    weighted: bool = Field(
        default=False,
        description="Whether edges have weights"
    )
    default_node_capacity: int = Field(
        default=10,
        ge=1,
        description="Default maximum agents per node"
    )
    enable_message_passing: bool = Field(
        default=True,
        description="Whether to enable message passing between agents"
    )


# Engine-specific configurations

class LLMConfig(BaseModel):
    """Configuration for LLM engine"""
    provider: Literal["openai"] = Field(
        default="openai",
        description="LLM provider to use"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key (if None, uses environment variable)"
    )
    model: str = Field(
        default="gpt-3.5-turbo",
        description="Model to use for completions"
    )
    max_tokens: int = Field(
        default=150,
        ge=1,
        le=4000,
        description="Maximum tokens per response"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts"
    )
    
    # Batch processing settings
    batch_size: int = Field(
        default=15,
        ge=1,
        le=25,
        description="Optimal batch size for processing"
    )
    
    # Circuit breaker settings
    circuit_breaker_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Failure rate threshold for circuit breaker"
    )
    circuit_breaker_timeout: float = Field(
        default=30.0,
        gt=0,
        description="Circuit breaker recovery timeout"
    )
    
    # Caching settings
    cache_size: int = Field(
        default=1000,
        ge=0,
        description="Maximum cache entries (0 to disable)"
    )
    cache_ttl_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Cache entry time to live"
    )


class RuleConfig(BaseModel):
    """Configuration for Rule engine (future implementation)"""
    rule_type: str = Field(
        default="default",
        description="Type of rule-based behavior"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Rule-specific parameters"
    )


class HybridConfig(BaseModel):
    """Configuration for Hybrid engine (future implementation)"""
    llm_config: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM engine configuration"
    )
    rule_config: RuleConfig = Field(
        default_factory=RuleConfig,
        description="Rule engine configuration"
    )
    fallback_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for falling back to rules"
    )


# Performance and monitoring configurations

class PerformanceConfig(BaseModel):
    """Configuration for performance monitoring"""
    enable_monitoring: bool = Field(
        default=True,
        description="Whether to enable performance monitoring"
    )
    metrics_collection_interval: float = Field(
        default=1.0,
        gt=0,
        description="Interval for collecting metrics (seconds)"
    )
    memory_tracking: bool = Field(
        default=True,
        description="Whether to track memory usage"
    )
    performance_alerts: bool = Field(
        default=True,
        description="Whether to generate performance alerts"
    )
    
    # Performance targets (from Technical_Specification.md)
    target_agent_throughput: float = Field(
        default=10.0,
        gt=0,
        description="Target agents processed per second"
    )
    target_memory_per_agent: int = Field(
        default=1024,
        gt=0,
        description="Target memory usage per agent (bytes)"
    )
    max_decision_time: float = Field(
        default=0.456,
        gt=0,
        description="Maximum decision time per agent (seconds)"
    )


class SpatialConfig(BaseModel):
    """Configuration for spatial indexing"""
    auto_select: bool = Field(
        default=True,
        description="Whether to automatically select spatial index"
    )
    rebuild_threshold: int = Field(
        default=100,
        ge=1,
        description="Movements before spatial index rebuild"
    )
    rebuild_interval: float = Field(
        default=1.0,
        gt=0,
        description="Maximum time between rebuilds (seconds)"
    )


# Main simulation configuration

class SimulationConfig(BaseModel):
    """
    Complete simulation configuration.
    
    This is the main configuration class that contains all settings
    for a LAMB simulation.
    """
    
    # Basic simulation settings
    name: str = Field(
        default="LAMB Simulation",
        description="Simulation name"
    )
    description: Optional[str] = Field(
        default=None,
        description="Simulation description"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    
    # Paradigm configuration
    paradigm: ParadigmType = Field(
        description="Simulation paradigm to use"
    )
    grid_config: Optional[GridConfig] = Field(
        default=None,
        description="Grid paradigm configuration"
    )
    physics_config: Optional[PhysicsConfig] = Field(
        default=None,
        description="Physics paradigm configuration"
    )
    network_config: Optional[NetworkConfig] = Field(
        default=None,
        description="Network paradigm configuration"
    )
    
    # Engine configuration
    engine_type: EngineType = Field(
        default=EngineType.LLM,
        description="Decision engine to use"
    )
    llm_config: Optional[LLMConfig] = Field(
        default=None,
        description="LLM engine configuration"
    )
    rule_config: Optional[RuleConfig] = Field(
        default=None,
        description="Rule engine configuration"
    )
    hybrid_config: Optional[HybridConfig] = Field(
        default=None,
        description="Hybrid engine configuration"
    )
    
    # Agent configuration
    num_agents: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Number of agents in simulation"
    )
    agent_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific configuration parameters"
    )
    
    # Simulation execution
    max_steps: int = Field(
        default=1000,
        ge=1,
        description="Maximum simulation steps"
    )
    step_interval: float = Field(
        default=0.1,
        gt=0,
        description="Time interval between steps (seconds)"
    )
    
    # Performance and monitoring
    performance_config: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance monitoring configuration"
    )
    spatial_config: SpatialConfig = Field(
        default_factory=SpatialConfig,
        description="Spatial indexing configuration"
    )
    
    # Output and logging
    output_dir: Optional[str] = Field(
        default=None,
        description="Directory for simulation output"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    save_state_interval: int = Field(
        default=100,
        ge=0,
        description="Steps between state saves (0 to disable)"
    )
    
    @model_validator(mode='before')
    @classmethod
    def validate_paradigm_config(cls, values):
        """Validate that paradigm-specific config is provided"""
        if isinstance(values, dict):
            paradigm = values.get('paradigm')
            
            if paradigm == ParadigmType.GRID and not values.get('grid_config'):
                values['grid_config'] = GridConfig()
            elif paradigm == ParadigmType.PHYSICS and not values.get('physics_config'):
                values['physics_config'] = PhysicsConfig()
            elif paradigm == ParadigmType.NETWORK and not values.get('network_config'):
                values['network_config'] = NetworkConfig()
        
        return values
    
    @model_validator(mode='before')
    @classmethod
    def validate_engine_config(cls, values):
        """Validate that engine-specific config is provided"""
        if isinstance(values, dict):
            engine_type = values.get('engine_type')
            
            if engine_type == EngineType.LLM and not values.get('llm_config'):
                values['llm_config'] = LLMConfig()
            elif engine_type == EngineType.RULE and not values.get('rule_config'):
                values['rule_config'] = RuleConfig()
            elif engine_type == EngineType.HYBRID and not values.get('hybrid_config'):
                values['hybrid_config'] = HybridConfig()
        
        return values
    
    @model_validator(mode='after')
    def validate_num_agents(self):
        """Validate number of agents based on paradigm"""
        # Performance recommendations from Technical_Specification.md
        if self.paradigm == ParadigmType.GRID and self.num_agents > 5000:
            raise ValueError("Grid paradigm recommended for ≤5,000 agents")
        elif self.paradigm == ParadigmType.PHYSICS and self.num_agents > 50000:
            raise ValueError("Physics paradigm recommended for ≤50,000 agents")
        
        return self
    
    def get_paradigm_config(self) -> Union[GridConfig, PhysicsConfig, NetworkConfig]:
        """Get the active paradigm configuration"""
        if self.paradigm == ParadigmType.GRID:
            return self.grid_config
        elif self.paradigm == ParadigmType.PHYSICS:
            return self.physics_config
        elif self.paradigm == ParadigmType.NETWORK:
            return self.network_config
        else:
            raise ValueError(f"Unknown paradigm: {self.paradigm}")
    
    def get_engine_config(self) -> Union[LLMConfig, RuleConfig, HybridConfig]:
        """Get the active engine configuration"""
        if self.engine_type == EngineType.LLM:
            return self.llm_config
        elif self.engine_type == EngineType.RULE:
            return self.rule_config
        elif self.engine_type == EngineType.HYBRID:
            return self.hybrid_config
        else:
            raise ValueError(f"Unknown engine type: {self.engine_type}")
    
    model_config = {
        "use_enum_values": True,
        "validate_assignment": True,
        "extra": "forbid",  # Prevent extra fields
        "json_schema_extra": {
            "example": {
                "name": "Boids Flocking Simulation",
                "paradigm": "physics",
                "engine_type": "llm",
                "num_agents": 50,
                "max_steps": 1000,
                "physics_config": {
                    "world_bounds": ((-50, -50), (50, 50)),
                    "dt": 0.1,
                    "enable_collisions": True
                },
                "llm_config": {
                    "model": "gpt-3.5-turbo",
                    "batch_size": 15,
                    "temperature": 0.7
                }
            }
        }
    }
