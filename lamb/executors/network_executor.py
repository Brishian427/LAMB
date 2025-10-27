"""
Network executor for event-driven propagation.

This executor handles network-based simulations where agents interact
through graph connections and events propagate through the network.
This is suitable for social networks, disease spread, and information diffusion.
"""

from typing import List, Dict, Any, Optional
import time
from collections import deque

from .base_executor import BaseExecutor
from ..core.base_agent import BaseAgent
from ..core.base_environment import BaseEnvironment
from ..core.base_engine import BaseEngine
from ..core.simulation import SimulationResults
from ..core.types import Observation, Action, AgentID
from ..config.simulation_config import SimulationConfig


class NetworkEvent:
    """Represents an event in the network simulation"""
    
    def __init__(self, event_type: str, source_agent: AgentID, target_agent: AgentID, 
                 data: Dict[str, Any], timestamp: float):
        self.event_type = event_type
        self.source_agent = source_agent
        self.target_agent = target_agent
        self.data = data
        self.timestamp = timestamp
    
    def process(self, environment: BaseEnvironment, agents: List[BaseAgent]):
        """Process this event in the network"""
        # Find target agent
        target_agent = None
        for agent in agents:
            if agent.id == self.target_agent:
                target_agent = agent
                break
        
        if target_agent and hasattr(target_agent, 'handle_event'):
            target_agent.handle_event(self)


class NetworkExecutor(BaseExecutor):
    """
    Executor for network-based simulations with event-driven propagation.
    
    This executor implements event-driven simulation:
    1. Agents generate events based on their decisions
    2. Events propagate through the network
    3. Agents respond to incoming events
    4. Network structure can change dynamically
    
    This is suitable for:
    - Social network models
    - Disease spread (SIR models)
    - Information diffusion
    - Opinion dynamics
    - Network formation
    """
    
    def run(self, 
            environment: BaseEnvironment,
            agents: List[BaseAgent],
            engine: BaseEngine,
            max_steps: int,
            config: Optional[SimulationConfig] = None) -> SimulationResults:
        """
        Execute network-based simulation with event-driven propagation.
        
        Args:
            environment: The network environment
            agents: List of agents in the simulation
            engine: Decision-making engine
            max_steps: Maximum number of steps to run
            config: Optional simulation configuration
            
        Returns:
            SimulationResults containing all simulation data
        """
        start_time = time.time()
        step_metrics = []
        agent_decisions = []
        
        # Initialize event queue
        event_queue = deque()
        current_time = 0.0
        
        # Run simulation steps
        for step in range(max_steps):
            step_start_time = time.time()
            current_decisions = []
            
            # 1. Process pending events
            self._process_events(event_queue, environment, agents)
            
            # 2. Environment updates (network structure, etc.)
            environment.step()
            
            # 3. All agents observe network state
            observations = self._create_observations(agents, environment)
            
            # 4. All agents decide based on network state
            decisions = self._get_agent_decisions(observations, engine)
            current_decisions = decisions
            
            # 5. All agents act and generate events
            for agent, decision in zip(agents, decisions):
                if hasattr(agent, 'act'):
                    agent.act(decision, environment)
                elif hasattr(agent, 'step'):
                    agent.step(environment)
                
                # Generate events from agent actions
                new_events = self._generate_events_from_decision(agent, decision, current_time)
                event_queue.extend(new_events)
            
            # 6. Update network structure if needed
            self._update_network_structure(environment, agents)
            
            # Record step metrics
            step_metrics.append(self._record_step_metrics(step, environment, agents, current_decisions))
            agent_decisions.append({agent.id: decision for agent, decision in zip(agents, current_decisions)})
            
            # Update time
            current_time += 1.0
            
            # Log progress for long simulations
            if step % 50 == 0 and step > 0:
                elapsed = time.time() - start_time
                print(f"Network simulation: Step {step}/{max_steps} completed in {elapsed:.2f}s")
        
        total_time = time.time() - start_time
        
        # Create results
        results = SimulationResults(
            step_count=max_steps,
            total_time=total_time,
            step_metrics=step_metrics,
            final_state=environment.get_state(),
            agent_decisions=agent_decisions
        )
        
        return results
    
    def get_paradigm(self) -> str:
        """Return the paradigm this executor handles"""
        return "network"
    
    def _process_events(self, event_queue: deque, environment: BaseEnvironment, agents: List[BaseAgent]):
        """Process all pending events in the queue"""
        while event_queue:
            event = event_queue.popleft()
            event.process(environment, agents)
    
    def _generate_events_from_decision(self, agent: BaseAgent, decision: Action, timestamp: float) -> List[NetworkEvent]:
        """Generate network events from agent decision"""
        events = []
        
        # Extract event information from decision
        action_type = decision.get("action_type", "")
        parameters = decision.get("parameters", {})
        
        # Generate events based on action type
        if action_type == "send_message":
            target_agents = parameters.get("target_agents", [])
            message = parameters.get("message", "")
            
            for target_id in target_agents:
                event = NetworkEvent(
                    event_type="message",
                    source_agent=agent.id,
                    target_agent=target_id,
                    data={"message": message, "content": parameters.get("content", "")},
                    timestamp=timestamp
                )
                events.append(event)
        
        elif action_type == "influence":
            target_agents = parameters.get("target_agents", [])
            influence_strength = parameters.get("influence_strength", 1.0)
            
            for target_id in target_agents:
                event = NetworkEvent(
                    event_type="influence",
                    source_agent=agent.id,
                    target_agent=target_id,
                    data={"influence_strength": influence_strength, "topic": parameters.get("topic", "")},
                    timestamp=timestamp
                )
                events.append(event)
        
        elif action_type == "create_connection":
            target_agent = parameters.get("target_agent")
            if target_agent:
                event = NetworkEvent(
                    event_type="connection_request",
                    source_agent=agent.id,
                    target_agent=target_agent,
                    data={"weight": parameters.get("weight", 1.0)},
                    timestamp=timestamp
                )
                events.append(event)
        
        return events
    
    def _update_network_structure(self, environment: BaseEnvironment, agents: List[BaseAgent]):
        """Update network structure based on agent actions"""
        if hasattr(environment, 'update_network'):
            environment.update_network()
    
    def _record_step_metrics(self, step: int, environment: BaseEnvironment, 
                           agents: List[BaseAgent], decisions: List[Action]) -> Dict[str, Any]:
        """Record metrics specific to network simulations"""
        base_metrics = super()._record_step_metrics(step, environment, agents, decisions)
        
        # Add network-specific metrics
        network_metrics = {
            "agent_nodes": [getattr(agent, 'node_id', None) for agent in agents],
            "network_size": len(agents),
            "network_density": self._calculate_network_density(environment),
            "decision_summary": self._summarize_decisions(decisions)
        }
        
        base_metrics.update(network_metrics)
        return base_metrics
    
    def _calculate_network_density(self, environment: BaseEnvironment) -> float:
        """Calculate network density if available"""
        if hasattr(environment, 'get_network_density'):
            return environment.get_network_density()
        return 0.0
    
    def _summarize_decisions(self, decisions: List[Action]) -> Dict[str, int]:
        """Summarize decision patterns for this step"""
        decision_counts = {}
        for decision in decisions:
            action_type = decision.action_type
            decision_counts[action_type] = decision_counts.get(action_type, 0) + 1
        return decision_counts
