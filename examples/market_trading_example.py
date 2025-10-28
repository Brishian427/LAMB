#!/usr/bin/env python3
"""
LAMB Framework: Market Trading Model Example

This example implements a market trading model using the LAMB framework's Global paradigm.
The model simulates how agents trade in a market environment with price discovery,
market dynamics, and behavioral strategies.

Key Features:
- Global market environment (no spatial constraints)
- Multiple trading strategies
- Price discovery mechanisms
- Market volatility and trends
- Agent wealth tracking
- Market efficiency analysis

The market model demonstrates:
- How individual trading strategies affect market dynamics
- Price formation through supply and demand
- Market bubbles and crashes
- Behavioral finance principles
- Wealth distribution over time
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random

from lamb.config import SimulationConfig, ParadigmType, EngineType
from lamb.api import ResearchAPI
from lamb.core import Action, Observation, AgentID, Position
from lamb.engines import RuleEngine


class TradingStrategy(Enum):
    """Trading strategies for market agents"""
    FUNDAMENTALIST = "fundamentalist"  # Trade based on fundamental value
    NOISE_OPTIMIST = "noise_optimist"  # Random optimistic trading
    NOISE_PESSIMIST = "noise_pessimist"  # Random pessimistic trading
    MOMENTUM = "momentum"  # Follow price trends
    CONTRARIAN = "contrarian"  # Trade against trends


@dataclass
class MarketParameters:
    """Market model parameters"""
    num_agents: int = 100
    initial_price: float = 100.0
    fundamental_value: float = 100.0
    volatility: float = 0.02  # Price volatility
    transaction_cost: float = 0.001  # Cost per transaction
    max_position: float = 10.0  # Maximum position size
    initial_wealth: float = 1000.0  # Initial wealth per agent
    strategy_distribution: Dict[str, float] = None  # Strategy distribution
    
    def __post_init__(self):
        if self.strategy_distribution is None:
            self.strategy_distribution = {
                "fundamentalist": 0.4,
                "noise_optimist": 0.2,
                "noise_pessimist": 0.2,
                "momentum": 0.1,
                "contrarian": 0.1
            }


class MarketAgent:
    """Agent for market trading model"""
    
    def __init__(self, agent_id: AgentID, strategy: TradingStrategy = None, 
                 initial_wealth: float = 1000.0, **kwargs):
        self.agent_id = agent_id
        self.position = (0, 0)  # Global paradigm - position doesn't matter
        self.strategy = strategy or random.choice(list(TradingStrategy))
        self.wealth = initial_wealth
        self.cash = initial_wealth
        self.shares = 0
        self.wealth_history = [initial_wealth]
        self.trade_history = []
        
        # Strategy-specific parameters
        self.confidence = random.uniform(0.5, 1.0)  # Trading confidence
        self.risk_tolerance = random.uniform(0.1, 0.5)  # Risk tolerance
        self.memory_length = random.randint(5, 20)  # Price memory length
        self.price_memory = []
        
        # Metadata
        self.metadata = {
            "agent_type": "trader",
            "strategy": self.strategy.value,
            "wealth": self.wealth,
            "cash": self.cash,
            "shares": self.shares,
            "confidence": self.confidence,
            "risk_tolerance": self.risk_tolerance
        }
    
    def observe(self, environment) -> Observation:
        """Observe market state"""
        market_data = environment.get_market_data()
        
        return Observation(
            agent_id=self.agent_id,
            position=self.position,
            neighbors=[],  # No neighbors in global paradigm
            paradigm="global",
            data={
                "current_price": market_data["current_price"],
                "price_history": market_data["price_history"][-self.memory_length:],
                "volume": market_data["volume"],
                "my_wealth": self.wealth,
                "my_cash": self.cash,
                "my_shares": self.shares,
                "my_strategy": self.strategy.value,
                "fundamental_value": market_data["fundamental_value"]
            }
        )
    
    def decide(self, observation: Observation, engine) -> Action:
        """Decide trading action based on strategy"""
        current_price = observation.data["current_price"]
        price_history = observation.data["price_history"]
        fundamental_value = observation.data["fundamental_value"]
        
        # Update price memory
        self.price_memory = price_history[-self.memory_length:]
        
        # Calculate position size based on wealth and risk tolerance
        max_position_value = self.wealth * self.risk_tolerance
        max_shares = int(max_position_value / current_price) if current_price > 0 else 0
        
        # Strategy-specific decision making
        if self.strategy == TradingStrategy.FUNDAMENTALIST:
            return self._fundamentalist_decision(current_price, fundamental_value, max_shares)
        elif self.strategy == TradingStrategy.NOISE_OPTIMIST:
            return self._noise_optimist_decision(current_price, max_shares)
        elif self.strategy == TradingStrategy.NOISE_PESSIMIST:
            return self._noise_pessimist_decision(current_price, max_shares)
        elif self.strategy == TradingStrategy.MOMENTUM:
            return self._momentum_decision(current_price, price_history, max_shares)
        elif self.strategy == TradingStrategy.CONTRARIAN:
            return self._contrarian_decision(current_price, price_history, max_shares)
        else:
            return Action(agent_id=self.agent_id, action_type="hold")
    
    def _fundamentalist_decision(self, current_price: float, fundamental_value: float, 
                                max_shares: int) -> Action:
        """Fundamentalist trading strategy"""
        price_diff = (fundamental_value - current_price) / current_price
        
        if price_diff > 0.05:  # Price significantly below fundamental value
            shares_to_buy = min(max_shares, int(self.cash / current_price * 0.1))
            if shares_to_buy > 0:
                return Action(
                    agent_id=self.agent_id,
                    action_type="buy",
                    parameters={"shares": shares_to_buy, "price": current_price}
                )
        elif price_diff < -0.05:  # Price significantly above fundamental value
            shares_to_sell = min(self.shares, int(self.shares * 0.1))
            if shares_to_sell > 0:
                return Action(
                    agent_id=self.agent_id,
                    action_type="sell",
                    parameters={"shares": shares_to_sell, "price": current_price}
                )
        
        return Action(agent_id=self.agent_id, action_type="hold")
    
    def _noise_optimist_decision(self, current_price: float, max_shares: int) -> Action:
        """Noise optimist trading strategy"""
        if random.random() < 0.1:  # 10% chance to trade
            if random.random() < 0.7:  # 70% chance to buy
                shares_to_buy = random.randint(1, max_shares)
                if shares_to_buy > 0 and self.cash >= shares_to_buy * current_price:
                    return Action(
                        agent_id=self.agent_id,
                        action_type="buy",
                        parameters={"shares": shares_to_buy, "price": current_price}
                    )
            else:  # 30% chance to sell
                shares_to_sell = random.randint(1, self.shares)
                if shares_to_sell > 0:
                    return Action(
                        agent_id=self.agent_id,
                        action_type="sell",
                        parameters={"shares": shares_to_sell, "price": current_price}
                    )
        
        return Action(agent_id=self.agent_id, action_type="hold")
    
    def _noise_pessimist_decision(self, current_price: float, max_shares: int) -> Action:
        """Noise pessimist trading strategy"""
        if random.random() < 0.1:  # 10% chance to trade
            if random.random() < 0.3:  # 30% chance to buy
                shares_to_buy = random.randint(1, max_shares)
                if shares_to_buy > 0 and self.cash >= shares_to_buy * current_price:
                    return Action(
                        agent_id=self.agent_id,
                        action_type="buy",
                        parameters={"shares": shares_to_buy, "price": current_price}
                    )
            else:  # 70% chance to sell
                shares_to_sell = random.randint(1, self.shares)
                if shares_to_sell > 0:
                    return Action(
                        agent_id=self.agent_id,
                        action_type="sell",
                        parameters={"shares": shares_to_sell, "price": current_price}
                    )
        
        return Action(agent_id=self.agent_id, action_type="hold")
    
    def _momentum_decision(self, current_price: float, price_history: List[float], 
                          max_shares: int) -> Action:
        """Momentum trading strategy"""
        if len(price_history) < 2:
            return Action(agent_id=self.agent_id, action_type="hold")
        
        # Calculate price momentum
        recent_prices = price_history[-5:] if len(price_history) >= 5 else price_history
        if len(recent_prices) < 2:
            return Action(agent_id=self.agent_id, action_type="hold")
        
        momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        if momentum > 0.02:  # Strong upward momentum
            shares_to_buy = min(max_shares, int(self.cash / current_price * 0.05))
            if shares_to_buy > 0:
                return Action(
                    agent_id=self.agent_id,
                    action_type="buy",
                    parameters={"shares": shares_to_buy, "price": current_price}
                )
        elif momentum < -0.02:  # Strong downward momentum
            shares_to_sell = min(self.shares, int(self.shares * 0.05))
            if shares_to_sell > 0:
                return Action(
                    agent_id=self.agent_id,
                    action_type="sell",
                    parameters={"shares": shares_to_sell, "price": current_price}
                )
        
        return Action(agent_id=self.agent_id, action_type="hold")
    
    def _contrarian_decision(self, current_price: float, price_history: List[float], 
                           max_shares: int) -> Action:
        """Contrarian trading strategy"""
        if len(price_history) < 2:
            return Action(agent_id=self.agent_id, action_type="hold")
        
        # Calculate price momentum
        recent_prices = price_history[-5:] if len(price_history) >= 5 else price_history
        if len(recent_prices) < 2:
            return Action(agent_id=self.agent_id, action_type="hold")
        
        momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        if momentum > 0.02:  # Strong upward momentum - sell (contrarian)
            shares_to_sell = min(self.shares, int(self.shares * 0.05))
            if shares_to_sell > 0:
                return Action(
                    agent_id=self.agent_id,
                    action_type="sell",
                    parameters={"shares": shares_to_sell, "price": current_price}
                )
        elif momentum < -0.02:  # Strong downward momentum - buy (contrarian)
            shares_to_buy = min(max_shares, int(self.cash / current_price * 0.05))
            if shares_to_buy > 0:
                return Action(
                    agent_id=self.agent_id,
                    action_type="buy",
                    parameters={"shares": shares_to_buy, "price": current_price}
                )
        
        return Action(agent_id=self.agent_id, action_type="hold")
    
    def execute_action(self, action: Action, environment) -> bool:
        """Execute trading action"""
        if action.action_type == "buy":
            shares = action.parameters.get("shares", 0)
            price = action.parameters.get("price", 0)
            cost = shares * price
            
            if self.cash >= cost:
                self.cash -= cost
                self.shares += shares
                self.trade_history.append(("buy", shares, price))
                return True
        
        elif action.action_type == "sell":
            shares = action.parameters.get("shares", 0)
            price = action.parameters.get("price", 0)
            revenue = shares * price
            
            if self.shares >= shares:
                self.cash += revenue
                self.shares -= shares
                self.trade_history.append(("sell", shares, price))
                return True
        
        return True  # "hold" action always succeeds
    
    def update_wealth(self, current_price: float):
        """Update wealth based on current price"""
        self.wealth = self.cash + self.shares * current_price
        self.wealth_history.append(self.wealth)
        self.metadata["wealth"] = self.wealth
        self.metadata["cash"] = self.cash
        self.metadata["shares"] = self.shares


class MarketEnvironment:
    """Environment for market trading simulation"""
    
    def __init__(self, market_params: MarketParameters, **kwargs):
        self.market_params = market_params
        self.step_count = 0
        
        # Market state
        self.current_price = market_params.initial_price
        self.price_history = [market_params.initial_price]
        self.volume = 0
        self.fundamental_value = market_params.fundamental_value
        
        # Agents
        self.agents = {}
        self._create_agents()
        
        # Statistics
        self.market_data_history = []
        self.wealth_distribution_history = []
    
    def _create_agents(self):
        """Create agents with different strategies"""
        for i in range(self.market_params.num_agents):
            # Select strategy based on distribution
            strategy_choice = random.random()
            cumulative_prob = 0
            selected_strategy = TradingStrategy.FUNDAMENTALIST
            
            for strategy, prob in self.market_params.strategy_distribution.items():
                cumulative_prob += prob
                if strategy_choice <= cumulative_prob:
                    selected_strategy = TradingStrategy(strategy)
                    break
            
            # Create agent
            agent = MarketAgent(
                agent_id=i,
                strategy=selected_strategy,
                initial_wealth=self.market_params.initial_wealth
            )
            self.agents[i] = agent
    
    def get_market_data(self) -> Dict:
        """Get current market data"""
        return {
            "current_price": self.current_price,
            "price_history": self.price_history,
            "volume": self.volume,
            "fundamental_value": self.fundamental_value,
            "step": self.step_count
        }
    
    def update_price(self, buy_orders: int, sell_orders: int):
        """Update price based on supply and demand"""
        # Calculate net demand
        net_demand = buy_orders - sell_orders
        
        # Price change based on net demand and volatility
        price_change = (net_demand / self.market_params.num_agents) * self.market_params.volatility
        
        # Add random noise
        noise = random.gauss(0, self.market_params.volatility * 0.1)
        price_change += noise
        
        # Update price
        self.current_price *= (1 + price_change)
        self.current_price = max(self.current_price, 0.01)  # Prevent negative prices
        
        # Update volume
        self.volume = buy_orders + sell_orders
        
        # Add to history
        self.price_history.append(self.current_price)
    
    def update_state(self):
        """Update market state"""
        self.step_count += 1
        
        # Update fundamental value (random walk)
        fundamental_change = random.gauss(0, 0.001)
        self.fundamental_value *= (1 + fundamental_change)
        
        # Update agent wealth
        for agent in self.agents.values():
            agent.update_wealth(self.current_price)
        
        # Store market data
        market_data = {
            "step": self.step_count,
            "price": self.current_price,
            "volume": self.volume,
            "fundamental_value": self.fundamental_value,
            "price_change": (self.current_price - self.price_history[-2]) / self.price_history[-2] if len(self.price_history) > 1 else 0
        }
        self.market_data_history.append(market_data)
        
        # Store wealth distribution
        wealths = [agent.wealth for agent in self.agents.values()]
        self.wealth_distribution_history.append({
            "step": self.step_count,
            "mean_wealth": np.mean(wealths),
            "std_wealth": np.std(wealths),
            "min_wealth": np.min(wealths),
            "max_wealth": np.max(wealths),
            "gini_coefficient": self._calculate_gini(wealths)
        })
    
    def _calculate_gini(self, wealths: List[float]) -> float:
        """Calculate Gini coefficient for wealth inequality"""
        if len(wealths) <= 1:
            return 0.0
        
        wealths = sorted(wealths)
        n = len(wealths)
        cumsum = np.cumsum(wealths)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def get_market_summary(self) -> Dict:
        """Get comprehensive market summary"""
        if not self.market_data_history:
            return {}
        
        final_data = self.market_data_history[-1]
        final_wealth = self.wealth_distribution_history[-1] if self.wealth_distribution_history else {}
        
        # Calculate price statistics
        prices = [data["price"] for data in self.market_data_history]
        price_returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        return {
            "total_steps": self.step_count,
            "final_price": final_data["price"],
            "initial_price": self.market_params.initial_price,
            "price_change": (final_data["price"] - self.market_params.initial_price) / self.market_params.initial_price,
            "price_volatility": np.std(price_returns) if price_returns else 0,
            "max_price": max(prices),
            "min_price": min(prices),
            "total_volume": sum(data["volume"] for data in self.market_data_history),
            "final_wealth_mean": final_wealth.get("mean_wealth", 0),
            "final_wealth_std": final_wealth.get("std_wealth", 0),
            "final_gini": final_wealth.get("gini_coefficient", 0),
            "agent_count": len(self.agents)
        }


def run_market_simulation(market_params: MarketParameters, 
                         max_steps: int = 200) -> Tuple[MarketEnvironment, List[Dict]]:
    """Run a complete market trading simulation"""
    print(f"📈 Running Market Trading Simulation")
    print(f"   Agents: {market_params.num_agents}")
    print(f"   Initial Price: {market_params.initial_price}")
    print(f"   Volatility: {market_params.volatility}")
    print(f"   Strategies: {market_params.strategy_distribution}")
    
    # Create environment
    environment = MarketEnvironment(market_params)
    
    # Create engine
    engine = RuleEngine()
    
    # Run simulation
    results = []
    for step in range(max_steps):
        # Collect observations
        observations = []
        for agent in environment.agents.values():
            obs = agent.observe(environment)
            observations.append(obs)
        
        # Make decisions
        actions = []
        for agent, obs in zip(environment.agents.values(), observations):
            action = agent.decide(obs, engine)
            actions.append(action)
        
        # Execute actions and count orders
        buy_orders = 0
        sell_orders = 0
        
        for agent, action in zip(environment.agents.values(), actions):
            if action.action_type == "buy":
                buy_orders += action.parameters.get("shares", 0)
            elif action.action_type == "sell":
                sell_orders += action.parameters.get("shares", 0)
            
            agent.execute_action(action, environment)
        
        # Update market price
        environment.update_price(buy_orders, sell_orders)
        
        # Update environment
        environment.update_state()
        
        # Store step results
        step_data = {
            "step": step,
            "price": environment.current_price,
            "volume": environment.volume,
            "fundamental_value": environment.fundamental_value,
            "buy_orders": buy_orders,
            "sell_orders": sell_orders
        }
        results.append(step_data)
    
    return environment, results


def visualize_market_results(environment: MarketEnvironment, results: List[Dict], 
                           title: str = "Market Trading Model"):
    """Visualize market simulation results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract data
    steps = [r["step"] for r in results]
    prices = [r["price"] for r in results]
    volumes = [r["volume"] for r in results]
    fundamental_values = [r["fundamental_value"] for r in results]
    
    # Plot 1: Price and fundamental value over time
    axes[0, 0].plot(steps, prices, 'b-', linewidth=2, label='Market Price')
    axes[0, 0].plot(steps, fundamental_values, 'r--', linewidth=2, label='Fundamental Value')
    axes[0, 0].set_title(f'{title} - Price Evolution')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Trading volume over time
    axes[0, 1].plot(steps, volumes, 'g-', linewidth=2)
    axes[0, 1].set_title('Trading Volume Over Time')
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Volume')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Price returns (volatility)
    price_returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    axes[0, 2].plot(steps[1:], price_returns, 'm-', linewidth=1, alpha=0.7)
    axes[0, 2].set_title('Price Returns (Volatility)')
    axes[0, 2].set_xlabel('Time Steps')
    axes[0, 2].set_ylabel('Price Return')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Wealth distribution over time
    if environment.wealth_distribution_history:
        wealth_steps = [w["step"] for w in environment.wealth_distribution_history]
        mean_wealths = [w["mean_wealth"] for w in environment.wealth_distribution_history]
        std_wealths = [w["std_wealth"] for w in environment.wealth_distribution_history]
        
        axes[1, 0].plot(wealth_steps, mean_wealths, 'b-', linewidth=2, label='Mean Wealth')
        axes[1, 0].fill_between(wealth_steps, 
                               [m - s for m, s in zip(mean_wealths, std_wealths)],
                               [m + s for m, s in zip(mean_wealths, std_wealths)],
                               alpha=0.3, color='blue')
        axes[1, 0].set_title('Wealth Distribution Over Time')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Wealth')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Strategy distribution
    strategy_counts = {}
    for agent in environment.agents.values():
        strategy = agent.metadata.get("strategy", "unknown")
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    axes[1, 1].bar(strategy_counts.keys(), strategy_counts.values(), alpha=0.7)
    axes[1, 1].set_title('Agent Strategy Distribution')
    axes[1, 1].set_xlabel('Strategy')
    axes[1, 1].set_ylabel('Number of Agents')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Market summary
    summary = environment.get_market_summary()
    summary_text = f"""
    Initial Price: {summary.get('initial_price', 0):.2f}
    Final Price: {summary.get('final_price', 0):.2f}
    Price Change: {summary.get('price_change', 0):.1%}
    Volatility: {summary.get('price_volatility', 0):.3f}
    Max Price: {summary.get('max_price', 0):.2f}
    Min Price: {summary.get('min_price', 0):.2f}
    Total Volume: {summary.get('total_volume', 0):.0f}
    Final Gini: {summary.get('final_gini', 0):.3f}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    axes[1, 2].set_title('Market Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def compare_strategies():
    """Compare different strategy distributions"""
    print("\n🔬 Comparing Strategy Distributions")
    
    strategy_configs = [
        ("All Fundamentalist", {
            "fundamentalist": 1.0, "noise_optimist": 0.0, "noise_pessimist": 0.0,
            "momentum": 0.0, "contrarian": 0.0
        }),
        ("All Noise", {
            "fundamentalist": 0.0, "noise_optimist": 0.5, "noise_pessimist": 0.5,
            "momentum": 0.0, "contrarian": 0.0
        }),
        ("Mixed", {
            "fundamentalist": 0.4, "noise_optimist": 0.2, "noise_pessimist": 0.2,
            "momentum": 0.1, "contrarian": 0.1
        })
    ]
    
    results = {}
    for name, strategy_dist in strategy_configs:
        print(f"   Testing {name}...")
        params = MarketParameters(
            num_agents=50, strategy_distribution=strategy_dist,
            volatility=0.02, initial_price=100.0
        )
        env, sim_results = run_market_simulation(params, max_steps=100)
        results[name] = (env, sim_results)
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    for i, (name, (env, sim_results)) in enumerate(results.items()):
        plt.subplot(1, 3, i+1)
        steps = [r["step"] for r in sim_results]
        prices = [r["price"] for r in sim_results]
        
        plt.plot(steps, prices, linewidth=2)
        plt.title(f'{name} Strategy')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n📊 Strategy Comparison Summary:")
    for name, (env, sim_results) in results.items():
        summary = env.get_market_summary()
        print(f"   {name}:")
        print(f"     Price Change: {summary.get('price_change', 0):.1%}")
        print(f"     Volatility: {summary.get('price_volatility', 0):.3f}")
        print(f"     Final Gini: {summary.get('final_gini', 0):.3f}")


def main():
    """Main function to run market trading examples"""
    print("📈 LAMB Framework: Market Trading Model Example")
    print("=" * 60)
    
    # Example 1: Basic market simulation
    print("\n1️⃣ Basic Market Simulation")
    market_params = MarketParameters(
        num_agents=100,
        initial_price=100.0,
        fundamental_value=100.0,
        volatility=0.02,
        strategy_distribution={
            "fundamentalist": 0.4,
            "noise_optimist": 0.2,
            "noise_pessimist": 0.2,
            "momentum": 0.1,
            "contrarian": 0.1
        }
    )
    
    environment, results = run_market_simulation(market_params, max_steps=200)
    visualize_market_results(environment, results, "Basic Market Model")
    
    # Print summary
    summary = environment.get_market_summary()
    print(f"\n📈 Market Summary:")
    print(f"   Price Change: {summary.get('price_change', 0):.1%}")
    print(f"   Volatility: {summary.get('price_volatility', 0):.3f}")
    print(f"   Total Volume: {summary.get('total_volume', 0):.0f}")
    print(f"   Final Gini: {summary.get('final_gini', 0):.3f}")
    
    # Example 2: Compare strategies
    print("\n2️⃣ Strategy Comparison")
    compare_strategies()
    
    # Example 3: Volatility analysis
    print("\n3️⃣ Volatility Analysis")
    print("   Testing different volatility levels...")
    
    volatilities = [0.01, 0.02, 0.03, 0.04, 0.05]
    price_changes = []
    
    for vol in volatilities:
        params = MarketParameters(volatility=vol, num_agents=50)
        env, sim_results = run_market_simulation(params, max_steps=100)
        summary = env.get_market_summary()
        price_changes.append(summary.get('price_change', 0))
    
    plt.figure(figsize=(10, 6))
    plt.plot(volatilities, price_changes, 'o-', linewidth=2, markersize=8)
    plt.title('Price Change vs Market Volatility')
    plt.xlabel('Volatility')
    plt.ylabel('Price Change')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n✅ Market Trading Model Examples Complete!")
    print("   - Demonstrated global market dynamics")
    print("   - Compared different trading strategies")
    print("   - Analyzed volatility effects")
    print("   - Showed realistic market behavior")


if __name__ == "__main__":
    main()
