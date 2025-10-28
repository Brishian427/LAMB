"""
Researcher Prompting and Rule-Setting Example

This example demonstrates how researchers can customize agent behavior,
set custom prompts, define behavioral rules, and create agent personalities
using the LAMB framework's flexible prompting system.
"""

import asyncio
import os
from lamb.api import ResearchAPI
from lamb.config import SimulationConfig, ParadigmType, EngineType
from lamb.paradigms.grid import GridAgent, GridEnvironment
from lamb.llm.agent_prompts import (
    PromptManager, AgentPromptBuilder, AgentPersonality, 
    PromptTemplate, PromptType, BehavioralRule, RuleType,
    COMMON_PERSONALITIES, create_research_personality, create_behavioral_rule
)


def example_1_custom_prompts():
    """
    Example 1: Creating Custom Prompts for Research
    Shows how researchers can define their own prompts for specific research questions.
    """
    print("=== Example 1: Custom Research Prompts ===")
    
    # Create a prompt manager
    prompt_manager = PromptManager()
    
    # Define a custom prompt for a cooperation study
    cooperation_prompt = PromptTemplate(
        name="cooperation_study",
        prompt_type=PromptType.DECISION,
        template="""You are participating in a cooperation study.

Agent ID: ${agent_id}
Position: ${position}
Neighbors: ${neighbor_count} agents nearby

Research Context:
- You are in a resource-sharing environment
- Other agents may cooperate or compete with you
- Your decisions affect both your own and others' outcomes
- The study is investigating how cooperation emerges

Your Personality:
- Cooperation Level: ${cooperation_level} (0-1, where 1 is highly cooperative)
- Trust Level: ${trust_level} (0-1, where 1 is highly trusting)
- Risk Tolerance: ${risk_tolerance} (0-1, where 1 is risk-seeking)

Available Actions:
- cooperate: Share resources with neighbors (costs you, benefits them)
- compete: Try to take resources from neighbors (benefits you, costs them)
- observe: Just watch what others are doing
- move: Move to a different location

Current Situation:
${situation_description}

Based on your personality traits and the research context, choose your action.
Consider: What would a person with your cooperation level do in this situation?

Respond with JSON: {"action_type": "cooperate", "parameters": {"target_agent": "neighbor_1"}}

Your response:""",
        variables=["agent_id", "position", "neighbor_count", "cooperation_level", 
                  "trust_level", "risk_tolerance", "situation_description"],
        description="Custom prompt for cooperation research study"
    )
    
    # Add the custom prompt
    prompt_manager.add_template(cooperation_prompt)
    
    print("✅ Custom cooperation study prompt created!")
    print(f"   Template: {cooperation_prompt.name}")
    print(f"   Variables: {cooperation_prompt.variables}")
    print()


def example_2_agent_personalities():
    """
    Example 2: Creating Agent Personalities
    Shows how to define different personality types for agents.
    """
    print("=== Example 2: Agent Personalities ===")
    
    # Create custom personalities for a social dynamics study
    personalities = {
        "altruist": create_research_personality(
            name="altruist",
            traits={"cooperation": 0.9, "altruism": 0.95, "trust": 0.8},
            goals=["help others", "create harmony", "build community"],
            constraints=["never harm others", "always be fair"],
            description="Highly cooperative agent that prioritizes others' welfare"
        ),
        
        "selfish": create_research_personality(
            name="selfish",
            traits={"selfishness": 0.9, "competition": 0.8, "skepticism": 0.7},
            goals=["maximize personal gain", "outcompete others", "accumulate resources"],
            constraints=["follow basic rules", "avoid detection"],
            description="Self-interested agent focused on personal benefit"
        ),
        
        "conditional_cooperator": create_research_personality(
            name="conditional_cooperator",
            traits={"cooperation": 0.6, "reciprocity": 0.9, "caution": 0.7},
            goals=["cooperate with cooperators", "punish defectors", "maintain fairness"],
            constraints=["be fair but firm", "respond to others' behavior"],
            description="Agent that cooperates conditionally based on others' behavior"
        ),
        
        "neutral": create_research_personality(
            name="neutral",
            traits={"neutrality": 0.8, "indifference": 0.6, "stability": 0.7},
            goals=["maintain status quo", "cooperate", "stay safe"],
            constraints=["don't rock the boat", "follow the crowd"],
            description="Neutral agent that avoids taking strong positions"
        )
    }
    
    # Add personalities to prompt manager
    prompt_manager = PromptManager()
    for personality in personalities.values():
        prompt_manager.add_personality(personality)
    
    print("✅ Custom personalities created:")
    for name, personality in personalities.items():
        print(f"   {name}: {personality.description}")
        print(f"      Traits: {personality.traits}")
        print(f"      Goals: {personality.goals}")
    print()


def example_3_behavioral_rules():
    """
    Example 3: Defining Behavioral Rules
    Shows how to create rules that constrain or guide agent behavior.
    """
    print("=== Example 3: Behavioral Rules ===")
    
    # Create behavioral rules for a traffic simulation
    traffic_rules = [
        create_behavioral_rule(
            name="stop_at_red_light",
            condition="environment_state.get('traffic_light') == 'red'",
            action="action_type = 'stop'",
            rule_type=RuleType.CONSTRAINT,
            priority=10,
            description="Must stop when traffic light is red"
        ),
        
        create_behavioral_rule(
            name="maintain_safe_distance",
            condition="min([abs(agent_id - n) for n in neighbors]) < 2",
            action="action_type = 'slow_down'",
            rule_type=RuleType.CONSTRAINT,
            priority=8,
            description="Slow down when too close to other agents"
        ),
        
        create_behavioral_rule(
            name="prefer_right_lane",
            condition="position[1] > 0.5",  # If in left lane
            action="action_type = 'move_right'",
            rule_type=RuleType.PREFERENCE,
            priority=3,
            description="Prefer to move to right lane when possible"
        ),
        
        create_behavioral_rule(
            name="avoid_congestion",
            condition="len(neighbors) > 5",  # Too many neighbors
            action="action_type = 'change_route'",
            rule_type=RuleType.STRATEGY,
            priority=5,
            description="Change route when encountering congestion"
        )
    ]
    
    # Add rules to prompt manager
    prompt_manager = PromptManager()
    for rule in traffic_rules:
        prompt_manager.add_rule(rule)
    
    print("✅ Behavioral rules created:")
    for rule in traffic_rules:
        print(f"   {rule.name}: {rule.description}")
        print(f"      Condition: {rule.condition}")
        print(f"      Action: {rule.action}")
        print(f"      Priority: {rule.priority}")
    print()


def example_4_research_scenario():
    """
    Example 4: Complete Research Scenario
    Shows how to combine custom prompts, personalities, and rules for a research study.
    """
    print("=== Example 4: Complete Research Scenario ===")
    
    # Create a prompt manager with custom templates
    prompt_manager = PromptManager()
    
    # Define a custom prompt for a social influence study
    influence_prompt = PromptTemplate(
        name="social_influence_study",
        prompt_type=PromptType.DECISION,
        template="""You are participating in a social influence study.

Agent ID: ${agent_id}
Position: ${position}
Neighbors: ${neighbor_count} agents nearby

Study Context:
- You are in a social network where opinions spread
- You can observe others' opinions and behaviors
- Your decisions influence others around you
- The study examines how social influence affects group dynamics

Your Characteristics:
- Influence Susceptibility: ${influence_susceptibility} (0-1, how easily you're influenced)
- Influence Strength: ${influence_strength} (0-1, how much you influence others)
- Opinion Strength: ${opinion_strength} (0-1, how strongly you hold your opinions)
- Social Orientation: ${social_orientation} (individualist/collectivist)

Current Situation:
- Your current opinion: ${current_opinion}
- Neighbors' opinions: ${neighbor_opinions}
- Group consensus: ${group_consensus}
- Social pressure level: ${social_pressure}

Available Actions:
- maintain_opinion: Keep your current opinion
- change_opinion: Adopt a different opinion
- influence_others: Try to convince neighbors
- observe: Just watch and learn
- move: Change your social position

Research Question: How do individual characteristics affect social influence dynamics?

Based on your characteristics and the current social situation, choose your action.
Consider: How would someone with your influence susceptibility and strength behave?

Respond with JSON: {"action_type": "maintain_opinion", "parameters": {"confidence": 0.8}}

Your response:""",
        variables=["agent_id", "position", "neighbor_count", "influence_susceptibility",
                  "influence_strength", "opinion_strength", "social_orientation",
                  "current_opinion", "neighbor_opinions", "group_consensus", "social_pressure"],
        description="Custom prompt for social influence research"
    )
    
    # Create personality types for the study
    personalities = {
        "high_influencer": create_research_personality(
            name="high_influencer",
            traits={"influence_strength": 0.9, "confidence": 0.8, "charisma": 0.7},
            goals=["influence others", "lead group", "shape consensus"],
            constraints=["be persuasive but not manipulative"],
            description="Agent with high influence over others"
        ),
        
        "highly_susceptible": create_research_personality(
            name="highly_susceptible",
            traits={"influence_susceptibility": 0.9, "conformity": 0.8, "uncertainty": 0.6},
            goals=["fit in", "follow group", "cooperate"],
            constraints=["don't be too easily swayed"],
            description="Agent easily influenced by others"
        ),
        
        "independent": create_research_personality(
            name="independent",
            traits={"independence": 0.9, "critical_thinking": 0.8, "resistance": 0.7},
            goals=["think independently", "resist pressure", "form own opinions"],
            constraints=["be open to good arguments"],
            description="Agent that resists social influence"
        )
    }
    
    # Add everything to prompt manager
    prompt_manager.add_template(influence_prompt)
    for personality in personalities.values():
        prompt_manager.add_personality(personality)
    
    print("✅ Complete research scenario created:")
    print(f"   Custom prompt: {influence_prompt.name}")
    print(f"   Personalities: {list(personalities.keys())}")
    print(f"   Variables: {len(influence_prompt.variables)}")
    print()


def example_5_using_in_simulation():
    """
    Example 5: Using Custom Prompts in a Simulation
    Shows how to integrate custom prompts and personalities into a LAMB simulation.
    """
    print("=== Example 5: Using Custom Prompts in Simulation ===")
    
    # Create a custom prompt manager
    prompt_manager = PromptManager()
    
    # Add a custom prompt for a market simulation
    market_prompt = PromptTemplate(
        name="market_trading",
        prompt_type=PromptType.DECISION,
        template="""You are a trader in a market simulation.

Agent ID: ${agent_id}
Position: ${position}
Wealth: ${wealth}
Inventory: ${inventory}

Market Conditions:
- Current Price: ${current_price}
- Price Trend: ${price_trend}
- Market Volatility: ${volatility}
- Number of Traders: ${num_traders}

Your Trading Style:
- Risk Tolerance: ${risk_tolerance} (0-1)
- Patience: ${patience} (0-1)
- Aggressiveness: ${aggressiveness} (0-1)

Available Actions:
- buy: Purchase goods at current price
- sell: Sell goods at current price
- hold: Wait and observe
- negotiate: Try to negotiate a better price
- exit: Leave the market

Current Situation:
${market_situation}

Based on your trading style and market conditions, choose your action.
Consider: What would a trader with your risk tolerance and patience do?

Respond with JSON: {"action_type": "buy", "parameters": {"quantity": 10, "max_price": 100}}

Your response:""",
        variables=["agent_id", "position", "wealth", "inventory", "current_price",
                  "price_trend", "volatility", "num_traders", "risk_tolerance",
                  "patience", "aggressiveness", "market_situation"],
        description="Custom prompt for market trading simulation"
    )
    
    prompt_manager.add_template(market_prompt)
    
    # Create trader personalities
    trader_personalities = {
        "conservative": create_research_personality(
            name="conservative",
            traits={"risk_tolerance": 0.2, "patience": 0.9, "caution": 0.8},
            goals=["preserve capital", "avoid losses", "steady gains"],
            constraints=["never risk more than 10%", "always have reserves"],
            description="Conservative trader focused on capital preservation"
        ),
        
        "aggressive": create_research_personality(
            name="aggressive",
            traits={"risk_tolerance": 0.9, "aggressiveness": 0.8, "confidence": 0.7},
            goals=["maximize profits", "take opportunities", "beat market"],
            constraints=["don't go bankrupt", "follow basic rules"],
            description="Aggressive trader seeking maximum returns"
        )
    }
    
    for personality in trader_personalities.values():
        prompt_manager.add_personality(personality)
    
    print("✅ Market simulation setup created:")
    print(f"   Custom prompt: {market_prompt.name}")
    print(f"   Trader types: {list(trader_personalities.keys())}")
    print()
    
    # Show how to use in a simulation
    print("To use this in a simulation:")
    print("""
    # Create configuration
    config = SimulationConfig(
        name="Market Trading Study",
        paradigm=ParigmType.GRID,
        num_agents=50,
        max_steps=1000,
        llm_config={
            "model": "gpt-3.5-turbo",
            "temperature": 0.7
        }
    )
    
    # Create engine with custom prompt manager
    engine = LLMEngine(
        prompt_manager=prompt_manager,
        custom_prompt_template="market_trading"
    )
    
    # Create simulation
    api = ResearchAPI()
    env, engine = api.create_simulation(config, GridAgent, GridEnvironment)
    
    # Run simulation with custom prompts
    metrics = await api.run_simulation(env, engine)
    """)


def main():
    """
    Main function demonstrating all prompting capabilities.
    """
    print("LAMB Framework: Researcher Prompting and Rule-Setting")
    print("=" * 60)
    print()
    
    # Run all examples
    example_1_custom_prompts()
    example_2_agent_personalities()
    example_3_behavioral_rules()
    example_4_research_scenario()
    example_5_using_in_simulation()
    
    print("🎉 All examples completed!")
    print()
    print("Key Benefits for Researchers:")
    print("✅ Custom Prompts - Define exactly how agents should think")
    print("✅ Agent Personalities - Create diverse behavioral types")
    print("✅ Behavioral Rules - Set constraints and preferences")
    print("✅ Research Scenarios - Tailor simulations to specific studies")
    print("✅ Easy Integration - Use custom prompts in any simulation")
    print()
    print("This gives researchers complete control over agent behavior")
    print("while maintaining the power and flexibility of LLM-driven agents!")


if __name__ == "__main__":
    main()
