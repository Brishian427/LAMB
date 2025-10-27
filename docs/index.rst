Welcome to LAMB's documentation!
================================

LAMB (LLM Agent Model Base) is a unified framework for building agent-based models with Large Language Model integration, supporting multiple simulation paradigms and behavioral engines.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   user_guide
   api_reference
   examples
   theory_guide
   migration_guide
   performance_guide
   development_guide

Features
--------

* **Multi-Paradigm Support**: Grid, Physics, and Network simulation paradigms
* **LLM Integration**: Seamless Large Language Model agent behavior
* **Composition Architecture**: Modular design for easy extension
* **High Performance**: Optimized for 10,000+ agents
* **Research Ready**: Built-in metrics, visualization, and analysis tools
* **Academic Focus**: Designed for social science and complexity research

Quick Start
-----------

.. code-block:: python

   from lamb import ResearchAPI

   # Create a simple grid simulation
   api = ResearchAPI()
   api.create_simulation(
       paradigm="grid",
       num_agents=100,
       engine_type="rule",
       max_steps=1000
   )

   # Run simulation
   results = api.run_simulation()
   print(f"Simulation completed with {len(results)} steps")

Installation
------------

.. code-block:: bash

   pip install lamb-abm

For LLM integration:

.. code-block:: bash

   pip install lamb-abm[llm]

For full functionality:

.. code-block:: bash

   pip install lamb-abm[all]

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
