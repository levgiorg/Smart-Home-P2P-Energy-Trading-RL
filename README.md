# SMART HOME CARTEL

This project provides an environment that simulates a group of houses where each house can consume, generate, buy, and sell energy. The environment exposes the following key concepts:

- **House State**: Inside temperature, ambient temperature, battery level, power generation, power demand, etc.  
- **Actions**:  
  1. HVAC adjustment (e_t).  
  2. Battery charge/discharge (a_batt).  
  3. Setting a selling price (bounded by the grid price).  

Over each time step, the environment updates:
- Energy balance, temperature, and battery usage based on house actions. 
- Costs, trading profits, and penalties if temperatures exceed specified thresholds.  
- Peer-to-peer energy transactions among houses with excess energy and those with energy deficits.