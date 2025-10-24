# 🧠 Beltone AI Hackathon — Multi-Warehouse Vehicle Routing Optimizer

<p align="center">
  <img src="assets/cover.png" width="80%" alt="MWVRP Routing System Visualization">
</p>

### 🚚 Intelligent Multi-Warehouse Vehicle Routing (MWVRP)
This project is our submission for the **Beltone 2nd AI Hackathon**, powered by **Robin Logistics**.  
It aims to solve a **real-world multi-warehouse vehicle routing problem (MWVRP)** under heavy operational constraints such as:
- Limited warehouse inventory  
- Directed, large-scale road networks (Cairo map with 332K nodes)  
- Vehicle capacity (weight/volume) limits  
- Multi-pickup, multi-warehouse coordination  
- Real-time inventory tracking and cost efficiency

---

## 🧩 Problem Overview

Modern logistics requires dynamic planning that adapts to road directionality, limited stock, and constrained vehicle fleets.  
Our solver optimizes:
1. **Order fulfillment** — ensuring every customer receives their order.  
2. **Cost efficiency** — balancing fixed + variable transport costs.  
3. **Vehicle utilization** — reducing underused routes.  
4. **Feasible routes** — ensuring all nodes are connected through valid road paths.

---

## ⚙️ Environment

The project runs inside the `robin-logistics-env` simulation package.  
Install it with:

```bash
pip install robin-logistics-env
