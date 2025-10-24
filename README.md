# 🧠 Beltone AI Hackathon — Multi-Warehouse Vehicle Routing Optimizer

![Fulfillment 100%](https://img.shields.io/badge/Fulfillment-100%25-brightgreen)
![Cost £1350](https://img.shields.io/badge/Total%20Cost-£1350-blue)
![Python 3.10](https://img.shields.io/badge/Python-3.10-yellow)
![Top Rank](https://img.shields.io/badge/Leaderboard-Top%203-orange)


---

### 🚚 Intelligent Multi-Warehouse Vehicle Routing (MWVRP)
This project is our official submission for the **Beltone 2nd AI Hackathon**, powered by **Robin Logistics**.  
It solves a **large-scale multi-warehouse vehicle routing problem (MWVRP)** for Cairo’s 332K-node directed road network, achieving **100% order fulfillment** with a competitive **low-cost solution**.

---

## 🏆 Leaderboard Performance

| Metric | Value |
|--------|-------:|
| **Leaderboard Rank** | 🥈 **#77 Overall (Final Round)** |
| **Fulfillment Rate** | **100%** |
| **Total Cost** | **£1,350** |
| **Orders Served** | **50 / 50** |
| **Vehicles Used** | **8 / 12** |

<p align="center">
  <img src="Core Metrics - Beltone Hackathon.png" width="80%" alt="Solver Core Metrics Summary">
</p>

---

## ⚙️ Problem Overview

Modern logistics faces complex optimization challenges:
- Multiple warehouses with **limited inventory**
- **Directed** urban road network with 332,000+ nodes
- Vehicle **weight & volume capacity limits**
- **Multi-pickup** & **multi-warehouse** coordination
- Real-time **inventory tracking** and **fulfillment validation**

Our solver intelligently assigns orders, builds feasible routes, and minimizes operational cost under all constraints.

---

## 🧩 Environment Overview

The solution runs within the `robin-logistics-env` simulation package:

```bash
pip install robin-logistics-env
