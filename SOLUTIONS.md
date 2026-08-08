# Practical Solutions for Low Food Access

The ML model should be used as a **screening and prioritization tool**, not as an automatic policy engine.

## 1. Healthy-food retail expansion
Use when grocery/supercenter availability is comparatively low.

Possible actions:
- healthy-food financing
- tax/lease incentives
- community-owned grocery stores
- farmers' markets
- mobile markets

## 2. SNAP/WIC retailer expansion
Use when SNAP/WIC-authorized retailer density is low.

Possible actions:
- recruit existing retailers into authorization programs
- technical assistance for application/compliance
- pair authorization with minimum healthy-food stocking goals

## 3. Healthy corner-store conversion
Use when convenience-store density is high while grocery availability is low.

Possible actions:
- refrigeration grants
- produce stocking support
- healthier staple requirements
- supplier/logistics partnerships

## 4. Rural access
Use especially for nonmetro counties.

Possible actions:
- scheduled mobile grocery routes
- delivery and pickup hubs
- transport vouchers
- partnerships with schools, libraries, clinics, or community centers as pickup locations

## 5. Integrate food assistance with physical access
Benefits alone do not solve distance/access barriers.

Possible actions:
- combine SNAP/school-meal outreach with retailer-access projects
- food-box delivery for vulnerable populations
- coordinate with local food banks and public-health departments

## 6. Validation before investment
Before selecting an intervention, collect finer-grained evidence:
- census-tract food-access data
- GIS travel distance/time
- households without vehicles
- public transit availability
- food prices
- store hours and assortment
- community interviews
- retailer feasibility

## 7. How to use model output operationally

```text
Model risk score
      ↓
Priority screening
      ↓
Local GIS + community validation
      ↓
Identify dominant access barrier
      ↓
Choose intervention
      ↓
Pilot
      ↓
Measure before/after access indicators
```

## 8. Suggested success KPIs
- average travel distance/time to a healthy-food retailer
- population share with low access
- number of grocery/SNAP/WIC retailers
- healthy-food availability score
- program utilization
- intervention cost per resident reached

**Do not interpret Random Forest feature importance as causal effect.**
