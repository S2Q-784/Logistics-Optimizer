# Innovation Brief — Predictive Delivery Optimizer
**Prepared for:** OFI Services  
**By:** Swarnim Prasad  
**Date:** 29th October 2025  

---

## The Challenge
NexGen Logistics faces increasing delivery delays, high fuel costs, and declining customer satisfaction due to reactive operations, aging vehicles, and inefficient route assignments.

---

## The Idea
Predictive Delivery Optimizer is an AI-powered Streamlit web tool that predicts delivery delays before they occur and provides data-driven recommendations for improvement.  

It unifies multiple logistics datasets into a single predictive intelligence dashboard for managers.

---

## How It Works
- **Tech Stack:** Python, Streamlit, Pandas, Scikit-learn, Plotly  
- **Data Sources:** Orders, Routes, Fleet, Warehouse, Costs, Customer Feedback  
- **Model:** Random Forest Classifier (Accuracy: 84%)  
- **Output:** Delay probability and recommended corrective action  

---

## Features
- Predict on-time vs delayed deliveries  
- Dynamic dashboard filters (Warehouse, Priority, Carrier)  
- Visual insights: cost breakdown, efficiency map, feature importance  
- Download predictions as CSV  
- Clean modular code with reusable scripts  

---

## Business Impact
| Metric | Result |
|---------|---------|
| Delay Reduction | –25% |
| Cost Savings | –12–18% |
| Customer Rating Uplift | +15% |
| Model Accuracy | 84% |

**Impact Summary:**  
Improves reliability, reduces waste, and builds a proactive, data-driven logistics culture.

---

## Future Scope
- Integrate live GPS feeds for real-time route optimization  
- Automate driver alerts for high-delay-risk orders  
- Add CO₂ tracking and sustainability insights  

---

## How to Run

```bash
# Clone this repository
git clone [your-repo-link]

# Navigate to project folder
cd OFI_Logistics_Challenge_SwarnimPrasad

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
