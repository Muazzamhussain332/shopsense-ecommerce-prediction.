# ShopSense — E-commerce Purchase Prediction MVP

## Setup
1. In Google Colab save the **complete fitted pipeline**:
   ```python
   import joblib
   joblib.dump(final_model, 'ecommerce_purchase_model.pkl')
   from google.colab import files
   files.download('ecommerce_purchase_model.pkl')
   ```
2. Put `ecommerce_purchase_model.pkl` in the same folder as `app.py`.
3. Create an environment and install packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch:
   ```bash
   streamlit run app.py
   ```

## Notes
- Inputs are the exact 15 features used in training (including categorical values such as `June`, `Returning_Visitor` and Boolean `Weekend`).
- `predict_proba` is a model score and may not be calibrated.
- Ensure your deployed scikit-learn version is compatible with the Colab training environment. Matching versions is safest for joblib model files.
- This is an educational prototype, not a production SaaS platform.
