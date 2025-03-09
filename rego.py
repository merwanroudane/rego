import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import pingouin as pg
from dominance_analysis import Dominance
import io
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Regression Analysis Tool")

st.title("Comprehensive Regression Analysis")
st.markdown("""
This application performs a complete analysis for linear regression, including:
- Descriptive statistics and correlation analysis
- Pre-regression tests (multicollinearity, normality, etc.)
- Linear regression with detailed results
- Dominance analysis for variable importance
- Random Forest comparison with feature importance
- Visualizations throughout the analysis
""")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Load data
    df = pd.read_excel(uploaded_file)

    # Display basic information about the dataset
    st.sidebar.header("Dataset Information")
    st.sidebar.write(f"Number of rows: {df.shape[0]}")
    st.sidebar.write(f"Number of columns: {df.shape[1]}")

    # Variable selection
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Target variable selection
    target_var = st.sidebar.selectbox("Select Target Variable", numeric_cols)

    # Predictor variables selection
    remaining_cols = [col for col in numeric_cols if col != target_var]
    predictors = st.sidebar.multiselect("Select Predictor Variables", remaining_cols,
                                        default=remaining_cols[:min(5, len(remaining_cols))])

    if not predictors:
        st.warning("Please select at least one predictor variable.")
    else:
        # Create a DataFrame with only the selected variables
        analysis_df = df[[target_var] + predictors].copy()

        # Handle missing values
        if analysis_df.isnull().sum().sum() > 0:
            handle_na = st.sidebar.radio("How to handle missing values?",
                                         ["Drop rows with missing values", "Fill with mean"])
            if handle_na == "Drop rows with missing values":
                analysis_df = analysis_df.dropna()
                st.sidebar.write(f"Rows after dropping NA: {analysis_df.shape[0]}")
            else:
                analysis_df = analysis_df.fillna(analysis_df.mean())

        # Main content divided into tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Descriptive Statistics", "Pre-regression Tests", "Linear Regression", "Dominance Analysis",
             "Random Forest"])

        with tab1:
            st.header("Descriptive Statistics")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Summary Statistics")
                st.write(analysis_df.describe())

                st.subheader("Correlation Matrix")
                corr_matrix = analysis_df.corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                cmap = sns.diverging_palette(230, 20, as_cmap=True)
                sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                            square=True, linewidths=.5, annot=True, fmt=".2f", cbar_kws={"shrink": .5})
                st.pyplot(fig)

            with col2:
                st.subheader("Distributions")
                for col in [target_var] + predictors:
                    fig = px.histogram(analysis_df, x=col, title=f"Distribution of {col}")
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("Correlation with Target")
                corr_with_target = corr_matrix[target_var].drop(target_var).sort_values(ascending=False)
                fig = px.bar(
                    x=corr_with_target.values,
                    y=corr_with_target.index,
                    orientation='h',
                    title=f"Correlation with {target_var}",
                    labels={'x': 'Correlation', 'y': 'Variable'}
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.header("Pre-regression Tests")

            # Prepare X and y
            X = analysis_df[predictors]
            y = analysis_df[target_var]

            col1, col2 = st.columns(2)

            with col1:
                # Multicollinearity test
                st.subheader("Multicollinearity Test (VIF)")
                X_with_const = sm.add_constant(X)
                vif_data = pd.DataFrame()
                vif_data["Variable"] = X_with_const.columns
                vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in
                                   range(X_with_const.shape[1])]
                vif_data = vif_data.drop(0)  # Drop the constant term
                st.write(vif_data)

                # VIF interpretation
                if any(vif_data["VIF"] > 10):
                    st.warning("VIF values > 10 indicate potential multicollinearity issues.")
                else:
                    st.success("No severe multicollinearity detected (all VIF values < 10).")

                # Plot VIF values
                fig = px.bar(
                    vif_data,
                    x="Variable",
                    y="VIF",
                    title="Variance Inflation Factor (VIF)",
                    labels={'VIF': 'VIF Value', 'Variable': 'Predictor Variable'}
                )
                fig.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="Critical threshold")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Normality test for residuals
                st.subheader("Normality Test")

                # Fit a simple model to get residuals
                model = sm.OLS(y, sm.add_constant(X)).fit()
                residuals = model.resid

                # Shapiro-Wilk test
                shapiro_test = stats.shapiro(residuals)
                st.write("Shapiro-Wilk Test for Normality:")
                st.write(f"W-statistic: {shapiro_test[0]:.4f}")
                st.write(f"p-value: {shapiro_test[1]:.4f}")

                if shapiro_test[1] < 0.05:
                    st.warning("Residuals are not normally distributed (p < 0.05).")
                else:
                    st.success("Residuals appear to be normally distributed (p > 0.05).")

                # QQ plot for residuals
                fig, ax = plt.subplots(figsize=(10, 6))
                stats.probplot(residuals, plot=ax)
                st.pyplot(fig)

                # Residuals histogram
                fig = px.histogram(residuals, title="Residuals Distribution", labels={'value': 'Residuals'})
                st.plotly_chart(fig, use_container_width=True)

                # Durbin-Watson test for autocorrelation
                st.subheader("Autocorrelation Test")
                dw_stat = durbin_watson(residuals)
                st.write(f"Durbin-Watson statistic: {dw_stat:.4f}")

                if dw_stat < 1.5 or dw_stat > 2.5:
                    st.warning("Possible autocorrelation in residuals (Durbin-Watson not close to 2).")
                else:
                    st.success("No significant autocorrelation detected (Durbin-Watson close to 2).")

        with tab3:
            st.header("Linear Regression Analysis")

            # Train-test split option
            use_split = st.checkbox("Use train-test split", value=True)

            if use_split:
                test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                st.write(f"Training set: {X_train.shape[0]} samples")
                st.write(f"Test set: {X_test.shape[0]} samples")
            else:
                X_train, y_train = X, y
                X_test, y_test = X, y

            # Standardize option
            standardize = st.checkbox("Standardize features", value=False)

            if standardize:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                X_train_for_model = X_train_scaled
                X_test_for_model = X_test_scaled
            else:
                X_train_for_model = X_train
                X_test_for_model = X_test

            # Statsmodels for detailed statistics
            X_train_sm = sm.add_constant(X_train)
            X_test_sm = sm.add_constant(X_test)

            model_sm = sm.OLS(y_train, X_train_sm).fit()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Model Summary")
                model_summary = model_sm.summary()
                st.text(str(model_summary))

                # Extract coefficients with confidence intervals
                coef_df = pd.DataFrame({
                    'Coefficient': model_sm.params,
                    'Std Error': model_sm.bse,
                    'P-value': model_sm.pvalues,
                    'Lower CI': model_sm.conf_int()[0],
                    'Upper CI': model_sm.conf_int()[1]
                })

                st.subheader("Coefficients")
                st.write(coef_df)

                # Coefficient plot
                fig = go.Figure()
                # Skip the intercept for the plot
                coef_data = coef_df.iloc[1:].reset_index()

                # Confidence intervals
                fig.add_trace(go.Scatter(
                    x=coef_data['index'],
                    y=coef_data['Coefficient'],
                    mode='markers',
                    marker=dict(size=10, color='blue'),
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=coef_data['Upper CI'] - coef_data['Coefficient'],
                        arrayminus=coef_data['Coefficient'] - coef_data['Lower CI']
                    ),
                    name='Coefficient'
                ))

                fig.add_hline(y=0, line_dash='dash', line_color='red')
                fig.update_layout(
                    title='Regression Coefficients with 95% Confidence Intervals',
                    xaxis_title='Variable',
                    yaxis_title='Coefficient Value'
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Model Evaluation")

                # Predictions
                y_train_pred = model_sm.predict(X_train_sm)
                y_test_pred = model_sm.predict(X_test_sm)

                # Metrics
                train_metrics = {
                    'R² (Coefficient of Determination)': r2_score(y_train, y_train_pred),
                    'Adjusted R²': 1 - (1 - r2_score(y_train, y_train_pred)) * (len(y_train) - 1) / (
                                len(y_train) - X_train.shape[1] - 1),
                    'Mean Squared Error (MSE)': mean_squared_error(y_train, y_train_pred),
                    'Root Mean Squared Error (RMSE)': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                    'Mean Absolute Error (MAE)': mean_absolute_error(y_train, y_train_pred)
                }

                if use_split:
                    test_metrics = {
                        'R² (Coefficient of Determination)': r2_score(y_test, y_test_pred),
                        'Adjusted R²': 1 - (1 - r2_score(y_test, y_test_pred)) * (len(y_test) - 1) / (
                                    len(y_test) - X_test.shape[1] - 1),
                        'Mean Squared Error (MSE)': mean_squared_error(y_test, y_test_pred),
                        'Root Mean Squared Error (RMSE)': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                        'Mean Absolute Error (MAE)': mean_absolute_error(y_test, y_test_pred)
                    }

                st.write("Training Metrics:")
                for metric, value in train_metrics.items():
                    st.write(f"{metric}: {value:.4f}")

                if use_split:
                    st.write("\nTest Metrics:")
                    for metric, value in test_metrics.items():
                        st.write(f"{metric}: {value:.4f}")

                # Actual vs Predicted Plot
                fig = px.scatter(
                    x=y_train,
                    y=y_train_pred,
                    labels={'x': 'Actual', 'y': 'Predicted'},
                    title='Actual vs Predicted (Training Set)'
                )
                fig.add_trace(
                    go.Scatter(
                        x=[min(y_train), max(y_train)],
                        y=[min(y_train), max(y_train)],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name='Perfect Prediction'
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

                if use_split:
                    fig = px.scatter(
                        x=y_test,
                        y=y_test_pred,
                        labels={'x': 'Actual', 'y': 'Predicted'},
                        title='Actual vs Predicted (Test Set)'
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[min(y_test), max(y_test)],
                            y=[min(y_test), max(y_test)],
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name='Perfect Prediction'
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Residuals plot
                fig = px.scatter(
                    x=y_train_pred,
                    y=model_sm.resid,
                    labels={'x': 'Predicted', 'y': 'Residuals'},
                    title='Residuals vs Predicted'
                )
                fig.add_hline(y=0, line_dash='dash', line_color='red')
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.header("Dominance Analysis")

            st.write("""
            Dominance Analysis is a method for determining the relative importance of predictors in a regression model.
            It assesses how much each predictor contributes to the overall model by examining its contribution 
            across all possible subset models.
            """)

            try:
                # Create a new dataframe with only the needed columns to avoid issues
                dom_df = analysis_df.copy()

                # Run dominance analysis
                dominance_reg = Dominance(data=dom_df, target=target_var, objective=1)

                # Get incremental R-squared
                incr_rsquare = dominance_reg.incremental_rsquare()
                st.subheader("Incremental R-squared")
                st.write(incr_rsquare)

                # Get dominance statistics
                dominance_stats = dominance_reg.dominance_stats()
                st.subheader("Dominance Statistics")
                st.write(dominance_stats)

                # Get dominance levels
                dominance_levels = dominance_reg.dominance_level()
                st.subheader("Dominance Levels")
                st.write(dominance_levels)

                # Create plots for dominance analysis
                col1, col2 = st.columns(2)

                with col1:
                    # Plot incremental R-squared
                    fig = plt.figure(figsize=(12, 8))
                    dominance_reg.plot_incremental_rsquare()
                    st.pyplot(fig)

                with col2:
                    # Create a custom bar chart of overall average contribution
                    overall_contribution = dominance_stats['Total Dominance']
                    fig = px.bar(
                        x=overall_contribution.index,
                        y=overall_contribution.values,
                        title="Total Dominance (Overall Average R² Contribution)",
                        labels={'x': 'Predictor', 'y': 'R² Contribution'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Create a pie chart showing relative importance
                    fig = px.pie(
                        values=overall_contribution.values,
                        names=overall_contribution.index,
                        title="Relative Importance of Predictors"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Also show Pingouin's relative importance
                st.subheader("Pingouin Relative Importance Analysis")
                pingouin_results = pg.linear_regression(
                    X=X,
                    y=y,
                    relimp=True
                )
                st.write(pingouin_results)

                # Plot Pingouin's relative importance
                if 'relimp' in pingouin_results.columns:
                    fig = px.bar(
                        x=pingouin_results.index,
                        y=pingouin_results['relimp'],
                        title="Pingouin Relative Importance",
                        labels={'x': 'Predictor', 'y': 'Relative Importance'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error in Dominance Analysis: {str(e)}")
                st.error("Dominance Analysis may fail with certain datasets. Trying alternative methods...")

                # Alternative: Pingouin's relative importance
                st.subheader("Pingouin Relative Importance Analysis")
                try:
                    pingouin_results = pg.linear_regression(
                        X=X,
                        y=y,
                        relimp=True
                    )
                    st.write(pingouin_results)

                    # Plot Pingouin's relative importance
                    if 'relimp' in pingouin_results.columns:
                        fig = px.bar(
                            x=pingouin_results.index,
                            y=pingouin_results['relimp'],
                            title="Pingouin Relative Importance",
                            labels={'x': 'Predictor', 'y': 'Relative Importance'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e2:
                    st.error(f"Error in Pingouin analysis: {str(e2)}")

        with tab5:
            st.header("Random Forest Regression")

            st.write("""
            Random Forest is an ensemble learning method that builds multiple decision trees and merges their 
            predictions. It provides an alternative measure of feature importance.
            """)

            # Random Forest parameters
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("Number of trees", 10, 500, 100, 10)
                max_depth = st.slider("Maximum depth", 2, 30, 10, 1)
            with col2:
                min_samples_split = st.slider("Minimum samples to split", 2, 20, 2, 1)
                min_samples_leaf = st.slider("Minimum samples in leaf", 1, 20, 1, 1)

            # Train Random Forest
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )

            if use_split:
                rf.fit(X_train, y_train)
                y_train_pred_rf = rf.predict(X_train)
                y_test_pred_rf = rf.predict(X_test)

                train_r2_rf = r2_score(y_train, y_train_pred_rf)
                test_r2_rf = r2_score(y_test, y_test_pred_rf)

                st.write(f"Training R²: {train_r2_rf:.4f}")
                st.write(f"Test R²: {test_r2_rf:.4f}")
            else:
                rf.fit(X, y)
                y_pred_rf = rf.predict(X)
                r2_rf = r2_score(y, y_pred_rf)
                st.write(f"R²: {r2_rf:.4f}")

            # Feature importance
            importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=False)

            st.subheader("Random Forest Feature Importance")
            st.write(importance)

            # Plot feature importance
            fig = px.bar(
                importance,
                x='Feature',
                y='Importance',
                title='Random Forest Feature Importance',
                labels={'Importance': 'Importance Score', 'Feature': 'Predictor Variable'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Compare with Linear Regression coefficients
            if st.checkbox("Compare with Linear Regression coefficients"):
                # Get absolute values of coefficients (excluding intercept)
                coef_abs = pd.DataFrame({
                    'Feature': X.columns,
                    'Absolute Coefficient': np.abs(model_sm.params.values[1:])
                }).sort_values('Absolute Coefficient', ascending=False)

                # Normalize coefficients for comparison
                coef_abs['Normalized Coefficient'] = coef_abs['Absolute Coefficient'] / coef_abs[
                    'Absolute Coefficient'].sum()

                # Prepare comparison data
                comparison = pd.merge(
                    importance,
                    coef_abs[['Feature', 'Normalized Coefficient']],
                    on='Feature'
                )

                # Create comparison plot
                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=comparison['Feature'],
                    y=comparison['Importance'],
                    name='Random Forest Importance',
                    marker_color='blue'
                ))

                fig.add_trace(go.Bar(
                    x=comparison['Feature'],
                    y=comparison['Normalized Coefficient'],
                    name='Linear Regression Coefficient (absolute, normalized)',
                    marker_color='red'
                ))

                fig.update_layout(
                    title='Feature Importance Comparison',
                    xaxis_title='Feature',
                    yaxis_title='Importance Score',
                    barmode='group'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Correlation between importance measures
                corr = np.corrcoef(comparison['Importance'], comparison['Normalized Coefficient'])[0, 1]
                st.write(f"Correlation between the two importance measures: {corr:.4f}")

            # Partial dependence plots
            st.subheader("Partial Dependence Plots")
            st.write(
                "These plots show how the model prediction changes when a feature varies, with all other features held constant.")

            selected_feature = st.selectbox("Select feature for partial dependence plot", X.columns)

            # Create partial dependence plot
            feature_idx = list(X.columns).index(selected_feature)

            # Get feature values
            values = np.linspace(X[selected_feature].min(), X[selected_feature].max(), 100)

            # Create a dataset where only the selected feature varies
            X_pdp = np.tile(X.mean().values, (len(values), 1))
            X_pdp[:, feature_idx] = values

            # Make predictions
            y_pdp = rf.predict(X_pdp)

            # Plot
            fig = px.line(
                x=values,
                y=y_pdp,
                title=f'Partial Dependence Plot for {selected_feature}',
                labels={'x': selected_feature, 'y': f'Predicted {target_var}'}
            )

            # Add scatter points for original data
            fig.add_trace(
                go.Scatter(
                    x=X[selected_feature],
                    y=y,
                    mode='markers',
                    marker=dict(size=5, color='red', opacity=0.3),
                    name='Original data'
                )
            )

            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload an Excel file to start the analysis.")

    # Example section
    st.header("Example Usage")
    st.write("""
    1. Upload your Excel file using the sidebar uploader
    2. Select the target variable (what you want to predict)
    3. Select the predictor variables (features to use for prediction)
    4. Explore the different tabs for comprehensive analysis:
        - Descriptive Statistics: View summary statistics and correlations
        - Pre-regression Tests: Check assumptions for linear regression
        - Linear Regression: Run the regression and view results
        - Dominance Analysis: Analyze relative importance using dominance analysis
        - Random Forest: Compare with Random Forest regression and feature importance
    """)