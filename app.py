
import gradio as gr
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np 
 
# --- File Paths ---
MODEL_PATH = 'car_price_predictoR.joblib'
DATA_PATH = 'cars24_CLeaned_dataset.csv'

# --- Load Model & Data ---
try:
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Required file not found: {e}. Ensure '{MODEL_PATH}' and '{DATA_PATH}' exist.")

# --- Dropdown values ---
unique_brands = sorted(df['Car_Brand'].unique().tolist())
unique_models = sorted(df['Car_Model'].unique().tolist())
unique_fuel_types = sorted(df['Fuel Type'].unique().tolist())
unique_transmission_types = sorted(df['Transmission Type'].unique().tolist())
unique_ownerships = sorted(df['Ownership'].unique().tolist())

# --- Prediction Function ---
def predict_price(car_age, car_brand, car_model, km_driven, fuel_type, transmission_type, ownership):
    input_df = pd.DataFrame([[car_age, car_brand, car_model, km_driven,
                              fuel_type, transmission_type, ownership]],
                            columns=['Car_Age', 'Car_Brand', 'Car_Model',
                                     'KM Driven', 'Fuel Type', 'Transmission Type', 'Ownership'])
    try:
        prediction = model.predict(input_df)[0]
        return f"💰 Predicted Price: {prediction:.2f} Lakhs"
    except Exception as e:
        return f"Error: {e}"

def update_models(car_brand):
    filtered_df = df[df['Car_Brand'] == car_brand]
    return gr.Dropdown(choices=sorted(filtered_df['Car_Model'].unique().tolist()), label="Car Model")

# --- Visualization Functions ---
def create_price_distribution_plot():
    return px.histogram(df, x='Price(in Lakhs)', nbins=30, title='Distribution of Car Prices', template='plotly_white')

def create_kms_driven_scatter_plot():
    return px.scatter(df, x='KM Driven', y='Price(in Lakhs)', color='Fuel Type',
                      title='Price vs. KM Driven', template='plotly_white')

def create_fuel_type_box_plot():
    return px.box(df, x='Fuel Type', y='Price(in Lakhs)', color='Fuel Type',
                  title='Price Distribution by Fuel Type', template='plotly_white')

def create_brand_price_box_plot():
    return px.box(df, x='Car_Brand', y='Price(in Lakhs)', color='Car_Brand',
                  title='Price Distribution by Car Brand', template='plotly_white')

def create_transmission_price_box_plot():
    return px.box(df, x='Transmission Type', y='Price(in Lakhs)', color='Transmission Type',
                  title='Price Distribution by Transmission Type', template='plotly_white')

def create_ownership_price_box_plot():
    return px.box(df, x='Ownership', y='Price(in Lakhs)', color='Ownership',
                  title='Price Distribution by Ownership', template='plotly_white')

def create_cars_by_brand_plot():
    counts = df['Car_Brand'].value_counts().reset_index()
    counts.columns = ['Car_Brand', 'Count']
    return px.bar(counts, x='Car_Brand', y='Count', title='Count of Cars by Brand', template='plotly_white')

def create_cars_by_fuel_type_plot():
    counts = df['Fuel Type'].value_counts().reset_index()
    counts.columns = ['Fuel Type', 'Count']
    return px.pie(counts, names='Fuel Type', values='Count', title='Distribution of Fuel Types', template='plotly_white')

def create_cars_by_transmission_plot():
    counts = df['Transmission Type'].value_counts().reset_index()
    counts.columns = ['Transmission Type', 'Count']
    return px.pie(counts, names='Transmission Type', values='Count', title='Distribution of Transmission Types', template='plotly_white')

def create_price_vs_age_scatter():
    return px.scatter(df, x='Car_Age', y='Price(in Lakhs)', color='Fuel Type', title='Price vs. Car Age', template='plotly_white')

def create_correlation_heatmap():
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr(numeric_only=True)
    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                     x=corr_matrix.columns,
                                     y=corr_matrix.columns,
                                     colorscale='Viridis'))
    fig.update_layout(title='Correlation Heatmap')
    return fig

def create_brand_price_violin_plot():
    return px.violin(df, x='Car_Brand', y='Price(in Lakhs)', color='Car_Brand',
                     title='Price Distribution by Car Brand (Violin Plot)', template='plotly_white')

def create_kms_driven_density_plot():
    return px.density_contour(df, x='KM Driven', y='Price(in Lakhs)', color='Fuel Type',
                              title='Density of Price vs. KM Driven', template='plotly_white')

def create_car_age_box_plot():
    return px.box(df, x='Car_Age', y='Price(in Lakhs)', color='Fuel Type',
                  title='Price Distribution by Car Age', template='plotly_white')

def create_ownership_transmission_plot():
    return px.histogram(df, x='Ownership', color='Transmission Type', barmode='group',
                        title='Count of Cars by Ownership & Transmission', template='plotly_white')

# --- Gradio App with Sidebar ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📌 Navigation")
            page_selector = gr.Radio(
                choices=["About Me", "Project Overview", "Prediction App", "EDA & Visualizations"],
                value="About Me",
                label="Select Page",
                interactive=True
            )

        with gr.Column(scale=4) as main_content:
            about_md = gr.Markdown(visible=True)
            overview_md = gr.Markdown(visible=False)

            # Prediction UI
            with gr.Group(visible=False) as pred_ui:
                gr.Markdown("### 🚗 Car Price Prediction Tool")
                with gr.Row():
                    with gr.Column():
                        car_age = gr.Number(label="Car Age")
                        km_driven = gr.Number(label="KM Driven")
                        fuel_type = gr.Dropdown(choices=unique_fuel_types, label="Fuel Type")
                        transmission = gr.Dropdown(choices=unique_transmission_types, label="Transmission Type")
                        ownership = gr.Dropdown(choices=unique_ownerships, label="Ownership")
                    with gr.Column():
                        brand = gr.Dropdown(choices=unique_brands, label="Car Brand")
                        model_dd = gr.Dropdown(choices=[], label="Car Model")

                brand.change(fn=update_models, inputs=brand, outputs=model_dd)

                output = gr.Textbox(label="Predicted Price")
                predict_btn = gr.Button("🔮 Predict")
                predict_btn.click(predict_price,
                                  inputs=[car_age, brand, model_dd, km_driven, fuel_type, transmission, ownership],
                                  outputs=output)

            # EDA UI
            with gr.Group(visible=False) as eda_ui:
                gr.Markdown("### 📊 Exploratory Data Analysis")
                plots = [
                    create_price_distribution_plot(),
                    create_kms_driven_scatter_plot(),
                    create_fuel_type_box_plot(),
                    create_brand_price_box_plot(),
                    create_transmission_price_box_plot(),
                    create_ownership_price_box_plot(),
                    create_cars_by_brand_plot(),
                    create_cars_by_fuel_type_plot(),
                    create_cars_by_transmission_plot(),
                    create_price_vs_age_scatter(),
                    create_correlation_heatmap(),
                    create_brand_price_violin_plot(),
                    create_kms_driven_density_plot(),
                    create_car_age_box_plot(),
                    create_ownership_transmission_plot()
                ]
                for p in plots:
                    gr.Plot(p)

    # --- Function to switch pages ---
    def update_page(page):
        if page == "About Me":
            about_md_content = """
            ## 👋 About Me
            Hello! I'm a passionate data scientist and ML enthusiast with experience in building end-to-end ML applications.

            ### Connect with Me
            * [LinkedIn](https://www.linkedin.com/)  
            * [GitHub](https://github.com/)  
            * [Streamlit](https://share.streamlit.io/)  
            * [Hugging Face](https://huggingface.co/)
            """
            return gr.update(visible=True, value=about_md_content), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        elif page == "Project Overview":
            overview_md_content = """
            ## 📊 Project Overview
            This project predicts used car prices using a machine learning model.  

            **Objectives:**  
            1. Data Preprocessing & Exploration  
            2. Model Development  
            3. Interactive Deployment  
            4. Comprehensive Insights with EDA
            """
            return gr.update(visible=False), gr.update(visible=True, value=overview_md_content), gr.update(visible=False), gr.update(visible=False)
        elif page == "Prediction App":
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        elif page == "EDA & Visualizations":
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    page_selector.change(update_page, inputs=page_selector,
                         outputs=[about_md, overview_md, pred_ui, eda_ui])

if __name__ == "__main__":
    demo.launch()
