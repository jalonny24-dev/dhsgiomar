import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris

# Configuración de la página
st.set_page_config(
    page_title="Iris Dashboard",
    page_icon="🌸",
    layout="wide"
)

# Cargar datos
iris = load_iris(as_frame=True)

df = iris.frame.copy()
df["species"] = df["target"].map(
    {i: name for i, name in enumerate(iris.target_names)}
)

# Título
st.title("🌸 Iris Dataset Dashboard")
st.markdown(
    "Dashboard interactivo para explorar el conjunto de datos Iris."
)

# Sidebar
st.sidebar.header("Filtros")

species = st.sidebar.multiselect(
    "Selecciona especies",
    options=df["species"].unique(),
    default=df["species"].unique()
)

filtered_df = df[df["species"].isin(species)]

# KPIs
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Registros", len(filtered_df))

with c2:
    st.metric(
        "Sepal Length Prom.",
        f"{filtered_df['sepal length (cm)'].mean():.2f}"
    )

with c3:
    st.metric(
        "Petal Length Prom.",
        f"{filtered_df['petal length (cm)'].mean():.2f}"
    )

with c4:
    st.metric(
        "Especies",
        filtered_df["species"].nunique()
    )

st.divider()

# Gráficos
col1, col2 = st.columns(2)

with col1:
    fig_scatter = px.scatter(
        filtered_df,
        x="sepal length (cm)",
        y="petal length (cm)",
        color="species",
        color_discrete_sequence=px.colors.sequential.Viridis,
        title="Relación entre Sepal Length y Petal Length"
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )

with col2:
    fig_box = px.box(
        filtered_df,
        x="species",
        y="petal width (cm)",
        color="species",
        color_discrete_sequence=px.colors.sequential.Viridis,
        title="Distribución de Petal Width"
    )

    st.plotly_chart(
        fig_box,
        use_container_width=True
    )

# Histograma
fig_hist = px.histogram(
    filtered_df,
    x="sepal width (cm)",
    color="species",
    barmode="overlay",
    color_discrete_sequence=px.colors.sequential.Viridis,
    title="Distribución de Sepal Width"
)

st.plotly_chart(
    fig_hist,
    use_container_width=True
)

# Tabla de datos
st.subheader("Datos")

st.dataframe(
    filtered_df,
    use_container_width=True,
    height=400
)
