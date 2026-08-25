from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


st.set_page_config(
    page_title="Bank Marketing EDA",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


EXPECTED_COLUMNS = {
    "age": "Edad del cliente",
    "job": "Tipo de trabajo del cliente",
    "marital": "Estado civil",
    "education": "Nivel educativo",
    "default": "Tiene crédito en mora",
    "housing": "Tiene crédito hipotecario",
    "loan": "Tiene crédito personal",
    "contact": "Canal de comunicación usado",
    "month": "Último mes de contacto",
    "day_of_week": "Día del último contacto",
    "duration": "Duración del contacto en segundos",
    "campaign": "Número de contactos en la campaña actual",
    "pdays": "Días desde la última gestión",
    "previous": "Contactos previos antes de la campaña actual",
    "poutcome": "Resultado de la campaña anterior",
    "emp.var.rate": "Tasa de variación del empleo",
    "cons.price.idx": "Índice de precios al consumidor",
    "cons.conf.idx": "Índice de confianza del consumidor",
    "euribor3m": "Ratio de tipo de cambio medio a 3 meses",
    "nr.employed": "Número de empleados",
    "y": "Resultado final de aceptación de campaña",
}


@dataclass
class VariableGroups:
    numeric: list[str]
    categorical: list[str]
    datetime: list[str]
    boolean: list[str]


class DataAnalyzer:
    """Encapsula operaciones de EDA para el dataset BankMarketing."""

    def __init__(self, data: pd.DataFrame, target: str = "y") -> None:
        self.df = data.copy()
        self.target = target if target in data.columns else None

    def dataset_info(self) -> str:
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        return buffer.getvalue()

    def classify_variables(self) -> VariableGroups:
        numeric = self.df.select_dtypes(include=np.number).columns.tolist()
        datetime_cols = self.df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
        boolean = self.df.select_dtypes(include=["bool"]).columns.tolist()
        categorical = [
            col
            for col in self.df.columns
            if col not in numeric and col not in datetime_cols and col not in boolean
        ]
        return VariableGroups(
            numeric=numeric,
            categorical=categorical,
            datetime=datetime_cols,
            boolean=boolean,
        )

    def missing_values(self, include_unknown: bool = False) -> pd.DataFrame:
        rows = []
        for column in self.df.columns:
            null_count = int(self.df[column].isna().sum())
            unknown_count = 0
            if include_unknown and self.df[column].dtype == "object":
                unknown_count = int(self.df[column].astype(str).str.lower().eq("unknown").sum())
            total = null_count + unknown_count
            rows.append(
                {
                    "variable": column,
                    "nulos": null_count,
                    "unknown": unknown_count,
                    "total_observado": total,
                    "porcentaje": round(total / len(self.df) * 100, 2) if len(self.df) else 0,
                }
            )
        return pd.DataFrame(rows).sort_values("total_observado", ascending=False)

    def descriptive_statistics(self) -> pd.DataFrame:
        numeric = self.classify_variables().numeric
        if not numeric:
            return pd.DataFrame()
        description = self.df[numeric].describe().T
        description["median"] = self.df[numeric].median(numeric_only=True)
        modes = self.df[numeric].mode(dropna=True)
        description["mode"] = modes.iloc[0] if not modes.empty else np.nan
        description["range"] = description["max"] - description["min"]
        return description.round(3)

    def categorical_summary(self, column: str, top_n: int = 10) -> pd.DataFrame:
        summary = self.df[column].value_counts(dropna=False).head(top_n).reset_index()
        summary.columns = [column, "conteo"]
        summary["proporcion"] = (summary["conteo"] / len(self.df) * 100).round(2)
        return summary

    def group_comparison(self, numeric_col: str, category_col: str) -> pd.DataFrame:
        return (
            self.df.groupby(category_col, dropna=False)[numeric_col]
            .agg(conteo="count", media="mean", mediana="median", desviacion="std")
            .sort_values("conteo", ascending=False)
            .round(3)
            .reset_index()
        )

    def target_rate_by_category(self, column: str) -> pd.DataFrame:
        if not self.target:
            return pd.DataFrame()
        temp = self.df[[column, self.target]].dropna().copy()
        temp["_acepta"] = temp[self.target].astype(str).str.lower().eq("yes")
        return (
            temp.groupby(column)["_acepta"]
            .agg(conteo="count", tasa_aceptacion="mean")
            .sort_values("tasa_aceptacion", ascending=False)
            .assign(tasa_aceptacion=lambda d: (d["tasa_aceptacion"] * 100).round(2))
            .reset_index()
        )

    def plot_histograms(self, columns: Iterable[str], bins: int, kde: bool) -> plt.Figure:
        columns = list(columns)
        fig, axes = plt.subplots(len(columns), 1, figsize=(9, max(4, 3.2 * len(columns))))
        if len(columns) == 1:
            axes = [axes]
        for axis, column in zip(axes, columns):
            sns.histplot(data=self.df, x=column, bins=bins, kde=kde, color="#2f6fed", ax=axis)
            axis.set_title(f"Distribución de {column}", loc="left", fontweight="bold")
            axis.set_xlabel(column)
            axis.set_ylabel("Frecuencia")
        fig.tight_layout()
        return fig

    def plot_categorical(self, column: str, top_n: int) -> plt.Figure:
        data = self.categorical_summary(column, top_n=top_n)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        sns.barplot(data=data, y=column, x="conteo", color="#16a085", ax=ax)
        ax.set_title(f"Top {top_n} categorías de {column}", loc="left", fontweight="bold")
        ax.set_xlabel("Conteo")
        ax.set_ylabel(column)
        fig.tight_layout()
        return fig

    def plot_boxplot(self, numeric_col: str, category_col: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        sns.boxplot(data=self.df, x=category_col, y=numeric_col, color="#87bfff", ax=ax)
        ax.set_title(f"{numeric_col} por {category_col}", loc="left", fontweight="bold")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        return fig

    def plot_crosstab_heatmap(self, row_col: str, column_col: str) -> plt.Figure:
        table = pd.crosstab(self.df[row_col], self.df[column_col], normalize="index") * 100
        fig, ax = plt.subplots(figsize=(9, max(4.5, len(table) * 0.45)))
        sns.heatmap(table, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5, ax=ax)
        ax.set_title(f"Proporción de {column_col} según {row_col}", loc="left", fontweight="bold")
        ax.set_xlabel(column_col)
        ax.set_ylabel(row_col)
        fig.tight_layout()
        return fig


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #18202f;
            --muted: #5f6b7a;
            --line: #dce3ec;
            --panel: #ffffff;
            --blue: #2f6fed;
            --green: #16a085;
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }
        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
        }
        .section-note {
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.55;
            margin-bottom: 1rem;
        }
        .status-box {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px 18px;
            background: #fbfcff;
        }
        .small-caption {
            color: var(--muted);
            font-size: 0.88rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    encodings = ["utf-8", "latin-1"]
    separators = [",", ";", "\t"]
    last_error: Exception | None = None
    for encoding in encodings:
        for separator in separators:
            try:
                data = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding, sep=separator)
                if data.shape[1] > 1:
                    return data
            except Exception as error:
                last_error = error
    try:
        return pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1", sep=None, engine="python")
    except Exception as error:
        raise ValueError(f"No se pudo leer el CSV. Último error: {last_error or error}") from error


def get_uploaded_data() -> pd.DataFrame | None:
    return st.session_state.get("bankmarketing_data")


def set_uploaded_data(data: pd.DataFrame) -> None:
    st.session_state["bankmarketing_data"] = data


def metric_row(data: pd.DataFrame, analyzer: DataAnalyzer) -> None:
    groups = analyzer.classify_variables()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", f"{data.shape[0]:,}")
    c2.metric("Columnas", f"{data.shape[1]:,}")
    c3.metric("Variables numéricas", len(groups.numeric))
    c4.metric("Variables categóricas", len(groups.categorical))


def show_home(author: str, course: str, year: str) -> None:
    st.title("Análisis Exploratorio del Dataset BankMarketing")
    st.markdown(
        """
        <p class="section-note">
        Aplicación interactiva en Streamlit para analizar la última campaña de marketing de una
        institución financiera. El objetivo es identificar patrones de comportamiento, relaciones
        entre variables y oportunidades de decisión basadas en datos, sin construir modelos
        predictivos.
        </p>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1.1, 1, 1])
    c1.metric("Autor", author)
    c2.metric("Curso", course)
    c3.metric("Año", year)

    st.subheader("Contexto del dataset")
    st.markdown(
        """
        El archivo `BankMarketing.csv` contiene información de clientes contactados durante una
        campaña comercial. La variable objetivo `y` indica si el cliente aceptó la campaña.
        El análisis busca entender qué variables podrían explicar cambios en la efectividad,
        que pasó de 12% a 8% durante los últimos 6 meses según el caso.
        """
    )

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Tecnologías utilizadas")
        st.dataframe(
            pd.DataFrame(
                {
                    "Tecnología": ["Python", "Pandas", "NumPy", "Streamlit", "Matplotlib", "Seaborn"],
                    "Uso en el proyecto": [
                        "Lenguaje base de desarrollo",
                        "Manipulación y análisis tabular",
                        "Cálculos numéricos y clasificación de tipos",
                        "Interfaz interactiva y navegación",
                        "Construcción de gráficos estadísticos",
                        "Visualización exploratoria profesional",
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.subheader("Diccionario de variables")
        dictionary = pd.DataFrame(
            EXPECTED_COLUMNS.items(),
            columns=["Variable", "Descripción"],
        )
        st.dataframe(dictionary, use_container_width=True, hide_index=True, height=430)


def show_upload() -> None:
    st.title("Carga del Dataset")
    st.markdown(
        """
        <p class="section-note">
        Antes de ejecutar el análisis, carga el archivo `BankMarketing.csv`. La aplicación valida
        el archivo, muestra una vista previa y confirma sus dimensiones.
        </p>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Carga el archivo CSV para habilitar los módulos de análisis.")
        return

    try:
        data = load_csv_from_bytes(uploaded_file.getvalue())
        set_uploaded_data(data)
    except ValueError as error:
        st.error(str(error))
        return

    analyzer = DataAnalyzer(data)
    st.success("Dataset cargado correctamente.")
    metric_row(data, analyzer)

    preview_rows = st.slider("Filas para vista previa", min_value=5, max_value=30, value=10, step=5)
    st.dataframe(data.head(preview_rows), use_container_width=True)

    expected = set(EXPECTED_COLUMNS)
    present = set(data.columns)
    missing_expected = sorted(expected - present)
    extra_columns = sorted(present - expected)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Variables esperadas no encontradas")
        if missing_expected:
            st.warning(", ".join(missing_expected))
        else:
            st.success("El dataset contiene todas las variables esperadas del caso.")
    with c2:
        st.subheader("Variables adicionales")
        if extra_columns:
            st.write(", ".join(extra_columns))
        else:
            st.write("No se detectaron variables adicionales.")


def require_data() -> pd.DataFrame | None:
    data = get_uploaded_data()
    if data is None:
        st.warning("Primero carga `BankMarketing.csv` en el módulo Carga del dataset.")
    return data


def show_eda() -> None:
    data = require_data()
    if data is None:
        return

    analyzer = DataAnalyzer(data)
    groups = analyzer.classify_variables()

    st.title("Análisis Exploratorio de Datos")
    metric_row(data, analyzer)

    tabs = st.tabs(
        [
            "1. General",
            "2. Variables",
            "3. Descriptivas",
            "4. Faltantes",
            "5. Numéricas",
            "6. Categóricas",
            "7. Num vs Cat",
            "8. Cat vs Cat",
            "9. Dinámico",
            "10. Hallazgos",
        ]
    )

    with tabs[0]:
        st.subheader("Información general del dataset")
        st.markdown(
            "Se revisa la estructura general, tipos de datos, memoria utilizada y conteo de registros por variable."
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            st.code(analyzer.dataset_info(), language="text")
        with c2:
            type_table = (
                pd.DataFrame(data.dtypes.astype(str), columns=["tipo_dato"])
                .reset_index()
                .rename(columns={"index": "variable"})
            )
            type_table["nulos"] = data.isna().sum().values
            st.dataframe(type_table, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("Clasificación de variables")
        st.markdown(
            "La clasificación se realiza mediante una función encapsulada en la clase `DataAnalyzer`."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Numéricas", len(groups.numeric))
        c2.metric("Categóricas", len(groups.categorical))
        c3.metric("Booleanas", len(groups.boolean))
        c4.metric("Fecha / tiempo", len(groups.datetime))

        left, right = st.columns(2)
        with left:
            st.write("Variables numéricas")
            st.dataframe(pd.DataFrame({"variable": groups.numeric}), use_container_width=True, hide_index=True)
        with right:
            st.write("Variables categóricas")
            st.dataframe(
                pd.DataFrame({"variable": groups.categorical}),
                use_container_width=True,
                hide_index=True,
            )

    with tabs[2]:
        st.subheader("Estadísticas descriptivas")
        st.markdown(
            "Se comparan media, mediana, desviación, mínimos, máximos y rango para entender centro y dispersión."
        )
        stats = analyzer.descriptive_statistics()
        if stats.empty:
            st.info("No se detectaron variables numéricas para calcular estadísticas descriptivas.")
        else:
            st.dataframe(stats, use_container_width=True)
            col = st.selectbox("Variable numérica para interpretación", groups.numeric, key="desc_col")
            mean_value = data[col].mean()
            median_value = data[col].median()
            std_value = data[col].std()
            st.markdown(
                f"""
                **Interpretación básica:** `{col}` tiene media de `{mean_value:,.2f}`,
                mediana de `{median_value:,.2f}` y desviación estándar de `{std_value:,.2f}`.
                Cuando media y mediana se alejan, puede existir asimetría o presencia de valores extremos.
                """
            )

    with tabs[3]:
        st.subheader("Análisis de valores faltantes")
        include_unknown = st.checkbox(
            "Tratar la categoría 'unknown' como ausencia informativa",
            value=True,
            key="include_unknown",
        )
        missing = analyzer.missing_values(include_unknown=include_unknown)
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.dataframe(missing, use_container_width=True, hide_index=True)
        with c2:
            visible = missing[missing["total_observado"] > 0]
            if visible.empty:
                st.success("No se detectaron valores faltantes relevantes bajo el criterio seleccionado.")
            else:
                fig, ax = plt.subplots(figsize=(8, 4.5))
                sns.barplot(data=visible.head(12), y="variable", x="porcentaje", color="#d94f45", ax=ax)
                ax.set_title("Porcentaje de faltantes por variable", loc="left", fontweight="bold")
                ax.set_xlabel("% de registros")
                ax.set_ylabel("Variable")
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
                st.markdown(
                    "Las variables con mayor porcentaje de ausencia requieren revisión antes de tomar decisiones operativas."
                )

    with tabs[4]:
        st.subheader("Distribución de variables numéricas")
        if not groups.numeric:
            st.info("No se detectaron variables numéricas.")
        else:
            selected = st.multiselect(
                "Variables numéricas",
                groups.numeric,
                default=groups.numeric[: min(3, len(groups.numeric))],
                key="hist_cols",
            )
            bins = st.slider("Número de bins", min_value=5, max_value=80, value=30, step=5)
            kde = st.checkbox("Mostrar curva KDE", value=False)
            if selected:
                st.pyplot(analyzer.plot_histograms(selected, bins=bins, kde=kde), clear_figure=True)
                st.markdown(
                    "La distribución permite detectar asimetrías, concentración de valores y posibles outliers."
                )
            else:
                st.info("Selecciona al menos una variable numérica.")

    with tabs[5]:
        st.subheader("Análisis de variables categóricas")
        if not groups.categorical:
            st.info("No se detectaron variables categóricas.")
        else:
            cat_col = st.selectbox("Variable categórica", groups.categorical, key="cat_col")
            top_n = st.slider("Categorías a mostrar", min_value=3, max_value=20, value=10, step=1)
            c1, c2 = st.columns([1.25, 1])
            with c1:
                st.pyplot(analyzer.plot_categorical(cat_col, top_n=top_n), clear_figure=True)
            with c2:
                summary = analyzer.categorical_summary(cat_col, top_n=top_n)
                st.dataframe(summary, use_container_width=True, hide_index=True)
                st.markdown(
                    "Los conteos y proporciones ayudan a identificar segmentos dominantes o categorías poco representadas."
                )

    with tabs[6]:
        st.subheader("Análisis bivariado: numérico vs categórico")
        if not groups.numeric or not groups.categorical:
            st.info("Se requieren variables numéricas y categóricas para este análisis.")
        else:
            default_category = "y" if "y" in groups.categorical else groups.categorical[0]
            category_index = groups.categorical.index(default_category)
            num_col = st.selectbox(
                "Variable numérica",
                groups.numeric,
                index=groups.numeric.index("duration") if "duration" in groups.numeric else 0,
                key="num_vs_cat_num",
            )
            cat_col = st.selectbox(
                "Variable categórica",
                groups.categorical,
                index=category_index,
                key="num_vs_cat_cat",
            )
            c1, c2 = st.columns([1.25, 1])
            with c1:
                st.pyplot(analyzer.plot_boxplot(num_col, cat_col), clear_figure=True)
            with c2:
                st.dataframe(
                    analyzer.group_comparison(num_col, cat_col),
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown(
                    "La comparación de grupos muestra diferencias de media, mediana y dispersión entre categorías."
                )

    with tabs[7]:
        st.subheader("Análisis bivariado: categórico vs categórico")
        if len(groups.categorical) < 2:
            st.info("Se requieren al menos dos variables categóricas.")
        else:
            row_default = "education" if "education" in groups.categorical else groups.categorical[0]
            col_default = "y" if "y" in groups.categorical else groups.categorical[1]
            row_col = st.selectbox(
                "Variable de filas",
                groups.categorical,
                index=groups.categorical.index(row_default),
                key="row_cat",
            )
            col_col = st.selectbox(
                "Variable de columnas",
                groups.categorical,
                index=groups.categorical.index(col_default),
                key="col_cat",
            )
            c1, c2 = st.columns([1.1, 1])
            with c1:
                st.pyplot(analyzer.plot_crosstab_heatmap(row_col, col_col), clear_figure=True)
            with c2:
                table = pd.crosstab(data[row_col], data[col_col], margins=True)
                st.dataframe(table, use_container_width=True)
                st.markdown(
                    "La tabla cruzada permite evaluar asociación visual entre categorías, por ejemplo educación o canal frente a aceptación."
                )

    with tabs[8]:
        st.subheader("Análisis basado en parámetros seleccionados")
        selected_cols = st.multiselect(
            "Columnas a analizar",
            data.columns.tolist(),
            default=[col for col in ["age", "job", "duration", "campaign", "y"] if col in data.columns],
            key="dynamic_cols",
        )
        if not selected_cols:
            st.info("Selecciona columnas para construir un análisis dinámico.")
        else:
            filtered = data[selected_cols].copy()
            filter_col = st.selectbox("Variable para filtrar", ["Sin filtro"] + selected_cols, key="filter_col")
            if filter_col != "Sin filtro":
                values = data[filter_col].dropna().astype(str).unique().tolist()
                chosen = st.multiselect(
                    "Valores permitidos",
                    sorted(values)[:200],
                    default=sorted(values)[: min(3, len(values))],
                    key="filter_values",
                )
                if chosen:
                    filtered = filtered[data[filter_col].astype(str).isin(chosen)]
            max_rows = st.slider("Máximo de filas visibles", min_value=5, max_value=100, value=25, step=5)
            c1, c2 = st.columns([1, 1])
            with c1:
                st.dataframe(filtered.head(max_rows), use_container_width=True, hide_index=True)
            with c2:
                numeric_selected = [col for col in selected_cols if col in groups.numeric]
                cat_selected = [col for col in selected_cols if col in groups.categorical]
                if numeric_selected:
                    st.write("Resumen numérico dinámico")
                    st.dataframe(filtered[numeric_selected].describe().T.round(3), use_container_width=True)
                if cat_selected:
                    st.write("Conteo categórico dinámico")
                    selected_cat = st.selectbox("Categoría para resumen", cat_selected, key="dynamic_cat")
                    cat_counts = (
                        filtered[selected_cat]
                        .value_counts(dropna=False)
                        .rename_axis(selected_cat)
                        .reset_index(name="conteo")
                    )
                    st.dataframe(
                        cat_counts,
                        use_container_width=True,
                        hide_index=True,
                    )

    with tabs[9]:
        st.subheader("Hallazgos clave")
        findings = build_findings(data, analyzer)
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros duplicados", int(data.duplicated().sum()))
        c2.metric("Variables con faltantes", int((data.isna().sum() > 0).sum()))
        if analyzer.target:
            acceptance = data[analyzer.target].astype(str).str.lower().eq("yes").mean() * 100
            c3.metric("Tasa de aceptación", f"{acceptance:.2f}%")
        else:
            c3.metric("Variable objetivo", "No detectada")

        for finding in findings:
            st.markdown(f"- {finding}")

        if analyzer.target and "job" in data.columns:
            rate = analyzer.target_rate_by_category("job").head(12)
            fig, ax = plt.subplots(figsize=(9, 4.8))
            sns.barplot(data=rate, y="job", x="tasa_aceptacion", color="#2f6fed", ax=ax)
            ax.set_title("Tasa de aceptación por tipo de trabajo", loc="left", fontweight="bold")
            ax.set_xlabel("% aceptación")
            ax.set_ylabel("Trabajo")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)


def build_findings(data: pd.DataFrame, analyzer: DataAnalyzer) -> list[str]:
    findings = [
        f"El dataset contiene {data.shape[0]:,} registros y {data.shape[1]:,} variables disponibles para el análisis exploratorio.",
        f"Se identificaron {len(analyzer.classify_variables().numeric)} variables numéricas y {len(analyzer.classify_variables().categorical)} variables categóricas.",
    ]
    missing = analyzer.missing_values(include_unknown=True)
    top_missing = missing[missing["total_observado"] > 0].head(1)
    if not top_missing.empty:
        row = top_missing.iloc[0]
        findings.append(
            f"La variable con mayor ausencia informativa es `{row['variable']}`, con {row['porcentaje']:.2f}% de registros nulos o `unknown`."
        )
    else:
        findings.append("No se observan valores faltantes relevantes al considerar nulos y la categoría `unknown`.")

    if analyzer.target:
        target_counts = data[analyzer.target].astype(str).str.lower().value_counts(normalize=True) * 100
        yes_rate = target_counts.get("yes", 0)
        findings.append(
            f"La aceptación de campaña representa aproximadamente {yes_rate:.2f}% de los registros cargados."
        )
        for candidate in ["contact", "education", "poutcome", "job"]:
            if candidate in data.columns:
                top_rate = analyzer.target_rate_by_category(candidate)
                if not top_rate.empty:
                    best = top_rate.iloc[0]
                    findings.append(
                        f"En `{candidate}`, la categoría con mayor tasa de aceptación visible es `{best[candidate]}` con {best['tasa_aceptacion']:.2f}%."
                    )
                    break
    else:
        findings.append("No se detectó la variable objetivo `y`; los hallazgos se concentran en estructura y distribución.")
    return findings[:6]


def show_conclusions() -> None:
    data = require_data()
    if data is None:
        return

    analyzer = DataAnalyzer(data)
    st.title("Conclusiones Finales")
    st.markdown(
        """
        <p class="section-note">
        Conclusiones orientadas a toma de decisiones comerciales y calidad de análisis,
        derivadas del EDA ejecutado sobre el dataset cargado.
        </p>
        """,
        unsafe_allow_html=True,
    )

    conclusions = build_conclusions(data, analyzer)
    for index, conclusion in enumerate(conclusions, start=1):
        st.markdown(f"**{index}.** {conclusion}")

    st.subheader("Comentario final para documento PDF")
    st.markdown(
        """
        Los aprendizajes de este módulo fortalecen la capacidad de transformar datos operativos
        en evidencia útil para la toma de decisiones. En el trabajo diario, estas habilidades
        permiten validar supuestos, comunicar hallazgos y priorizar acciones comerciales con mayor
        criterio técnico. Los próximos desafíos consisten en profundizar en automatización,
        despliegue de aplicaciones analíticas y buenas prácticas de documentación. La metodología
        práctica del curso contribuyó a mejorar la forma de aprender tecnología al integrar teoría,
        análisis y construcción de productos funcionales.
        """
    )


def build_conclusions(data: pd.DataFrame, analyzer: DataAnalyzer) -> list[str]:
    groups = analyzer.classify_variables()
    conclusions = [
        "El análisis exploratorio permite entender la composición de clientes antes de tomar decisiones sobre campañas, segmentación o priorización comercial.",
        "La clasificación de variables numéricas y categóricas facilita elegir visualizaciones adecuadas y evita interpretaciones erróneas entre tipos de datos distintos.",
        "Las estadísticas descriptivas muestran el comportamiento central y la dispersión de variables como edad, duración, contactos de campaña e indicadores económicos.",
        "El análisis de faltantes y de la categoría `unknown` es clave porque la ausencia informativa puede afectar la lectura de segmentos comerciales.",
        "Las comparaciones bivariadas ayudan a identificar diferencias entre grupos, como aceptación por canal, educación, trabajo o resultado de campañas previas.",
    ]
    if analyzer.target:
        yes_rate = data[analyzer.target].astype(str).str.lower().eq("yes").mean() * 100
        conclusions[0] = (
            f"La tasa de aceptación observada es {yes_rate:.2f}%, por lo que el análisis exploratorio "
            "ayuda a identificar segmentos y condiciones asociadas a mejores respuestas comerciales."
        )
    if "duration" in groups.numeric:
        conclusions[2] = (
            "La duración del contacto debe analizarse con cuidado porque suele concentrar alta variabilidad; "
            "su comparación con `y` permite observar diferencias entre clientes que aceptan y no aceptan."
        )
    return conclusions


def sidebar() -> str:
    with st.sidebar:
        st.header("BankMarketing EDA")
        page = st.radio(
            "Menú principal",
            ["Home", "Carga del dataset", "EDA", "Conclusiones"],
            label_visibility="collapsed",
        )
        st.divider()
        st.subheader("Datos del autor")
        st.text_input("Nombre completo", value="Nombre del estudiante", key="author_name")
        st.text_input("Curso / Especialización", value="Especialización Python for Analytics", key="course_name")
        st.text_input("Año", value="2026", key="project_year")
        st.divider()
        data = get_uploaded_data()
        if data is None:
            st.caption("Estado: dataset pendiente de carga.")
        else:
            st.caption(f"Estado: {data.shape[0]:,} filas cargadas.")
    return page


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    inject_styles()
    page = sidebar()

    author = st.session_state.get("author_name", "Nombre del estudiante")
    course = st.session_state.get("course_name", "Especialización Python for Analytics")
    year = st.session_state.get("project_year", "2026")

    if page == "Home":
        show_home(author, course, year)
    elif page == "Carga del dataset":
        show_upload()
    elif page == "EDA":
        show_eda()
    elif page == "Conclusiones":
        show_conclusions()


if __name__ == "__main__":
    main()
