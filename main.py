import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dateutil
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import geopandas as gpd

st.title("Analiza Amenințărilor Cibernetice Globale (2015–2024)")

st.sidebar.header("Meniu")
section = st.sidebar.radio("Selectați o secțiune:",
                           ["Încărcare Fișier", "Vizualizare Date", "Filtrare & Analiză", "Prelucrare & Grafice",
                            "Codificare Date", "Scalare Date", "Outlieri", "Hartă Geospațială"])

data = None

# 1. Încărcare fișier
if section == "Încărcare Fișier":
    st.header(" Încărcați un fișier CSV")
    uploaded_file = st.file_uploader("Selectați fișierul", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        if "date" in data.columns:
            try:
                data["date"] = data["date"].apply(dateutil.parser.parse, dayfirst=True)
            except Exception as e:
                st.warning(f" Eroare la conversia coloanei 'date': {e}")

        st.session_state["data"] = data
        st.success(" Fișier încărcat cu succes!")
        st.write("Primele 5 rânduri:")
        st.dataframe(data.head())

# 2. Vizualizare generală
elif section == "Vizualizare Date":
    st.header(" Vizualizare Generală a Datelor")
    if "data" in st.session_state:
        data = st.session_state["data"]
        st.write(" Dimensiunea datasetului:", data.shape)

        if st.checkbox("Afișează primele 5 coloane și rânduri"):
            st.dataframe(data.iloc[:, :5].head())

        st.subheader(" Set complet de date:")
        st.dataframe(data)

        # Analiza valorilor lipsă
        st.subheader(" Analiza valorilor lipsă")
        missing_vals = data.isnull().sum()
        missing_percent = (missing_vals / len(data)) * 100
        missing_df = pd.DataFrame({
            'Missing Values': missing_vals,
            'Percentage': missing_percent
        })
        missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Percentage', ascending=False)

        if not missing_df.empty:
            fig, ax = plt.subplots()
            sns.barplot(x=missing_df['Percentage'], y=missing_df.index, color="orange", ax=ax)
            ax.set_title('Procentul valorilor lipsă per coloană')
            ax.set_xlabel('Procent (%)')
            ax.set_ylabel('Coloană')
            st.pyplot(fig)
        else:
            st.success(" Nu există valori lipsă în dataset.")

# 3. Filtrare și analiză
elif section == "Filtrare & Analiză":
    st.header(" Filtrare & Căutare")
    if "data" in st.session_state:
        data = st.session_state["data"]

        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            col_to_filter = st.selectbox("Coloană numerică pentru filtrare:", numeric_cols)
            min_val, max_val = data[col_to_filter].min(), data[col_to_filter].max()
            val_range = st.slider("Interval de valori:", float(min_val), float(max_val),
                                  (float(min_val), float(max_val)))
            filtered_data = data[(data[col_to_filter] >= val_range[0]) & (data[col_to_filter] <= val_range[1])]

            st.write(" Rezultate filtrate:")
            st.dataframe(filtered_data)

        if "Country" in data.columns:
            country = st.text_input("Introduceți o țară pentru căutare:")
            if country:
                filtered = data[data["Country"].str.contains(country, case=False, na=False)]
                st.write(f" Rezultate pentru: {country}")
                st.dataframe(filtered)

        if "Year" in data.columns:
            st.write(" Număr înregistrări pe ani:")
            st.bar_chart(data["Year"].value_counts().sort_index())
    else:
        st.warning("⚠ Încărcați un fișier mai întâi.")

# 4. Prelucrare + grafice
elif section == "Prelucrare & Grafice":
    st.header(" Prelucrare și Vizualizare Grafică")
    if "data" in st.session_state:
        data = st.session_state["data"]
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

        if numeric_cols:
            col_to_analyze = st.selectbox("Selectați o coloană numerică:", numeric_cols)
            st.subheader(f" Statistici descriptive pentru `{col_to_analyze}`:")
            st.write(data[col_to_analyze].describe())

            # Histogramă
            fig1, ax1 = plt.subplots()
            ax1.hist(data[col_to_analyze].dropna(), bins=20)
            ax1.set_title(f"Distribuția valorilor pentru {col_to_analyze}")
            st.pyplot(fig1)

            # Medie pe ani + regresie
            if "Year" in data.columns:
                grouped = data.groupby("Year", as_index=False)[col_to_analyze].mean()
                X = grouped[["Year"]]
                y = grouped[col_to_analyze]

                model = LinearRegression()
                model.fit(X, y)
                future_years = pd.DataFrame({"Year": list(range(X["Year"].min(), X["Year"].max() + 3))})
                future_preds = model.predict(future_years)

                fig2, ax2 = plt.subplots()
                ax2.plot(X["Year"], y, marker="o", label="Valori reale")
                ax2.plot(future_years["Year"], future_preds, linestyle="--", marker="x", label="Predicție")
                ax2.set_title(f" Evoluție și Predicție: {col_to_analyze} în funcție de Year")
                ax2.set_xlabel("An")
                ax2.set_ylabel(col_to_analyze)
                ax2.legend()
                st.pyplot(fig2)

                st.info(f" Predicție pentru anul {future_years['Year'].iloc[-1]}: **{future_preds[-1]:.2f}**")

        if "Attack Type" in data.columns:
            st.subheader(" Grupare după tipul de atac (Attack Type):")
            st.dataframe(data.groupby("Attack Type").agg({
                col_to_analyze: [min, max, sum, "mean"],
                "Country": "count"
            }))

        if "Target Industry" in data.columns and "Year" in data.columns:
            st.subheader(" Grupare după industrie și an:")
            st.dataframe(data.groupby(["Target Industry", "Year"]).agg({
                col_to_analyze: ["sum", "mean"],
                "Country": "count"
            }))

        if "Country" in data.columns and "Attack Source" in data.columns:
            st.subheader(" Grupare pe țară și sursa atacului:")
            st.dataframe(data.groupby(["Country", "Attack Source"]).agg({
                col_to_analyze: ["sum", "mean"]
            }))

        if "Attack Type" in data.columns and "Financial Loss (in Million $)" in data.columns:
            st.subheader(" Pierdere financiară totală pe tip de atac:")
            loss_by_attack = data.groupby("Attack Type")["Financial Loss (in Million $)"].sum().sort_values(ascending=False)
            fig3, ax3 = plt.subplots()
            loss_by_attack.plot(kind="bar", ax=ax3)
            ax3.set_ylabel("Pierdere financiară (mil. $)")
            ax3.set_title(" Pierderi financiare pe tipuri de atac")
            ax3.tick_params(axis='x', rotation=45)
            st.pyplot(fig3)

        # Matricea de corelatie
        st.subheader(" Matricea de corelație")
        corr = data[numeric_cols].corr()
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax4)
        ax4.set_title("Matricea de corelație pentru variabilele numerice")
        st.pyplot(fig4)

    else:
        st.warning(" Încărcați un fișier mai întâi.")

# Codificare date
# Codificare date
elif section == "Codificare Date":
    st.header("🌐 Codificare a datelor categorice")
    if "data" in st.session_state:
        data = st.session_state["data"]
        data_copy = data.copy()  # Copie temporară

        cat_cols = data.select_dtypes(include='object').columns.tolist()

        if not cat_cols:
            st.warning(" Nu există coloane categorice în dataset.")
        else:
            col_to_encode = st.selectbox("Selectați o coloană categorică pentru codificare:", cat_cols)
            encoding_type = st.radio("Tip de codificare:", ["Label Encoding", "One-Hot Encoding"])

            if encoding_type == "Label Encoding":
                encoded_col_name = col_to_encode + "_encoded"
                if encoded_col_name not in data.columns:
                    le = LabelEncoder()
                    data_copy[encoded_col_name] = le.fit_transform(data_copy[col_to_encode].astype(str))
                    st.success(f"Codificare Label aplicată pe {col_to_encode}")
                    data[encoded_col_name] = data_copy[encoded_col_name]  # salvăm înapoi doar coloana nouă
                else:
                    st.info(f" Coloana `{encoded_col_name}` există deja. Nu a fost recreată.")

                st.dataframe(data_copy[[col_to_encode, encoded_col_name]].head())

            else:  # One-Hot Encoding
                encoded_df = pd.get_dummies(data_copy[col_to_encode], prefix=col_to_encode)

                # Eliminăm coloanele care există deja
                encoded_df = encoded_df.loc[:, ~encoded_df.columns.isin(data.columns)]

                if not encoded_df.empty:
                    data_copy = pd.concat([data_copy, encoded_df], axis=1)
                    data = pd.concat([data, encoded_df], axis=1)  # salvăm și în sesiune
                    st.success(f"One-Hot Encoding aplicat pe {col_to_encode}")
                else:
                    st.info(f" Codificarea One-Hot pentru `{col_to_encode}` există deja în date.")

                st.dataframe(data_copy[[col_to_encode] + list(encoded_df.columns)].head())

            st.session_state["data"] = data



# Scalare date

elif section == "Scalare Date":
    st.header(" Scalare a variabilelor numerice")
    if "data" in st.session_state:
        data = st.session_state["data"]
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

        cols_to_scale = st.multiselect("Selectați coloane pentru scalare:", numeric_cols)
        scaler_type = st.radio("Alegeți metoda de scalare:", ["StandardScaler", "MinMaxScaler"])

        if cols_to_scale:
            scaler = StandardScaler() if scaler_type == "StandardScaler" else MinMaxScaler()
            scaled_data = scaler.fit_transform(data[cols_to_scale])

            # Evităm duplicate: suprascriem dacă există
            for i, col in enumerate(cols_to_scale):
                data[col + "_scaled"] = scaled_data[:, i]

            st.session_state["data"] = data
            st.success("Scalare aplicată cu succes.")
            st.dataframe(data[[col + "_scaled" for col in cols_to_scale]].head())


# Outlieri
elif section == "Outlieri":
    st.header(" Detectarea valorilor extreme (outlieri)")
    if "data" in st.session_state:
        data = st.session_state["data"]
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        col_to_check = st.selectbox("Alegeți o coloană numerică pentru analiză outlieri:", numeric_cols)

        Q1 = data[col_to_check].quantile(0.25)
        Q3 = data[col_to_check].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = data[(data[col_to_check] < lower) | (data[col_to_check] > upper)]

        st.write(f"Număr de outlieri detectați: {len(outliers)}")
        st.dataframe(outliers)

        fig, ax = plt.subplots()
        sns.boxplot(x=data[col_to_check], ax=ax)
        ax.set_title(f"Boxplot pentru {col_to_check}")
        st.pyplot(fig)

# Hartă geopandas
elif section == "Hartă Geospațială":
    st.header(" Hartă interactivă cu pierderi financiare")
    if "data" in st.session_state:
        data = st.session_state["data"]

        try:
            # Încarcă harta din fișierul shapefile descărcat
            world = gpd.read_file("data/ne_110m_admin_0_countries.shp")

            # Coloanele tale
            country_col = "Country"
            value_col = "Financial Loss (in Million $)"

            # Agregare pe țări
            losses = data.groupby(country_col)[value_col].sum().reset_index()

            # Unește datele tale cu harta
            world = world.merge(losses, how="left", left_on="NAME", right_on=country_col)

            # Afișează harta
            fig, ax = plt.subplots(figsize=(15, 8))
            world.plot(column=value_col, cmap="OrRd", legend=True, ax=ax,
                       missing_kwds={"color": "lightgrey", "label": "Fără date"})
            ax.set_title(" Pierderi financiare globale din atacuri cibernetice")
            st.pyplot(fig)

            st.markdown("""
            Harta evidențiază distribuția pierderilor financiare provocate de atacuri cibernetice în perioada 2015–2024.  
            Culorile mai închise indică pierderi mai mari, în timp ce zonele gri reprezintă lipsa de date.

            **Observații:**
            - Țări precum **Australia**, **Brazilia**, **Germania** și **China** par a fi cele mai afectate financiar.
            - În Europa, pierderile sunt concentrate în câteva țări, în timp ce alte state lipsesc din date.
            - **America de Nord și Africa** nu apar ca regiuni puternic afectate în acest set de date, ceea ce poate reflecta fie o subraportare, fie un nivel real mai scăzut al pierderilor raportate.
            - Zonele gri semnalează lipsa datelor sau imposibilitatea mapării exacte pe harta geopandas.

            """)


        except Exception as e:
            st.error(f" Eroare la afișarea hărții: {e}")
