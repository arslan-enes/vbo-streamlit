import streamlit as st
import plotly.express as px
from ml import plot_model, predict_model

if 'tahmin' not in st.session_state:
    st.session_state['tahmin'] = False


@st.cache
def get_data():
    df = px.data.gapminder()[["country", "iso_alpha", "continent", "year", "gdpPercap", "pop", "lifeExp"]]
    return df


def user_input_graph(dataframe, countries, variable, mean_line=False):

    df = dataframe[dataframe.country.isin(countries)]
    fig = px.line(df, x="year", y=variable, color="country")
    if mean_line:
        fig.add_hline(y=df[variable].mean(), line_dash="dash", line_color="black")
    fig.update_layout(
        title=f"{variable} Değişkeninin Ülkeler Arasındaki Değişimi",
        xaxis_title="Yıllar",
        yaxis_title=variable,
        legend_title="Ülkeler"
    )
    return fig

def main():

    tab1, tab2, tab3, tab4 = st.tabs(["Gapminder", "Veri Seti", "Görselleştirmeler", "Modelleme"])

    # Gapminder Tanıtım
    cont_info = tab1.container()
    cont_info.title("Gapminder Veri Seti ile Yaşam Beklentisi Tahmini")
    cont_info.header("Gapminder Nedir?")
    cont_info.write("Gapminder, önemli küresel eğilimler ve oranlar hakkında sistematik yanlış anlamaları tanımlar"
                    " ve insanları yanlış anlamalarından kurtarmak için anlaşılması kolay öğretim materyalleri geliştirmek"
                    " için güvenilir veriler kullanır.")
    cont_info.markdown("Gapminder, **siyasi, dini veya ekonomik bağlantıları olmayan** bağımsız bir İsveç vakfıdır.")

    col1, col2 = cont_info.columns(2)
    col1.image('https://mir-s3-cdn-cf.behance.net/projects/404/2eaf5951889477.Y3JvcCwzMjY4LDI1NTgsMTYsMA.jpg')
    col2.image('https://aydinersoz.files.wordpress.com/2014/10/5a950989e5d388e85fb0f0363789268aab3c3c43_2400x1800.jpg')
    cont_info.video('https://youtu.be/RUwS1uAdUcI?t=201')

    # Veri Seti Tanıtım
    df = get_data()
    cont_data = tab2.container()
    cont_data.header("Gapminder Veri Setine Genel Bakış")
    cont_data.dataframe(df)

    grouped_by_year = df.groupby("year").mean()
    col1, col2, col3 = cont_data.columns(3)
    col1.metric(label="Yaşam Beklentisi", value=round(grouped_by_year.lifeExp.iloc[-1], 2),
                delta=f"+{grouped_by_year.iloc[-1].lifeExp - grouped_by_year.iloc[0].lifeExp}")
    col2.metric(label="Gelir", value=round(grouped_by_year.gdpPercap.iloc[-1], 2),
                delta=f"+{grouped_by_year.iloc[-1].gdpPercap - grouped_by_year.iloc[0].gdpPercap}")
    col3.metric(label="Nüfus", value=round(grouped_by_year['pop'].mean(), 2),
                delta=f"+{grouped_by_year.iloc[-1]['pop'] - grouped_by_year.iloc[0]['pop']}")

    # Grafikler
    graph_cont = tab3.container()
    # Grafik 1
    graph_cont.header("Veri Setinin Görselleştirilmesi")
    graph_cont.subheader("Ülkelerin Yıllar İçerisindeki Değişimlerinin Karşılaştırılması")
    col1, col2 = graph_cont.columns(2)
    countries_selected = col1.multiselect("Ülkeler", df.country.unique())
    variable_select = col2.selectbox("Görselleştirilecek Değişkeni Seçiniz",
                                     options=["lifeExp", "gdpPercap", "pop"])
    mean_line = col2.checkbox("Ortalama Çizgisi")
    if countries_selected:
        fig1 = user_input_graph(df, countries_selected, variable_select, mean_line)
        graph_cont.plotly_chart(fig1)

    # Grafik 2
    graph_cont.subheader("Ülkelerin Yıllar İçerisinde Yaşam Beklentisi Değişikliğinin Harita Üzerinde Gösterilmesi")
    year_select_for_map = graph_cont.slider("Yıllar ", min_value=int(df.year.min()), max_value=int(df.year.max()),
                                            step=5)
    fig2 = px.choropleth(df[df.year == year_select_for_map], locations="iso_alpha",
                         color="lifeExp",
                         range_color=(df.lifeExp.min(), df.lifeExp.max()),
                         hover_name="country",
                         color_continuous_scale=px.colors.sequential.Plasma,
                         width=650, height=500)
    graph_cont.plotly_chart(fig2)

    # Grafik 3
    graph_cont.subheader("Ülkelerin Yıllar İçerisindeki Nüfus, GSMH ve Yaşam Beklentisi Değişimleri")

    fig3 = px.scatter(df, x="gdpPercap", y="lifeExp", size="pop", color="continent",
                      animation_group='country', animation_frame="year",
                      hover_name="country", range_x=[100, 100000], range_y=[25, 90], log_x=True, size_max=60)
    fig3.add_hline(y=50, line_dash="dash", line_color="black")
    graph_cont.plotly_chart(fig3)

    # Modelleme
    model_cont = tab4.container()
    selected_model = model_cont.selectbox("Model Seçiniz", options=["Linear Regression",
                                                                    "Support Vector Regression",
                                                                    "Light Gradient Boosting Machine"])
    model_button = model_cont.button("Modeli Eğit")
    if model_button:
        fig = plot_model(df, selected_model)
        model_cont.plotly_chart(fig)
        st.session_state['tahmin'] = True

    # Tahmin
    if st.session_state['tahmin']:
        model_cont.subheader("Tahmin")
        col1, col2, col3 = model_cont.columns(3)
        selected_gdp = col1.number_input("GSMH")
        selected_pop = col2.number_input("Nüfus")
        if col1.button("Tahminle"):
            prediction = predict_model(df, selected_model, selected_gdp, selected_pop)
            col3.metric(label="Tahmin Edilen Yaşam Beklentisi", value=(prediction[0]))
            st.balloons()

if __name__ == '__main__':
    main()