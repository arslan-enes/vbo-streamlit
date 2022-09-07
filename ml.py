import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.svm import SVR
from lightgbm import LGBMRegressor


def data_preprocessing(df):
    # data preprocessing
    df = df.groupby("country").mean().reset_index()
    df = df.drop(["year"], axis=1)
    countries = df["country"]
    num_cols = [col for col in df.columns if df[col].dtype != "object"]
    df = df[num_cols]
    df = np.log2(df)
    return df, countries


def models(model_name):
    models = {
        "Linear Regression": LinearRegression(),
        "Support Vector Regression": SVR(),
        "Light Gradient Boosting Machine": LGBMRegressor()}
    return models[model_name]


def plot_model(df, model):

    mesh_size = .02
    margin = 0

    # data preprocessing
    df, countries = data_preprocessing(df)

    X = df[['pop', 'gdpPercap']]
    y = df['lifeExp']

    # Condition the model on sepal width and length, predict the petal width
    model = models(model)
    model.fit(X, y)

    # Create a mesh grid on which we will run our model
    x_min, x_max = X['pop'].min() - margin, X['pop'].max() + margin
    y_min, y_max = X.gdpPercap.min() - margin, X.gdpPercap.max() + margin
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)
    xx, yy = np.meshgrid(xrange, yrange)

    # Run model
    pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
    pred = pred.reshape(xx.shape)

    # Generate the plot
    df["country"] = countries
    fig = px.scatter_3d(df, x='pop', y='gdpPercap', z='lifeExp', hover_name='country')
    fig.update_traces(marker=dict(size=5))
    fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))

    return fig


def predict_model(df, model, gdp, pop):
    df, _ = data_preprocessing(df)
    model = models(model)
    X = df[['pop', 'gdpPercap']]
    y = df['lifeExp']
    gdp = np.log2(gdp)
    pop = np.log2(pop)
    model.fit(X, y)
    pred = model.predict([[pop, gdp]])
    pred = 2**pred
    return pred
