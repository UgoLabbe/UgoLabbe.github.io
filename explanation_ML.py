import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "ML Classifier Visualization"

# Generate synthetic dataset
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X = StandardScaler().fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Interactive Machine Learning Classifier Visualization", style={'textAlign': 'center'}),
    html.Div([
        html.Label("Select Classifier:"),
        dcc.Dropdown(
            id='classifier-dropdown',
            options=[
                {'label': 'Decision Tree', 'value': 'dt'},
                {'label': 'K-Nearest Neighbors', 'value': 'knn'},
                {'label': 'Support Vector Machine', 'value': 'svm'}
            ],
            value='dt',
            style={'width': '50%'}
        ),
    ], style={'textAlign': 'center', 'padding': '20px'}),
    dcc.Graph(id='graph'),
], style={'fontFamily': 'Arial', 'padding': '20px'})

# Function to create the decision boundary plot
def create_decision_boundary(classifier, X_train, y_train):
    classifier.fit(X_train, y_train)
    h = .02  # Step size in the mesh
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return xx, yy, Z

# Update the graph based on the selected classifier
@app.callback(
    Output('graph', 'figure'),
    Input('classifier-dropdown', 'value')
)
def update_figure(selected_classifier):
    # Select the classifier based on dropdown value
    if selected_classifier == 'dt':
        classifier = DecisionTreeClassifier()
    elif selected_classifier == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=5)
    elif selected_classifier == 'svm':
        classifier = SVC(kernel='linear')
    
    # Create decision boundary
    xx, yy, Z = create_decision_boundary(classifier, X_train, y_train)
    
    # Create scatter plots for the data points
    trace0 = go.Scatter(
        x=X_train[y_train == 0][:, 0], 
        y=X_train[y_train == 0][:, 1],
        mode='markers',
        name='Class 0',
        marker=dict(color='blue', line=dict(width=1))
    )
    trace1 = go.Scatter(
        x=X_train[y_train == 1][:, 0], 
        y=X_train[y_train == 1][:, 1],
        mode='markers',
        name='Class 1',
        marker=dict(color='red', line=dict(width=1))
    )
    
    # Create contour plot for the decision boundary
    contour = go.Contour(
        x=xx[0],
        y=yy[:, 0],
        z=Z,
        showscale=False,
        colorscale='RdBu',
        opacity=0.4
    )
    
    return {
        'data': [trace0, trace1, contour],
        'layout': go.Layout(
            title=f"Decision Boundary for {selected_classifier.upper()}",
            xaxis={'title': 'Feature 1'},
            yaxis={'title': 'Feature 2'},
            showlegend=True,
            legend=dict(x=0, y=1),
            margin=dict(l=40, r=40, t=40, b=40)
        )
    }

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
