from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from sound_params import *
import plotly.express as px
from scipy.io import wavfile


app = Dash(__name__)

# read WAV file
framerate, sound = wavfile.read('./sounds/song.wav')
sound = sound[:, 1].astype(np.float64)
nframes = len(sound)
duration = nframes / framerate
times = np.array([i/framerate for i in range(nframes)])

#
# def plot_wave(times, sound, title, xaxis,  frame_size = 8):
#     return go.Scatter(x=times, y=sound)
print(len(sound), len(times))
frame_size = 8
app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='sound-wave',
                figure=go.Figure(data=[go.Scatter(x=times, y=sound, mode='lines', hoverinfo='none')],
                                 layout=go.Layout(title='Line Plot'))
            ),
            dcc.Graph(
                id='volume',
                figure=go.Figure(data=[go.Scatter(x=times[(frame_size-1)//2:-frame_size//2],
                               y=volume(sound, frame_size),
                                                  mode='lines',
                                                  hoverinfo='none')],
                                 layout=go.Layout(title='Line Plot'))
            ),
            dcc.Graph(
                id='zcr',
                figure=go.Figure(data=[go.Scatter(x=times[(frame_size-1)//2:-frame_size//2],
                                   y=zcr(sound, frame_size),
                                                  mode='lines',
                                                  hoverinfo='none')],
                                 layout=go.Layout(title='Line Plot'))
            ),
            # dcc.Graph(
            #     id='volume',
            #     figure=px.line(x=times[(frame_size-1)//2:-frame_size//2],
            #                    y=volume(sound, frame_size),
            #                    hover_data=[]),
            # ),
            # dcc.Graph(
            #         id='zcr',
            #         figure=px.line(x=times[(frame_size-1)//2:-frame_size//2],
            #                        y=zcr(sound, frame_size),
            #                        hover_data=[]),
            #     )
        ]),
        dbc.Col([

        ])
    ])
])


if __name__ == '__main__':
    app.run_server(debug=True)
