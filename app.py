from viktor import ViktorController
from viktor.parametrization import ViktorParametrization, NumberField, Section
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
import tensorflow as tf
from joblib import load
from munch import Munch
import numpy as np
from viktor.views import (
    ImageAndDataView,
    PlotlyView,
    PlotlyResult,
    PlotlyAndDataView,
    PlotlyAndDataResult,
    ImageAndDataResult,
    DataView,
    DataGroup,
    DataResult,
    DataItem,
    WebView,
    WebResult,
)
from sklearn.preprocessing import MinMaxScaler
def load_model():
    model_path = Path(__file__).parent / 'LSTM_Model' / 'LSTM.h5'
    return tf.keras.models.load_model(model_path)


class Parametrization(ViktorParametrization):
    section1 = Section('Basic tunnel parameters')
    section1.C = NumberField('Cover depth (m)', variant='slider', min=9.1, max=31.6,default = 18.2)
    section1.D = NumberField('Tunnel diameter (m)', variant='slider', min=3, max=10,default = 6)
    section1.k = NumberField('Trough width parameter', variant='slider', min=0.2, max=0.7,default = 0.5,step=0.1,
                             description= 'Stiff, fissured clay 0.4 to 0.5, Glacial deposits 0.5 to 0.6,Recent soft, silty clay 0.6 to 0.7, Granular soils above water table 0.2 to 0.3')


    section2 = Section('Shield operational parameters')
    section2.To = NumberField('Torque (MN · m)', variant='slider', min=0.29, max=4.7,default = 2.5)
    section2.Pr = NumberField('Penetration rate Pr (mm/min)', variant='slider', min=2.4, max=64,default = 27.2)
    section2.Th = NumberField('Thrust (MN)', variant='slider', min=3.8, max=24.2,default = 12)
    section2.Fp = NumberField('Face pressure Fp (bar)', variant='slider', min=0, max=2.5,default = 1)
    section2.Rs = NumberField('Rotational speed (rpm)', variant='slider', min=0.86, max=3.3,default = 1.48)
    section2.St = NumberField('Stoppage', variant='slider', min=0, max=1,default = 0,
                     description='1 for stoppage and 0 for continuous advancement.')

    section3 = Section('Geological and Geotechnical parameters')
    section3.W = NumberField('Water table W (m)', variant='slider', min=0, max=25.5,default = 10.8)
    section3.MSPT = NumberField('Modified SPT blow count', variant='slider', min=0, max=38.7,default = 5.7)
    section3.MDPT = NumberField('Modified DCP blow count', variant='slider', min=0, max=12.4,default = 2)
    section3.MUCS = NumberField('Modified UCS', variant='slider', min=0, max=36.3,default = 7.8)
    section3.Gc = NumberField('Ground condition at tunnel face GC', variant='slider', min=0, max=4,default = 1, description='1 for soil, 2 for gravel, 3 for rock, and 4 for mixed-face ground.')

class Controller(ViktorController):
    label = 'LSTM'
    parametrization = Parametrization

    # @DataView("OUTPUT", duration_guess=1)
    # def visualize_data(self, params: Munch, **kwargs) -> ImageAndDataResult:
    #     model = load_model()
    #     scaler = load('scaler.joblib')
    #     data = {
    #     'To(MN·m)':[params.section2.To],
    #     'Pr(mm/min)':[params.section2.Pr],
    #     'Th(MN)':[params.section2.Th],
    #     'Fp(bar)':[params.section2.Fp],
    #     'Rs(rpm)':[params.section2.Rs],
    #     'C(m)':[params.section1.C],
    #     'W(m)':[params.section3.W],
    #     'MSPT':[params.section3.MSPT],
    #     'MDPT':[params.section3.MDPT],
    #     'MUCS':[params.section3.MUCS],
    #     'Gc':[params.section3.Gc],
    #     'St':[params.section2.St],
    #     }
    #     data = pd.DataFrame(data)
    #     scaled_data = scaler.transform(data)
    #     input = pd.DataFrame(scaled_data, columns=data.columns)
    #     reshaped_input = np.array([input.values])
    #     time_steps = 3
    #     if reshaped_input.shape[1] < time_steps:
    #         # Pad the input with zeros if it has fewer than the required time steps
    #         padded_input = np.zeros((1, time_steps, reshaped_input.shape[2]))
    #         padded_input[:, -reshaped_input.shape[1]:, :] = reshaped_input
    #         reshaped_input = padded_input
    #     pred = model.predict(reshaped_input)
    #     # Generate results
    #     result = DataGroup(
    #         DataItem('Volume Loss Predition based on LSTM deep learning(%)', pred.item())
    #     )
    #     return DataResult(result)

    @PlotlyAndDataView("Plotly and data view", duration_guess=1)
    def get_plotly_and_data_view(self, params, **kwargs):
        model = load_model()
        scaler = load('scaler.joblib')
        data = {
        'To(MN·m)':[params.section2.To],
        'Pr(mm/min)':[params.section2.Pr],
        'Th(MN)':[params.section2.Th],
        'Fp(bar)':[params.section2.Fp],
        'Rs(rpm)':[params.section2.Rs],
        'C(m)':[params.section1.C],
        'W(m)':[params.section3.W],
        'MSPT':[params.section3.MSPT],
        'MDPT':[params.section3.MDPT],
        'MUCS':[params.section3.MUCS],
        'Gc':[params.section3.Gc],
        'St':[params.section2.St],
        }
        data = pd.DataFrame(data)
        scaled_data = scaler.transform(data)
        input = pd.DataFrame(scaled_data, columns=data.columns)
        reshaped_input = np.array([input.values])
        time_steps = 3
        if reshaped_input.shape[1] < time_steps:
            # Pad the input with zeros if it has fewer than the required time steps
            padded_input = np.zeros((1, time_steps, reshaped_input.shape[2]))
            padded_input[:, -reshaped_input.shape[1]:, :] = reshaped_input
            reshaped_input = padded_input
        pred = model.predict(reshaped_input)
        # Generate results
        result = DataGroup(
            DataItem('Volume Loss Predition based on LSTM eep learning(%)', pred.item())
        )

        diameter = params.section1.D # Diameter of the circle
        x = 0  # x-coordinate of the center
        y =  -params.section1.C # y-coordinate of the center
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = x + diameter / 2 * np.cos(theta)
        circle_y = y + diameter / 2 * np.sin(theta)

        settlement_x= x + 10*diameter / 2 * np.cos(theta)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=circle_x,
            y=circle_y,
            mode='lines',
            line=dict(color='red',shape='spline'),
            name='Tunnel',
        ))
        # Plot ground surface
        fig.add_trace(go.Scatter(
            x=[-30,30],
            y=[0, 0],
            mode='lines',
            line=dict(color='black', width=4),
            name='Ground Surface'
        ))
        # Plot settlement
        i = params.section1.k * params.section1.C
        pi = 3.1415926
        fig.add_trace(go.Scatter(
            x=settlement_x,
            y= -100*pred.item()*0.01 * pi * params.section1.D * params.section1.D /(4 * i * np.sqrt(2*pi)) * np.exp(-settlement_x*settlement_x/(2*i*i)),
            mode='lines',
            line=dict(color='blue', width=4),
            name='Settlement profile (Increased by a factor of 300)'
        ))
        x_min = -20
        x_max = 20
        y_min = -40
        y_max = 10
        fig.update_layout(
            xaxis=dict(range=[x_min, x_max], constrain='domain', scaleratio=1),
            yaxis=dict(range=[y_min, y_max], scaleratio=1),
            showlegend=False
            )


        return PlotlyAndDataResult(fig.to_json(),result)

