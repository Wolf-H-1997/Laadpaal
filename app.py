import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from pmdarima.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pmdarima as pm




def main():

    menu_options = ["Dataset", "Data modeling", "Data verkenning", "Profielen","Tijdreeks", "Voorspelling","Conclusie"]
    
    choice = st.sidebar.selectbox("Menu", menu_options)

    #PAGINA 1 DATASET
    if choice == "Dataset":
        st.title("Laadpalen dataset")
        st.markdown("Op deze pagina gaan we de dataset uitleggen. De gegevens zijn afkomstig van een oplaadsysteem voor elektrische voertuigen van Zeeburg P+R. Er is een on-site batterij gebruikt om het opladen van de EV's aan te vullen, gezien de relatief kleine netaansluiting. De gegevens zelf zijn in onbewerkt formaat en bevatten veel sensormetingen, tot 800 metingen per tijdstap van 5 minuten. Het energie netwerk waar de laadpalen aan zijn vastgesloten is Vattenfall. Op de P+R zijn 16 laadpalen aanwezig. Laten we een kijkje nemen naar de orginele dataset.")

        st.markdown("De orginele dataset:")
        
        data = pd.read_csv('data')
        st.dataframe(data.head())

        st.markdown("""
        De dataset bevat informatie over verschillende sensoren, metingen en tijdstempels:
                    
        - **Aantal Rijen:** De dataset bestaat uit 9 rijen.
        - **Kolommen:** Er zijn 8 kolommen, waaronder 'sensor', 'type', 'v', 'u', 'year', 'month', 'day', en 'hour'.
        
        Uitleg van de kolommen:

        - **Start:** Het tijdstip waarop de meting is gestart.
        - **Sensor:** De sensor die de meting heeft geregistreerd.
        - **Type:** Het type meting dat is uitgevoerd.
        - **V:** De gemeten waarde (bijvoorbeeld spanning, stroom, frequentie).
        - **U:** De eenheid van de gemeten waarde.
        - **Year:** Het jaar van de meting. 
        - **Month:** De maand van de meting.
        - **Day:** De dag van de meting.
        - **Hour:** Het uur van de meting.
        """)

    #PAGINA 2 DATA MODELING
    if choice == "Data modeling":
        st.title("Data modeling en aanpassen van de dataset")

        st.markdown("""
        In dit gedeelte bespreken we het datamodeling-proces en de aanpassingen die zijn aangebracht aan de originele dataset. We hebben nieuwe kolommen toegevoegd en de dataset verrijkt met aanvullende informatie. Hier zijn enkele van de belangrijkste wijzigingen:""")

        df= pd.read_csv('allelaadpalen')
        st.dataframe(df.head())

        st.markdown("""
        - **Start Time en End Time:** We hebben de tijdsinformatie opgesplitst in aparte kolommen om een duidelijker beeld te krijgen van de start- en eindtijden van metingen.
        - **Duration (min):** We hebben een nieuwe kolom toegevoegd om de duur van elke meting in minuten weer te geven.
        - **3PhaseActivePowW, L1ActivePowW, L2ActivePowW, L3ActivePowW:** We hebben kolommen toegevoegd voor actief vermogen in elk van de drie fasen.
        - **3PhaseRealEnergyDeliveredWh:** Een nieuwe kolom die het werkelijk geleverde energieverbruik over de drie fasen weergeeft.
        - **L1CurrentA, L2CurrentA, L3CurrentA:** Kolommen voor de stroom in elk van de drie fasen.
        - **L1L2VoltageV, L2L3VoltageV, L3L1VoltageV:** Kolommen voor de spanning tussen respectieve fasen.
        - **maxAppliedChargingCurrentA, maxChargingCurrentA, nPhasesCharging:** We hebben nieuwe kolommen toegevoegd die verband houden met oplaadparameters en het aantal fasen tijdens het opladen.
        - **VehicleCharging, VehicleConnected, Connected_without_charging:** We hebben informatie toegevoegd over de status van het voertuig, of het nu aan het opladen is, verbonden is of verbonden zonder opladen.
        """)


    #PAGINA 3 DATA VERKENNING
    if choice == "Data verkenning":
        st.title("Data verkenning")
        st.markdown("Op deze pagina gaan we de aangepaste dataset bekijken en analyseren. Hieronder ziet u een aantal plots. Elke plot is bedoeld om de tijdreeksgegevens van een specifieke sensor weer te geven op basis van de opgegeven laadpaal. Op elke plot bevat de naam van de sensor en de x-as is gelabeld als 'Index' en de y-as als 'Sensor Waarde'. ")

        #PLOT 1
        #data = pd.read_pickle('231113_Raw_data.pkl')
        image_path1 = 'image1.png'
        image_path2 = 'image2.png'
        image_path3 = 'image3.png'
        image_path4 = 'image4.png'
        image_path5 = 'image5.png'
        
        laadpalen = ['ams-a-chrg-0-0-', 'ams-a-chrg-0-1-', 'ams-a-chrg-1-0-', 'ams-a-chrg-1-1-',
                'ams-a-chrg-2-0-', 'ams-a-chrg-2-1-', 'ams-a-chrg-3-0-', 'ams-a-chrg-3-1-',
                'ams-a-chrg-4-0-', 'ams-a-chrg-4-1-', 'ams-a-chrg-5-0-', 'ams-a-chrg-5-1-',
                'ams-a-chrg-6-0-', 'ams-a-chrg-6-1-', 'ams-a-chrg-7-0-', 'ams-a-chrg-7-1-']

        sensoren = ['3PhaseActivePowW', '3PhaseRealEnergyDeliveredWh', 'L1ActivePowW', 'L1CurrentA', 'L1L2VoltageV',
                'L2ActivePowW', 'L2CurrentA', 'L2L3VoltageV', 'L3ActivePowW', 'L3CurrentA', 'L3L1VoltageV',
                'maxAppliedChargingCurrentA', 'maxChargingCurrentA', 'nPhasesCharging', 'VehicleCharging', 'VehicleConnected']
        
        laadpaal_1 = 'ams-a-chrg-0-0-'
        laadpaal_2 = 'ams-a-chrg-0-1-'
        laadpaal_3 = 'ams-a-chrg-1-0-'
        laadpaal_4 = 'ams-a-chrg-1-1-'  
        laadpaal_5 = 'ams-a-chrg-2-0-'
        laadpaal_6 = 'ams-a-chrg-2-1-'
        laadpaal_7 = 'ams-a-chrg-3-0-'
        laadpaal_8 = 'ams-a-chrg-3-1-'
        laadpaal_9 = 'ams-a-chrg-4-0-'
        laadpaal_10 = 'ams-a-chrg-4-1-'
        laadpaal_11 = 'ams-a-chrg-5-0-'
        laadpaal_12 = 'ams-a-chrg-5-1-'
        laadpaal_13 = 'ams-a-chrg-6-0-'
        laadpaal_14 = 'ams-a-chrg-6-1-'
        laadpaal_15 = 'ams-a-chrg-7-0-'
        laadpaal_16= 'ams-a-chrg-7-1-'
        active_power = '3PhaseActivePowW'
        real_energy = '3PhaseRealEnergyDeliveredWh'
        L1_W = 'L1ActivePowW'
        L1_A = 'L1CurrentA'
        L1_V = 'L1L2VoltageV'   
        L2_W = 'L2ActivePowW'
        L2_A = 'L2CurrentA'
        L2_V = 'L2L3VoltageV'
        L3_W = 'L3ActivePowW'
        L3_A = 'L3CurrentA'
        L3_V = 'L3L1VoltageV'
        max_applied = 'maxAppliedChargingCurrentA'
        max_charging = 'maxChargingCurrentA'
        phase = 'nPhasesCharging'
        charging = 'VehicleCharging'
        connected = 'VehicleConnected'
        laadpalen = [laadpaal_1, laadpaal_2, laadpaal_3, laadpaal_4, laadpaal_5, laadpaal_6, laadpaal_7, laadpaal_8, laadpaal_9, laadpaal_10 , laadpaal_11, laadpaal_12, laadpaal_13, laadpaal_14, laadpaal_15, laadpaal_16]
        sensoren = [active_power, real_energy, L1_W, L1_A, L1_V, L2_W, L2_A, L2_V, L3_W, L3_A, L3_V, max_applied, max_charging, phase, charging, connected ]

       image1 = st.image(image_path1, caption='3PhaseActivePowW', use_column_width=True)
       image2 = st.image(image_path2, caption = '3PhaseRealEnergyDeliveredWh', use_column_width = True)
       image3 = st.image(image_path3, caption='Voltage', use_column_width=True)
       image4 = st.image(image_path4, caption='Applied Ampere', use_column_width=True)
       image5 = st.image(image_path5, caption='Connected/Charging', use_column_width=True)
   


        st.markdown("""
        **Analyse van Sensoren**

        - 3PhaseActivePowW**:
        Op de eerste plot zie je de tijdreeks van sensor 3PhaseActivePowW. Er is een opvallende piek rond 04-27 12, waar de sensorwaarde ongeveer 9000 bereikt. Op 04-28 12 is er een kleine stijging naar 4000, gevolgd door een kort horizontaal stuk, en vervolgens een scherpe stijging naar meer dan 10000.

        - 3PhaseRealEnergyDeliveredWh**:
        De tweede plot toont de tijdreeks van sensor 3PhaseRealEnergyDeliveredWh. In het begin loopt deze horizontaal, stijgt dan naar ongeveer 10.15, loopt opnieuw horizontaal en stijgt vervolgens naar ongeveer 10.20, gevolgd door weer een horizontaal stuk.

        - **L1ActivePowW**:
        De derde plot geeft de tijdreeks van sensor L1ActivePowW weer. Hier zie je twee stijgingen, vergelijkbaar met 3PhaseActivePowW, maar met andere waarden. Het algemene patroon blijft echter hetzelfde.

        - **ActivePowW plots (L2ActivePowW en L3ActivePowW)**
        De vierde plot vertoont hetzelfde patroon als 3PhaseActivePowW en L1ActivePowW.

        - **Voltage Plots (L1L2VoltageV, L2L3VoltageV, L3L1VoltageV)**:
        Voor de voltage plots is er een duidelijke verspringing van energie, met een minimale waarde variërend van 388 tot 404.

        - **Plots voor VehicleCharging en VehicleConnected**:
        De plots voor VehicleCharging en VehicleConnected tonen een stijgende lijn van 0.0 naar 1.0. Dit suggereert het gebruik van de laadpaal wanneer de waarde 1.0 is en geen gebruik wanneer de waarde 0.0 is.
        """)

    #PAGINA 4 PROFIELEN
    if choice == "Profielen":
        st.title("Profielen")
        st.markdown("Op deze pagina worden de profielen getoond die wij als groep hebben ontdekt.")
        
        #PLOT 1
        st.markdown("**Profiel 1: Laden per dagdeel**")
        df= pd.read_csv('allelaadpalen')
        fig = px.bar(df, x='ChargingPeriod', title='Charging Period Distribution')
        st.plotly_chart(fig)

        st.markdown("""
        De plot laat zien dat er een duidelijk patroon te zien is over het laadgedrag van elektrische auto bestuurders. Over het algemeen wordt in de ochtenduren aanzienlijk meer opgeladen, wat resulteert in een piek in laadactiviteit gedurende deze periode.
        In de middag is er echter een opmerkelijke reductie van ongeveer 50% in vergelijking met de ochtenduren. Deze afname suggereert een patroon waarbij het aantal laadsessies aanzienlijk afneemt naarmate de dag vordert.
        Ook valt op dat laadpaal 15 van alle laadpalen het meest wordt gebruikt, wellicht heeft dit te maken met de locatie van deze laadpaal.
        """)

        #PLOT 2
        st.markdown("**Profiel 2: Verschillende dagen**")
        fig = px.bar(df, x='Day', title='Charging Period days')
        st.plotly_chart(fig)
        
        #PLOT 3
        st.title('Interactive Scatter Plot')

        # Create a scatter plot with dropdown menu for variable selection
        fig = px.scatter(df, x='Start Time', y='3PhaseActivePowW', color='type',
                     title='Scatter Plot', marginal_x='histogram')

        fig.update_layout(
            updatemenus=[
            {
                'buttons': [
                    {'method': 'update', 'label': '3PhaseActivePowW', 'args': [{'y': [df['3PhaseActivePowW']]}]},
                    {'method': 'update', 'label': 'L1CurrentA', 'args': [{'y': [df['L1CurrentA']]}]},
                    {'method': 'update', 'label': 'Duration (min)', 'args': [{'y': [df['Duration (min)']]}]},
                    ],
                    'direction': 'down',
                    'showactive': True,
                    'x': 0.1,
                    'xanchor': 'left',
                    'y': 1.15,
                    'yanchor': 'top',},])
        fig.update_xaxes(title_text='Start Time')
        fig.update_yaxes(title_text='Values')
        st.plotly_chart(fig)
        #PLOT4
       
        fig = px.scatter(df, x='Start Time', y='L1CurrentA', color='type',
                         title='Scatter Plot')

        fig.update_layout(
            updatemenus=[
                {
                    'buttons': [
                        {'method': 'update', 'label': 'L1CurrentA', 'args': [{'y': [df['L1CurrentA']]}]},
                        {'method': 'update', 'label': 'L2CurrentA', 'args': [{'y': [df['L2CurrentA']]}]},
                        {'method': 'update', 'label': 'L3CurrentA', 'args': [{'y': [df['L3CurrentA']]}]},
                        {'method': 'update', 'label': 'maxAppliedChargingCurrentA', 'args': [{'y': [df['maxAppliedChargingCurrentA']]}]},
                        {'method': 'update', 'label': 'maxChargingCurrentA', 'args': [{'y': [df['maxChargingCurrentA']]}]},
                    ],
                    'direction': 'down',
                    'showactive': True,
                    'x': 0.1,
                    'xanchor': 'left',
                    'y': 1.15,
                    'yanchor': 'top',},])

        fig.update_xaxes(title_text='Start Time')
        fig.update_yaxes(title_text = 'Ampere')
        st.plotly_chart(fig)
        #PLOT 6
        plt.figure(figsize=(10, 6))
        sns.countplot(x='type', data=df, palette='viridis')
        plt.title('Count Plot voor laadpalen')
        plt.xlabel('Type')
        plt.xticks(rotation =45)
        plt.ylabel('laadsessies')
        st.pyplot(plt)
    if choice == "Tijdreeks":
        dff = pd.read_csv('pw_uur')
        dff.set_index('start', inplace=True)
        dff.index = pd.to_datetime(dff.index)
        dff['day_of_week'] = dff.index.dayofweek

        # Define the days of the week
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        
        st.title("Average Total Active Power with Rolling Average")

        
        for day in days_of_week:
            day_data = dff[dff.index.day_name() == day]

            aggregated_data = day_data.groupby(day_data.index.time)['v'].mean()
            aggregated_data.index = aggregated_data.index.map(lambda x: x.strftime('%H:%M:%S'))

            rolling_avg = aggregated_data.rolling(window=5).mean()

            # Plotting
            st.subheader(f'Average Total Active Power on {day} with Rolling Average')
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(aggregated_data.index, aggregated_data.values, marker='o', linestyle='-', color='b', label='Original Data')
            ax.plot(rolling_avg.index, rolling_avg.values, linestyle='--', color='r', label='Rolling Average')
            plt.xticks(aggregated_data.index[::5])
            ax.set_xlabel('Time of Day')
            ax.set_ylabel('Total Active Power (Watt)')
            ax.legend()
            ax.grid(True)
            
            
            st.pyplot(fig)

       
    

    #PAGINA 5 VOORSPELLING
    if choice == "Voorspelling":
        st.title("Data voorspellen")
        st.markdown("Hier een arima plot met blablaablaablaablaaaa")
        pw_dag = pd.read_csv('pw_dag')
        pw_uur = pd.read_csv('pw_uur')
        pw_dag.set_index('start', inplace=True)
        pw_uur.set_index('start', inplace=True)
        pw_dag_diff = pw_dag.diff().dropna()


        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(pw_dag_diff, label='TotalActivePowW')
        ax.set_title('Totaal Power watt Total Over Time')
        ax.set_xlabel('Date')
        ax.set_xticks(range(0, len(pw_dag_diff), 5))
        ax.set_xticklabels(pw_dag_diff.index[::5], rotation=45, ha='right')
        ax.set_ylabel('TotalActivePowW')
        ax.legend()
        st.pyplot(fig)

        # Decompose the time series into trend, seasonality, and residual components
        result = seasonal_decompose(pw_dag_diff, model='additive', period=7)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

        result.trend.plot(ax=ax1)
        ax1.set_title('Trend Component')
        result.seasonal.plot(ax=ax2)
        ax2.set_title('Seasonal Component')
        result.resid.plot(ax=ax3)
        ax3.set_title('Residual Component')
        result.observed.plot(ax=ax4)
        ax4.set_title('Observed Data')

        plt.suptitle('Decomposition of TotalActivePowW Time Series', y=1.02)
        plt.tight_layout()
        st.pyplot(fig)

        # ADF TEST
        adf_result = adfuller(pw_dag_diff)
        st.write(f'ADF Statistic: {adf_result[0]}')
        st.write(f'p-value: {adf_result[1]}')
        st.write('Critical Values:')
        for key, value in adf_result[4].items():
            st.write(f'   {key}: {value}')

        # Plot ACF and PACF for determining the order of the ARIMA model
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(pw_dag_diff, ax=ax1, lags=20)
        plot_pacf(pw_dag_diff, ax=ax2, lags=20)
        plt.suptitle('ACF and PACF for TotalActivePowerW Time Series')
        st.pyplot(fig)
        st.divider()
        
        st.markdown('Het auto arima model geeft Best model:  ARIMA(4,0,2)(2,0,1)[7]')
        y = pw_dag_diff.sort_index()

        # Split the data into train and test data with a size of 0.8
        train_size = int(len(y) * 0.8)
        train, test = y[:train_size], y[train_size:]

        # Fit the ARIMA model with auto_arima parameters (these are calculated automatically)
        model = pm.auto_arima(train, seasonal=True, m=7, trace=True, suppress_warnings=True,
                              stepwise=True, maxiter=10, information_criterion='aic')

        # Make predictions
        predictions, conf_int = model.predict(n_periods=len(test), return_conf_int=True)

        # Calculate the MSE
        mse = mean_squared_error(test, predictions)
        st.write(f'Mean Squared Error: {mse}')

        # Make a plot with training data, test data, and predictions
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(train.index, train, label='Training Data')
        ax.plot(test.index, test, label='Test Data')
        ax.plot(test.index, predictions, label='Predictions', linestyle='--')
        ax.fill_between(test.index, conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.2, label='Confidence Interval')
        ax.set_title('ARIMA Model Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Total Power W')
        ax.legend()

        st.pyplot(fig)
        st.divider()
        pw_dag_diff.index = pd.to_datetime(pw_dag_diff.index)

        fit = ARIMA(pw_dag_diff, order=(4, 0, 2), seasonal_order=(2, 0, 1, 7)).fit()

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        pw_dag_diff.plot(marker='o', label='Actual Data', ax=ax)
        fit.fittedvalues.plot(style='--', label='Fitted Values', ax=ax)

        # Forecast
        forecast_steps = 21
        forecast_index = pd.date_range(start=pw_dag_diff.index[-1], periods=forecast_steps + 1, freq='D')[1:]
        forecast = fit.get_forecast(steps=forecast_steps)
        forecast_series = forecast.predicted_mean
        forecast_series.index = forecast_index
        forecast_series.plot(style='--', color='C1', label='ARIMA Forecast', ax=ax)

        ax.legend()

        st.pyplot(fig)
        st.divider()
        st.markdown('### Prophet voorspel model')
        df_prophet = pw_dag.reset_index()
        df_prophet.rename(columns={'start' : 'ds', 'v' : 'y'}, inplace =True)
        df_prophet = df_prophet.dropna()
        df_prophet = df_prophet.astype(object)
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        df_prophet['y'] = df_prophet['y'].astype(float)
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=21)  

        forecast = model.predict(future)

        fig = model.plot(forecast)
        st.pyplot(fig)
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)
        st.divider()
        df_prophet = pw_uur.reset_index()
        df_prophet.rename(columns={'start' : 'ds', 'v' : 'y'}, inplace =True)
        df_prophet = df_prophet.dropna()
        df_prophet = df_prophet.astype(object)
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        df_prophet['y'] = df_prophet['y'].astype(float)
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=21)

        forecast = model.predict(future)
        fig = model.plot(forecast)
        st.markdown('# Nu per uur')

        st.pyplot(fig)
        fig3 = model.plot_components(forecast)
        st.pyplot(fig3)
    #PAGINA 6 CONCLUSIE
    if choice == "Conclusie":
        st.title("Conclusie")
        st.markdown("xx")

if __name__ == "__main__":
    main()
        
    
  



