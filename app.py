import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from IPython.display import Image
from PIL import Image



def main():

    menu_options = ["Dataset", "Data modeling", "Data verkenning", "Profielen", "Voorspelling","Conclusie"]
    
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

        df= pd.read_csv('alle_ladpalen.csv')
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
        data = pd.read_pickle('231113_Raw_data.pkl')

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

        fig, axes = plt.subplots(nrows=len(sensoren), ncols=1, figsize=(10, 4 * len(sensoren)))

        # Itereer over sensoren
        for i, sensor in enumerate(sensoren):
        # Filter data voor de huidige sensor en laadpaal
            test = data[data['sensor'] == laadpalen[0] + sensor]
            testdag1 = test[(test['day'].isin([27, 28, 29])) & (test['month'] == 4)]

        # Plot op het overeenkomstige subplot
            axes[i].plot(testdag1.index, testdag1['v'])
            axes[i].set_title(f'Sensor {sensor}')
            axes[i].set_xlabel('Index')
            axes[i].set_ylabel('Sensor Waarde')

        plt.tight_layout()
        st.pyplot(fig)

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
        st.pyplot(fig)

        st.markdown("""
        De plot laat zien dat er een duidelijk patroon te zien is over het laadgedrag van elektrische auto bestuurders. Over het algemeen wordt in de ochtenduren aanzienlijk meer opgeladen, wat resulteert in een piek in laadactiviteit gedurende deze periode.
        In de middag is er echter een opmerkelijke reductie van ongeveer 50% in vergelijking met de ochtenduren. Deze afname suggereert een patroon waarbij het aantal laadsessies aanzienlijk afneemt naarmate de dag vordert.
        Ook valt op dat laadpaal 15 van alle laadpalen het meest wordt gebruikt, wellicht heeft dit te maken met de locatie van deze laadpaal.
        """)

        #PLOT 2
        st.markdown("**Profiel 2: Verschillende dagen**")
        fig = px.bar(df, x='Day', title='Charging Period days')
        st.pyplot(fig)
        
        #PLOT 3
        st.title('Interactive Scatter Plot')

        # Create a scatter plot with dropdown menu for variable selection
        fig = px.scatter(concatenated_laadpalen, x='Start Time', y='3PhaseActivePowW', color='type',
                     title='Scatter Plot', marginal_x='histogram')

        fig.update_layout(
            updatemenus=[
            {
                'buttons': [
                    {'method': 'update', 'label': '3PhaseActivePowW', 'args': [{'y': [concatenated_laadpalen['3PhaseActivePowW']]}]},
                    {'method': 'update', 'label': 'L1CurrentA', 'args': [{'y': [concatenated_laadpalen['L1CurrentA']]}]},
                    {'method': 'update', 'label': 'Duration (min)', 'args': [{'y': [concatenated_laadpalen['Duration (min)']]}]},
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
       
        fig = px.scatter(concatenated_laadpalen, x='Start Time', y='L1CurrentA', color='type',
                         title='Scatter Plot')

        fig.update_layout(
            updatemenus=[
                {
                    'buttons': [
                        {'method': 'update', 'label': 'L1CurrentA', 'args': [{'y': [concatenated_laadpalen['L1CurrentA']]}]},
                        {'method': 'update', 'label': 'L2CurrentA', 'args': [{'y': [concatenated_laadpalen['L2CurrentA']]}]},
                        {'method': 'update', 'label': 'L3CurrentA', 'args': [{'y': [concatenated_laadpalen['L3CurrentA']]}]},
                        {'method': 'update', 'label': 'maxAppliedChargingCurrentA', 'args': [{'y': [concatenated_laadpalen['maxAppliedChargingCurrentA']]}]},
                        {'method': 'update', 'label': 'maxChargingCurrentA', 'args': [{'y': [concatenated_laadpalen['maxChargingCurrentA']]}]},
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
        sns.countplot(x='type', data=prob, palette='viridis')
        plt.title('Count Plot voor laadpalen')
        plt.xlabel('Type')
        plt.xticks(rotation =45)
        plt.ylabel('Count')
        st.pyplot(plt)


        #PLOT 5
        st.markdown("**Profiel 3: Seizoensgebonden laden**")
        df['Season'] = df['Start Time'].dt.month.map({
        1: 'Winter', 2: 'Winter', 3: 'Lente',
        4: 'Lente', 5: 'Lente', 6: 'Zomer',
        7: 'Zomer', 8: 'Zomer', 9: 'Herfst',
        10: 'Herfst', 11: 'Herfst', 12: 'Winter'
        })

        # Voeg een multi-select widget toe voor seizoenen
        selected_seasons = st.multiselect('Selecteer seizoen(en)', df['Season'].unique())

        fig, ax = plt.subplots(figsize=(18, 12))

        season_colors = {
        'Winter': 'blue',
        'Lente': 'green',
        'Zomer': 'red',
        'Herfst': 'orange'
        }

        if selected_seasons:
            for season in selected_seasons:
                season_charging = df[df['Season'] == season]
                plt.hist(season_charging['Duration (min)'], bins=30, alpha=0.7, label=season)
        

        plt.title('Histogram van Laadduren per Seizoen')
        plt.xlabel('Laadduur (minuten)')
        plt.ylabel('Aantal laadsessies')
        plt.legend(title='Seizoen')
        st.pyplot(fig)

        st.markdown("""
        Het onderzoeken van het laadgedrag over de seizoenen onthult interessante inzichten. In de lente is er de hoogste activiteit, wat suggereert dat gebruikers meer geneigd zijn om hun elektrische voertuigen op te laden bij aangenamer weer. Dit patroon blijft grotendeels consistent in de zomer, met een merkbare daling in de winter.

        De seizoensgebonden variatie kan beïnvloed worden door verschillende factoren, waaronder temperatuur, daglichturen en wegomstandigheden. In warmer weer zijn mensen mogelijk meer bereid om hun elektrische voertuigen te gebruiken en op te laden, terwijl kouder weer en verminderd daglicht de laadactiviteit kunnen verminderen.
        """)
    

    #PAGINA 5 VOORSPELLING
    if choice == "Voorspelling":
        st.title("Data voorspellen")
        st.markdown("xx")

    #PAGINA 6 CONCLUSIE
    if choice == "Conclusie":
        st.title("Conclusie")
        st.markdown("xx")

if __name__ == "__main__":
    main()
        
    
  



