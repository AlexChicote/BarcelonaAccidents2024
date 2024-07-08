import math
import requests
from bs4 import BeautifulSoup
from datetime import date, datetime, timedelta
import json
import pandas as pd
import random
import numpy as np
import os


#from barcelona_keys import key as weather_key


def concatenating_dataframes(filter1,response):
    
    files=response['result']['results']
    for num, file in enumerate(files):
        
        if (('accidents-' +filter1+'gu') in file['name']) or ('accidents_' +filter1+'gu') in file['name']:
            #print(file['name'])
            filter_list=[]
            
            for fitxer in file['resources']:

                if fitxer['format']=='CSV':

                    #print(fitxer['name'],fitxer['format'])
                    
                    try:
                        filter_list.append(pd.read_csv(fitxer['url']))
                        #print(pd.read_csv(fitxer['url']).shape)
                    except:
                        try:
                            filter_list.append(pd.read_csv(fitxer['url'],encoding='ISO-8859-15'))
                            #print(pd.read_csv(fitxer['url']).shape)
                        except:
                            try:
                                filter_list.append(pd.read_csv(fitxer['url'],encoding="ISO-8859-1"))
                                #print(pd.read_csv(fitxer['url']).shape)
                            except :
                                filter_list.append(pd.read_csv(fitxer['url'],sep=';',encoding='ISO-8859-1'))
                    

    try:
        return filter_list
    except:
        print('NO FILE WITH THAT FILTER')
   





def utmToLatLng(zone, easting, northing, northernHemisphere=True):
    
   
    
    if not northernHemisphere:
        northing = 10000000 - northing

    a = 6378137
    e = 0.081819191
    e1sq = 0.006739497
    k0 = 0.9996
    
    
    arc = northing / k0
    mu = arc / (a * (1 - math.pow(e, 2) / 4.0 - 3 * math.pow(e, 4) / 64.0 - 5 * math.pow(e, 6) / 256.0))

    ei = (1 - math.pow((1 - e * e), (1 / 2.0))) / (1 + math.pow((1 - e * e), (1 / 2.0)))

    ca = 3 * ei / 2 - 27 * math.pow(ei, 3) / 32.0

    cb = 21 * math.pow(ei, 2) / 16 - 55 * math.pow(ei, 4) / 32
    cc = 151 * math.pow(ei, 3) / 96
    cd = 1097 * math.pow(ei, 4) / 512
    phi1 = mu + ca * math.sin(2 * mu) + cb * math.sin(4 * mu) + cc * math.sin(6 * mu) + cd * math.sin(8 * mu)

    n0 = a / math.pow((1 - math.pow((e * math.sin(phi1)), 2)), (1 / 2.0))

    r0 = a * (1 - e * e) / math.pow((1 - math.pow((e * math.sin(phi1)), 2)), (3 / 2.0))
    fact1 = n0 * math.tan(phi1) / r0

    _a1 = 500000 - easting
    dd0 = _a1 / (n0 * k0)
    fact2 = dd0 * dd0 / 2

    t0 = math.pow(math.tan(phi1), 2)
    Q0 = e1sq * math.pow(math.cos(phi1), 2)
    fact3 = (5 + 3 * t0 + 10 * Q0 - 4 * Q0 * Q0 - 9 * e1sq) * math.pow(dd0, 4) / 24

    fact4 = (61 + 90 * t0 + 298 * Q0 + 45 * t0 * t0 - 252 * e1sq - 3 * Q0 * Q0) * math.pow(dd0, 6) / 720

    lof1 = _a1 / (n0 * k0)
    lof2 = (1 + 2 * t0 + Q0) * math.pow(dd0, 3) / 6.0
    lof3 = (5 - 2 * Q0 + 28 * t0 - 3 * math.pow(Q0, 2) + 8 * e1sq + 24 * math.pow(t0, 2)) * math.pow(dd0, 5) / 120
    _a2 = (lof1 - lof2 + lof3) / math.cos(phi1)
    _a3 = _a2 * 180 / math.pi

    latitude = 180 * (phi1 - fact1 * (fact2 + fact3 + fact4)) / math.pi

    if not northernHemisphere:
        latitude = -latitude

    longitude = ((zone > 0) and (6 * zone - 183.0) or 3.0) - _a3

    return latitude, longitude
##correcting graphics of the letters. Having different accents and definitions is better if we get all names the same

def posant_accents(it):

    if type(it) == str:

        ##lletra ç

        if '\x87' in it:
            nova = it.replace('\x87','ç')
        elif 'Ă§' in it:
            nova = it.replace('Ă§','ç')
        
        else:
            nova = it

        ###lletra i
        if '¡' or 'Ã¯'or 'ï'or 'Ã\xad'or 'í' or '83Â\xad' or 'ÃÂ' in nova:
            renova = nova.replace('¡', 'i').replace('Ã¯', 'i').replace('ï', 'i').\
            replace('Ã\xad', 'i').replace('í', 'i').replace('83Â\xad','i').replace('ÃÂ','i')
        else:
            renova = nova
        ###Lletra o
        if '\x95' or 'Ã³' or '¢' or 'ã³'or 'Ã²' or 'ò' or 'x83Â³' or '83Â²' or 'ÃÂ³' in renova:
            trinova = renova.replace('\x95', 'o').replace('Ã³', 'o')\
            .replace('¢', 'o').replace('ã³', 'o').replace('Ã²', 'o').replace('ò', 'o').replace('x83Â³','o').replace('83Â²','o').replace('ÃÂ³','o')
        else:
            trinova = renova

        if '¢' in trinova:

            quatrinova = trinova.replace('¢', 'o')
        else:
            quatrinova = trinova

        ###lletra e
        if '\x82' or 'Ã©' or 'é' or 'è'or 'Ãš' or '\x8a' or 'č' or 'x83Â©' in quatrinova:
            cinquinova = quatrinova.replace('\x82', 'e').replace('Ã©', 'e').replace('é', 'e').replace('è', 'e').\
            replace('Ãš', 'e').replace('\x8a', 'e').replace('č','e').replace('x83Â©','e')
        else:
            cinquinova = quatrinova
          
        ###lletra a
        if '\x85' or 'Ã\xa0' or '' or '83Â\xa0' or 'ÃÂ' or 'ÃÂ ' in cinquinova:
            sixinova = cinquinova.replace('\x85', 'a').replace('Ã\xa0', 'a').replace('à', 'a').replace('83Â\xa0','a').replace('ÃÂ','a').replace('ÃÂ ','a')
        else:
            sixinova = cinquinova

        if 'Sarr' in sixinova:

            septinova = 'Sarria'
        else:
            septinova = sixinova

        if 'Ã§' in septinova:
            vuitinova = septinova.replace('Ã§', 'ç')
        else:
            vuitinova = septinova

        ##Lletra u
        if 'ãº' or 'ú' or '£'in vuitinova:
            nounova = vuitinova.replace('ãº', 'u').replace('ú', 'u').replace('£','u')
        else:
            nounova = vuitinova

        if 'ã³' or 'ó' in nounova:

            deunova = nounova.replace('ã³', 'o').replace('ó', 'o')
        else:
            deunova = nounova
       
    else:
        deunova = it

    return deunova

####Causes a angles

def cause_to_angles(it):

    cause_dict={'Alcoholemia': 'DrunkDriving',
               'Calçada en mal estat': 'Damaged_road',
               'Drogues o medicaments': 'DUI',
               'Estat de la senyalitzacio': 'Damaged_signal',
               'Exces de velocitat o inadequada': 'Speeding',
               'Factors meteorologics': 'Weather',
                'Objectes o animals a la calçada': 'Objects or animals on the road',
                'No hi ha causa mediata': 'No mediate cause'}


    return cause_dict[it]

##transalting to Catalan

def traduir_castella(word):
    if type(word) == str:
        if word.endswith('ismo'):
            nova = word.replace('ismo', 'isme')
        else:
            nova = word
        if 'ciclo' in nova:
            renova = nova.replace('ciclo', 'cicle')
        else:
            renova = nova

        if renova.startswith('Cuadri'):
            trinova = renova.replace('Cuadri', 'Quadri')
        else:
            trinova = renova

        if trinova.startswith('Camion'):

            quatrinova = trinova.replace('Camion', 'Camio rigid')

        else:
            quatrinova = trinova

        if quatrinova.endswith('  camion'):

            cinquinova = quatrinova.replace('camion', 'camio')
        else:
            cinquinova = quatrinova

        if 'Tm'in cinquinova:
            sixinova = cinquinova.replace('Tm', 'tones')
        else:
            sixinova = cinquinova
        if '75cc' in sixinova:
            septinova = sixinova.replace('75cc', ' 75 cc')
        else:
            septinova = sixinova
        if '> 75' in septinova:
            octinova = septinova.replace('> 75', '>= 75')
        else:
            octinova = septinova
        if octinova == 'Tranvia o tren':
            noninova = 'Tren o tramvia'
        else:
            noninova = octinova
        if 'de obras' in noninova:
            nova2 = noninova.replace('de obras', "d'obres i serveis")
        else:
            nova2 = noninova

        if 'Otros' or 'terreno' or 'articulado' or 'vehic. a' or 'Todo' or '17 plazas' in nova2:

            nova3 = nova2.replace('Otros', 'Altres').replace('terreno', 'terreny').\
            replace('articulado', 'articulat').replace('vehic. a', 'vehicles amb').replace('Todo', 'Tot')\
            .replace('17 plazas',' 17')
        else:
            nova3 = nova2
        if nova3 == 'Tractocamion':
            nova4 = "Tractor camio"
        else:
            nova4 = nova3
    else:
        nova4 = word

    return nova4

def ped_to_angles(it):

    ped_dict={'Desconegut': 'unknown',
             'Creuar per fora pas de vianants': 'Crossing outside ped crossing',
             'Desobeir el senyal del semàfor': 'Disobey the traffic light',
             'Transitar a peu per la calçada': 'Walk on the road',
             'Altres': 'Other',
             'Desobeir altres senyals': 'Disobey other signals',
             'No es causa del  vianant': 'No peds fault',
             'No és causa del  vianant': 'No peds fault'}

    return ped_dict.get(it,'NotFound')


def setmana_a_angles(dia):

    dia_dict={'Dilluns': 'Monday',
             'Dimarts': 'Tuesday',
             'Dimecres': 'Wednesday',
             'Dijous': 'Thursday',
             'Divendres': 'Friday',
             'Dissabte': 'Saturday',
             'Diumenge': 'Sunday',}
    return dia_dict.get(dia,'NotFound')


def mes_a_angles(mes):

    mes_dict={'Gener': 'January',
             'Febrer': 'February',
             'Març': 'March',
             'Abril': 'April',
             'Maig': 'May',
             'Juny': 'June',
             'Juliol': 'July',
             'Agost': 'August',
             'Setembre': 'September',
             'Octubre': 'October',
              'Novembre':'November',
              'Desembre':'December'
             }

    return mes_dict.get(mes,'NotFound')

def mes_english_number(mes):

    mes_dict={'January': '01',
             'February': '02',
             'March': '03',
             'April': '04',
             'May': '05',
             'June': '06',
             'July': '07',
             'August': '08',
             'September': '09',
             'October': '10',
              'November':'11',
              'December':'12'}

    return mes_dict[mes]

def counting_non_zeros(tup):
    count = 0
    for i in tup:
        if i > 0:
            count+=1
    if count == 0:
        count = 1
    return count

def debugging_strings(row, word):
    word = str(word)
    count = 0
    for i in row:
        if word in i:
            count =1
    return count

def mercedes(word):
    """Corregir tots els mercedes"""
    if word in ['mercedes-benz', 'mercedesb', 'mecedes']:
        word = 'mercedes'
    return word


def licenses(license):

    license_dict={'A': 'motorbike_license',
             'BTP': 'taxis_ambulances_license',
             'B': 'regular_license',
             'D': 'bus_license',
             'C': 'van_license'}

    return license_dict[license]

def fixing_codes(i):

    if i in desconegut_llista:
        i = int(-1)
    elif type(i) == float:
        i = int(i)

    elif (type(i) == str) and len(i) > 4:
        i = int(''.join(i.split("-", 2)[2:]))
    elif (type(i) == str) and len(i) <= 4:
        i = int(''.join(i.split('.')[0]))

    else:
        i = int(i)
    return i


def scraping_weather(date):
    url=f'https://www.meteo.cat/observacions/xema/dades?codi=D5&dia={date}T13:00Z'
    df = pd.read_html(url)[-1]
    df.rename(columns=renaming_columns,inplace=True)
    df['date']=date
    return df


def creating_datetime(row):
    full_datetime=row['date']+' '+row['period_UT'][:5]
    return full_datetime


def creating_yearly_weather(year,pathname):
    if f'weatherbarcelona{year}.csv' in os.listdir(pathname):
        weather_year=pd.read_csv(pathname+f'weatherbarcelona{year}.csv')
        print(f"Done with {year}")
        
    else:
        weather_year=pd.DataFrame()
        dates=pd.date_range(start=str(year)+'-01-01', end=str(year)+'-12-31')
        count=0
        for date in dates:
            #print(date)
            df=scraping_weather(str(date)[:10])
            df['datetime']=pd.to_datetime(df.apply(creating_datetime,axis=1))
            df.drop(['date','period_UT'],axis=1,inplace=True)
            count+=1
    
            if count%90==0:
                print(date)
                time.sleep(3)
            
            if weather_year.empty:
                weather_year=df
            else:
                weather_year=pd.concat([weather_year,df])
    
        weather_year.reset_index().to_csv(f'./data/weather/weatherbarcelona{year}.csv',index=False)
        print(f"Done with {year}")
    
    if 'weatherfinal.csv' in os.listdir(pathname):
        weather=pd.read_csv(pathname+'weatherfinal.csv',low_memory=False)
        weather['datetime']=pd.to_datetime(weather.datetime,utc=True,format='mixed',yearfirst=True)
        if year not in list(weather.datetime.dt.year):
            weather=pd.concat([weather,weather_year])
            weather.to_csv(pathname+'weatherfinal.csv',index=False)
    else:
        weather_year.to_csv(pathname+'weatherfinal.csv',index=False)

#mapping_vehicles


map_vehicles={'Motocicleta': 'motorcycle',
'Ciclomotor': 'moped',
'Turismo': 'car',
'Furgoneta':'van',
'Bicicleta':'bicycle',
'Taxi':'taxi',
'Tranvía o tren':'tram',
'Autobús':'bus',
'Cuadriciclo >=75cc':'quadricycle over 75cc',
'Camión <= 3,5 Tm':'truck under 3.5 tons',
'Microbus <=17 plazas': 'minibus <17 pass',
'Camión > 3,5 Tm':'truck over 3.5 tons',
'Autobús articulado':'articulated bus',
'Tractocamión':'tractor-truck',
'Todo terreno':'suv',
'Cuadriciclo <75cc':'quadricycle under 75cc',
'Otros vehíc. a motor':'other motor vehicles',
'Autocar':'bus',
'Maquinaria de obras':'construction machinery',
'Autob£s':'bus',
'Otros veh¡c. a motor':'other motor vehicles',
'Cami¢n <= 3,5 Tm':'truck under 3.5 tons',
'Autob£s articulado':'articulated bus',
'Cami¢n > 3,5 Tm':'truck over 3.5 tons',
'Tranv¡a o tren':'tram',
'Tractocami¢n':'tractor-truck',
'Autocaravana':'camper',
'Turisme':'car',
'Autobús articulat':'articulated bus',
'Altres vehicles sense motor':'other non-motor vehicles',
'Camió rígid <= 3,5 tones':'truck under 3.5 tons',
'Altres vehicles amb motor':'other engine vehicles',
'Quadricicle > 75 cc':'quadricycle over 75cc',
'Camió rígid > 3,5 tones':'truck over 3.5 tons',
'Tren o tramvia':'tram',
'Maquinària d"obres i serveis':'construction machinery',
'Tractor camió': 'tractor-truck',
'Tot terreny':'suv',
'Quadricicle < 75 cc':'quadricycle under 75cc',
'Desconegut':'unknown',
'Microbus <= 17': 'minibus <17 pass',
'Veh. mobilitat personal amb motor': 'personal motor vehicles',
'Veh. mobilitat personal sense motor':'personal non-motor vehicles',
'Microbús <= 17': 'minibus <17 pass',
'Carro':'wagon',
'Pick-up':'van',
"Maquinŕria d'obres i serveis":'construction machinery',
"Maquinària d'obres i serveis":'construction machinery',
'Ambulŕncia': 'ambulance'}

def organizing_types(string,type_list):
    set_string=set(string.split(','))
    if len(set_string)==1:
        return str(set_string)[2:-2]
    else:
        for ty in type_list:
            if ty in set_string:
                set_string.remove(ty)
                if len(set_string)==1:
                      return str(set_string)[2:-2]
        
    return set_string


def remove_accents(word):
    
    return word.replace('à', 'a').replace('è','e').replace('é', 'e').replace('ï', 'i').replace('í', 'i').replace('ò','o').replace('ó', 'o').replace('ü','u').replace('ú', 'u')


def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if (a_set & b_set): 
        print(a_set & b_set) 
    else: 
        print("No common elements")
        
def common_member_and_number(a,b):
    a_set = set(a) 
    b_set = set(b) 
    llista = []
    if (a_set & b_set): 
        llista = list(a_set & b_set) 
    return llista
def set_of_colors(n):
    color = []
    for i in range(n):
        color.append('#%06X' % random.randint(0, 0xFFFFFF))    
    return color

def random_color_generator(n):
    colors=[]
    for i in range(n):
        color = np.random.randint(0, 256, size=3)
        colors.append(tuple(color))
    return colors

def fixing_seniority(string):
    """Will replace Desconegut with null"""
    string=string.split(',')
    return string