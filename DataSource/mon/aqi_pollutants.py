from enum import Enum, unique

# 可在这个文件中，改为用本地监测因子的id

class AqiPollutants(Enum):
    primary=2001
    aqi_level=2002
    aqi_type=2003
    aqi=2004

O3_CID = 286
NO2_CID = 386
CO_CID = 486
SO2_CID = 586
PM10_CID = 8086
PM25_CID = 8088

IAQI_POLLUTANTS = {
    O3_CID: {'name':'O3分指数','value':28666},
    NO2_CID:{'name':'NO2分指数','value':38666},
    CO_CID:{'name':'CO分指数','value':48666},
    SO2_CID:{'name':'SO2分指数','value':58666},
    PM10_CID:{'name':'PM10分指数','value':80866},
    PM25_CID:{'name':'PM25分指数','value':80888},
}