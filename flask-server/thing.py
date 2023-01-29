import pandas as pd
import numpy as np
from datetime import datetime
from IPython.display import Image
from IPython.display import HTML
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from datetime import datetime as dt

import arcgis
from arcgis.gis import GIS
# from arcgis.learn import MLModel, prepare_tabulardata
from arcgis.raster import Raster

from fastai.vision import *

gis = GIS("home")   
gis_enterp = GIS("https://pythonapi.playground.esri.com/portal", "arcgis_python", "amazing_arcgis_123")

# get image
s2 = gis.content.get('fd61b9e0c69c4e14bebd50a9a968348c')
sentinel = s2.layers[0]
s2

#  extent in 3857 for amazon rainforest
amazon_extent = {
    "xmin": -6589488.51,
    "ymin": -325145.08,
    "xmax": -6586199.09,
    "ymax": -327024.74,
    "spatialReference": {"wkid": 3857}
}

# The respective scene having the above area is selected
selected = sentinel.filter_by(where="(Category = 1) AND (cloudcover <=0.05)",
                              geometry=arcgis.geometry.filters.intersects(amazon_extent))
df = selected.query(out_fields="AcquisitionDate, GroupName, CloudCover, DayOfYear",
                   order_by_fields="AcquisitionDate").sdf

df['AcquisitionDate'] = pd.to_datetime(df['acquisitiondate'], unit='ms')
df

amazon_scene = sentinel.filter_by('OBJECTID=1584818')
amazon_scene.extent = amazon_extent
amazon_scene

amazon_scene.save('amazon_scene'+ str(dt.now().microsecond), gis=gis_enterp)

raster_amazon_13bands = Raster("https://pythonapi.playground.esri.com/ra/rest/services/Hosted/amazon_scene_may26/ImageServer",
                               gis=gis_enterp,
                               engine="image_server")

raster_amazon_13bands.export_image(size=[3330,1880])

pd.DataFrame(amazon_scene.key_properties()['BandProperties'])

raster_amazon_13bands.name

preprocessors = [('Hosted/amazon_scene_may26_1', 
                  'Hosted/amazon_scene_may26_2', 
                  'Hosted/amazon_scene_may26_3',
                  'Hosted/amazon_scene_may26_7', MinMaxScaler())]

data = prepare_tabulardata(explanatory_rasters=[(raster_amazon_13bands,(1,2,3,7))], preprocessors=preprocessors)

data.show_batch()

model = MLModel(data, 'sklearn.cluster.KMeans', n_clusters=3, init='k-means++', random_state=43)

model.fit()

model.show_results()

pred_new = model.predict(explanatory_rasters=[raster_amazon_13bands],
                         prediction_type='raster',
                         output_layer_name=('deforest_predicted2'+str(dt.now().microsecond)),
                         output_raster_path=r"/tmp/result5.tif")

amazon_predict = gis.content.get('b81b89aac4cd4e08bcd7cd400fac558f')
amazon_predict

filepath_new = amazon_predict.download(file_name=amazon_predict.name)
with zipfile.ZipFile(filepath_new, 'r') as zip_ref:
    zip_ref.extractall(Path(filepath_new).parent)
output_path = Path(os.path.join(os.path.splitext(filepath_new)[0]))
output_path = os.path.join(output_path, "result5.tif") 

raster_predict = Raster(output_path)

raster_predict.export_image(size=[3290,1880])
