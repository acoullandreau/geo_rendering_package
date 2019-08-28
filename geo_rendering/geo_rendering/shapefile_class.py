import pandas as pd
import shapefile as shp

from .shape_class import ShapeOnMap


class ShapeFile:

    def __init__(self, file_path):
        self.path = file_path
        self.shapefile = self.sf_reader(self.path)
        self.df_sf = self.shp_to_df()
        self.shape_dict_sf = self.build_shape_dict(self.df_sf)

    def sf_reader(self, path):
        shapefile = shp.Reader(self.path)
        return shapefile

    def shp_to_df(self):
        sf = self.shapefile
        fields = [x[0] for x in sf.fields][1:]
        records = sf.records()
        shps = [s.points for s in sf.shapes()]
        df = pd.DataFrame(columns=fields, data=records)
        df = df.assign(coords=shps)
        return df

    def filter_shape_to_render(self, cond_stat, attr):
        # cond_stat in the form of a string or an array
        # attr in the form of a str, a column name of df

        try:
            if type(cond_stat) == str:
                filtered_df = self.df_sf[self.df_sf[attr] == cond_stat]
            elif type(cond_stat) == list:
                filtered_df = self.df_sf[self.df_sf[attr].isin(cond_stat)]
            else:
                raise AttributeError('Condition statement must be str or arr of values from the shapefile_dataframe')

        except:
            raise

        else:
            return filtered_df

    def build_shape_dict(self, ref_df):
        index_list = ref_df.index.tolist()
        shape_dict = {}
        for shape_id in index_list:
            shape = ShapeOnMap(self.shapefile, shape_id)
            shape_dict[shape_id] = shape

        return shape_dict
