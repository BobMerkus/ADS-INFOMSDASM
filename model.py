# import the campo package
import campo
from campo.phenomenon import Phenomenon
import pcraster as pcr
import pcraster.framework as pcrfw

# Environment class definition 
class EmissionEnvironment(pcrfw.StaticModel):
    def __init__(self):
        pcrfw.StaticModel.__init__(self)

    def initial(self):
        emission_env = campo.Campo()
        companies = emission_env.add_phenomenon("companies")
        companies.add_property_set("company", 'Data/company_xy.csv')
        companies.company.emission = 1000 #[x for x in range(0, companies.nr_agents)]
        
        nitrogen = emission_env.add_phenomenon("nitrogen")
        
        # foodstores.frontdoor.postal_code = 1234
        # foodstores.frontdoor.lower = -0.5
        # foodstores.frontdoor.upper = 0.5
        # foodstores.frontdoor.x_initial = campo.uniform(foodstores.frontdoor.lower, foodstores.frontdoor.upper)
        emission_env.create_dataset("Data/emission_environment.lue")
        emission_env.write()

# Runtime
if __name__ == '__main__':
    # Setup the environment
    emission_env = EmissionEnvironment()

    staticFrw = pcrfw.StaticFramework(emission_env)
    staticFrw.run()
    print(staticFrw)
    
    # Convert to pandas df
    import lue.data_model as ldm
    import campo
    # Open the dataset
    dataset = ldm.open_dataset('Data/emission_environment.lue')
    # Get the phenomenon
    dataframe = campo.dataframe.select(dataset.companies, property_names=['emission'])
    campo.to_csv(dataframe, 'company_export.csv')
    campo.to_gpkg(dataframe, 'company_export.gpkg', 'EPSG:28992')
        
    campo.create_field_pdf(dataframe, 'Data/company_xy.pdf')
    print(dataframe)
    print(type(dataframe))

        

# # set file directory
# tif_file = "Data/company_count_2020_1000x1000.tif"
# map_file = "Data/company_count_2020_1000x1000.map"

# # import
# ds = gdal.Open(tif_file)
# data = ds.ReadAsArray()

# # convert the numpy array to a PCRaster map
# map_data = pcr.numpy2pcr(pcr.Scalar, data)

# # write the map to a .map file
# pcr.report(map_data, map_file)

# pcr.numpy2pcr(dataType=pcr.Scalar, array = data, mv=0)

# emissionMap=pcr.readmap("Data/company_count_2020_1000x1000.map") 
# pcr.aguila(emissionMap)