# Class LoadData
    This class comprises of different functions to load different datasets.
    Initializing the class does not require any parameters.
    It can simply be done as:
        - ld = LoadData()
    This returns nothing but prints out available datasets.
    
### Available functions
    load_forge: requires no parameter and loads forge data from mglearn and returns a tuple X, y
        Optional parameter visualize - default:False used to visualize the data

    load_wave: requires no parameter and loads the wave data from mglearn and returns a tuple X, y
        Optional parameter n_sample - select sample size to generate
        Optional parameter visualize - default:False used to visualize the data

    load_cancer: requires no parameter and loads the breast cancer data from sklearn and returns a tuple 
                 containing X, y, feature names and target names

    load_boston: requires no parameter.
        By default returns the extended dataset of boston housing prices in the form of a tuple containing X and y
        To return the unextended version, specify 
        - extended=False as a parameter
        
