import pandas as pd
from rhodium import *

# use pandas to read the csv file
frame = pd.read_csv("lakeset.csv")

# convert the pandas data frame to a Python dict in record format
dictionary = frame.to_dict('records')

# create a Rhodium DataSet instance from the Python dictionary
dataset = DataSet(dictionary)

policy = dataset.find_min('concentration')
print(policy)

# define a trivial "dummy" model in Rhodium with an arbitrary function
model = Model(lambda x: x)

# set up the model's objective responses to match the keys in your dataset
# here, all objectives are minimized
# this is the only information needed to create a parallel coordinate plot
model.responses = [Response("benefit", Response.MINIMIZE),
                   Response("concentration", Response.MINIMIZE),
                   Response("inertia", Response.MINIMIZE),
                   Response("reliability", Response.MINIMIZE)]

# create the parallel coordinate plot from the results of our external optimization
fig = parallel_coordinates(model, dataset, target="bottom",
                           brush=[Brush("reliability < -0.95"), Brush("reliability >= -0.95")])
fig.savefig('parallel.png')
