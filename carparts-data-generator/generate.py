import pandas as pd
import numpy as np
import random
import plotly.express as px
import argparse
import requests


class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end


class carpartsDataGenerator():
    def __init__(self, path):
        self.path = path

    def __readDataset(self):
        self.data = pd.read_csv(self.path).fillna(0).to_numpy()

    def __generateData(self,serie_index):
        for j in range(self.lenght):
            i_rand = random.randint(0,self.data.shape[0]-1)
            j_rand = random.randint(0,self.data.shape[1]-1)
            
            self.generated_dataset[serie_index,j] = int(self.data[i_rand,j_rand] / 1e6)

    def __addNoise(self):
        
        for i in range(self.n_series):
            for j in range(self.lenght):
                self.generated_dataset[i,j] += int(np.random.normal(0, self.noise_factor*(self.generated_dataset[i,j] + 8)/3, 1)[0])
                if self.generated_dataset[i,j] < 0.0:
                    self.generated_dataset[i,j] = 0.0

    def generateDataset(self, n_series, lenght, noise_factor):
        
        self.n_series = n_series
        self.lenght = lenght
        self.noise_factor = noise_factor

        self.__readDataset()
        self.generated_dataset = np.zeros((n_series,lenght))
        for i in range(n_series):
            self.__generateData(i)

        self.__addNoise()

        return self.generated_dataset


    def writeDataset(self):
        
        headers = list(map(str,np.arange(self.generated_dataset.shape[0])))
        values = self.generated_dataset
        
        items = []
        for i in range(len(headers)):

            values_str=""
            for value in values[i]:
                values_str+="{},".format(value)
            values_str = values_str[:-1]

            items.append({
                        "data_group": "Sells",
                        "y_axis": "monthly_sells",
                        "tag": "series_id",
                        "tag_value": headers[i],
                        "values": values_str
                        })

        url = "http://0.0.0.0:8000/write/"
        for item in items:
            response = requests.get(url,item)
            print(response.content)
        
        




if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_series", help="Cantidad de series a generar",type=int,default=10,choices=range(1,101))
    parser.add_argument("--lenght", help="Tamaño de las series a generar",type=int,default=1000,choices=range(1,60001))
    parser.add_argument("--noise_factor", help="Tamaño de las series a generar",type=float,default=0.0,choices=[Range(0.0, 1.0)])
    parser.add_argument("--plot_list", help="Tamaño de las series a generar",type=int,nargs="+",default=[])
    parser.add_argument("--write_influx", help="Escribe datos en influx",type=int,default=0,choices=[0,1])
    args = parser.parse_args()


    dataset_path = "./carparts.csv"
    generator = carpartsDataGenerator(dataset_path)
    

    generated_data = generator.generateDataset(args.n_series,args.lenght,args.noise_factor)

    data_dict = {}

    for i in range(args.n_series):

        data_dict["serie{}".format(i)] = generated_data[i,:]

        if i in args.plot_list:
            tmp_df = pd.DataFrame(dict(
                x = np.arange(generated_data.shape[1]),
                y = generated_data[i,:]
            ))
            fig = px.line(tmp_df, x="x", y="y", title="Unsorted Input") 
            fig.show()

    df = pd.DataFrame.from_dict(data_dict).to_csv("series.csv")

    if args.write_influx:
        generator.writeDataset()


