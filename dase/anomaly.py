import pandas as pd
import random
from scipy.stats import uniform
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import itertools
from abc import ABC, abstractmethod

random.seed(329)
np.random.seed(329)


def zeros_serie(data : pd.Series) -> (np.array, np.array):
    
    zero_count = 0
    zero_data_values = []
    zeros_end_indexes = []
    for index,value in enumerate(data):
        if value == 0:
            zero_count+=1
        else:
            zeros_end_indexes.append(index)
            zero_data_values.append(zero_count)
            zero_count=0

    return np.array(zero_data_values), np.array(zeros_end_indexes)


def disjoint(a : list, b : list) -> bool:
    for item in a:
        for listb in b:
            if item in listb:
                return False
    return True


def empty_union(a : list, b : list) -> bool:
    for item in a:
        if item in b:
            return False
    return True


def peaks_serie(data : pd.Series) -> (list, list):
    
    peaks_data_values = []
    peaks_data_indexes = []
    for index,value in enumerate(data):
        if value != 0:
            peaks_data_values.append(value)
            peaks_data_indexes.append(index)

    return np.array(peaks_data_values), np.array(peaks_data_indexes)




class AnomalyGenerator(ABC):

    @abstractmethod
    def generate(self, data: pd.Series) -> (pd.Series, pd.Series):
        pass

    def __call__(self, data: pd.Series) -> (pd.Series, pd.Series):
        return self.generate(data)



class TrainAnomalyGenerator(AnomalyGenerator):

    def __init__(self):
        pass

    def generate(self, data: pd.Series) -> (pd.Series, pd.Series):


        if len(data) <= 1000:
            
            wd_size = max(1, int(len(data)*0.02))
            wait_size = max(1, int(len(data)*0.01))
            demand_size = max(1, int(len(data)*0.015))
            activity_size = max(1, int(len(data)*0.01))

        elif len(data) > 1000 and len(data) <= 5000:
            wd_size = max(1, int(len(data)*0.02))
            wait_size = max(1, int(len(data)*0.005))
            demand_size = max(1, int(len(data)*0.015))
            activity_size = max(1, int(len(data)*0.005))

        else:
            wd_size = max(1, int(len(data)*0.02))
            wait_size = max(1, int(len(data)*0.002))
            demand_size = max(1, int(len(data)*0.015))
            activity_size = max(1, int(len(data)*0.002))

        indexes_to_ignore = []

        wd_anomaly = WDAnomalyGenerator( wd_size )
        anomalies, anomalies_labels = wd_anomaly(data)
        indexes_to_ignore.extend( [i for i in range(len(data)) if anomalies_labels.iloc[i] ] )
        labels = anomalies_labels
        
        wait_anomaly = WaitAnomalyGenerator( wait_size , indexes_to_ignore)
        anomalies, anomalies_labels = wait_anomaly(anomalies)
        indexes_to_ignore.extend( [i for i in range(len(data)) if anomalies_labels.iloc[i] ] )
        labels |= anomalies_labels

        demand_anomaly = DemandAnomalyGenerator(demand_size, indexes_to_ignore)
        anomalies, anomalies_labels = demand_anomaly(anomalies)
        indexes_to_ignore.extend( [i for i in range(len(data)) if anomalies_labels.iloc[i] ] )
        labels |= anomalies_labels 
        
        a_generator = ActivityAnomalyGenerator(activity_size, indexes_to_ignore)
        anomalies, anomalies_labels = a_generator(anomalies)
        labels |= anomalies_labels 

        return anomalies, labels



#Anomalia de demanda
class DemandAnomalyGenerator(AnomalyGenerator):

    def __init__(self, size: float, indexes_to_ignore : list):
        if isinstance(size, float) and not 0 < size <= 1:
            raise ValueError('Size must be a float between 0 and 1')

        if isinstance(size, int) and size <= 0:
            raise ValueError('Size must be a positive integer')

        self.size: int | float = size
        self.indexes_to_ignore = indexes_to_ignore
        
    def generate(self, data: pd.Series) -> (pd.Series, pd.Series):
        
        data = data.copy()
        alpha = max(data)
        total = self.size if isinstance(self.size, int) else int(self.size * len(data))

        peak_indexes = np.array([i for i in np.where(data != 0)[0] if i not in self.indexes_to_ignore])
        if len(peak_indexes) > total:
            peak_indexes = peak_indexes[random.sample(range(len(peak_indexes)),total)]

        a = 1.05
        b = 1.4
        u = uniform(loc=alpha*a, scale=alpha*(b-a))  #alpha * U(a,b)
        
        anomalies_indexes = []
        for i in range(min(total,len(peak_indexes))):
            m = peak_indexes[i]
            anomalies_indexes.append(m)
            data.iloc[m] = u.rvs(size = 1)

        labels = pd.Series(index=data.index, data=0, name='anomaly', dtype=int)
        labels[anomalies_indexes] = 1

        return data, labels


#Anomalia de espera
class WaitAnomalyGenerator(AnomalyGenerator):

    def __init__(self, size: float, indexes_to_ignore : list):
        if isinstance(size, float) and not 0 < size <= 1:
            raise ValueError('Size must be a float between 0 and 1')

        if isinstance(size, int) and size <= 0:
            raise ValueError('Size must be a positive integer')

        self.size: int | float = size
        self.indexes_to_ignore = indexes_to_ignore
        
    def generate(self, data : pd.Series) -> (pd.Series, pd.Series):
        
        data = data.copy()
        total = self.size if isinstance(self.size, int) else int(self.size * len(data))
        
        zero_data_values, zeros_end_indexes = zeros_serie(data)
        max_zeros_lenght = max(zero_data_values)
        total_zeros = sum(zero_data_values)
        
        anomaly_zeros_indexes = []
        for i in zip(range(min(total, int(total_zeros / max_zeros_lenght) - 3))):
            
            zeros_cumsum = 0
            itercont = 0
            while zeros_cumsum < max_zeros_lenght and itercont < 1000:
                
                aux_zero_indexes = []
                start_index = random.sample( range(len(zeros_end_indexes)), 1)[0]
                zeros_cumsum = 0
                for offset, zeros in enumerate(zero_data_values[start_index:]):
                    
                    if start_index + offset in self.indexes_to_ignore:
                        break
                    
                    aux_zero_indexes.append(start_index + offset)
                    zeros_cumsum += zeros
                    if zeros_cumsum > max_zeros_lenght:
                        if not disjoint(aux_zero_indexes, anomaly_zeros_indexes):
                            zeros_cumsum = 0
                            break
                        else:
                            anomaly_zeros_indexes.append(aux_zero_indexes)
                            break
                itercont+=1

        anomalies_indexes = []
        for indexes in anomaly_zeros_indexes:
            

            data[ list(range(sum(zero_data_values[:indexes[0]]) + len(zero_data_values[:indexes[0]]), sum(zero_data_values[:indexes[-1]]) + len(zero_data_values[:indexes[-1]]) )) ]  = 0

            offset = zero_data_values[indexes[-1]] - ( sum(zero_data_values[indexes]) - max_zeros_lenght )
            original_index = sum(zero_data_values[:indexes[-1]]) + len(zero_data_values[:indexes[-1]])
            anomalies_indexes.extend([i + original_index for i in range(offset, zero_data_values[indexes[-1]]) ])
        
        anomalies_indexes = sorted(anomalies_indexes)
        labels = pd.Series(index=data.index, data=0, name='anomaly', dtype=int)
        labels[anomalies_indexes] = 1
        
        data[anomalies_indexes] = 0

        return data, labels


#Anomalia de espera-demanda
class WDAnomalyGenerator(AnomalyGenerator):

    def __init__(self, size: float):
        if isinstance(size, float) and not 0 < size <= 1:
            raise ValueError('Size must be a float between 0 and 1')

        if isinstance(size, int) and size <= 0:
            raise ValueError('Size must be a positive integer')

        self.size: int | float = size
        
    def generate(self, data : pd.Series) -> (pd.Series, pd.Series):
        
        data = data.copy()
        total = self.size if isinstance(self.size, int) else int(self.size * len(data))

        peaks_data_values, peaks_data_indexes = peaks_serie(data)
        zero_data_values, zeros_end_indexes = zeros_serie(data)
        zero_data_values_normalized = (zero_data_values - zero_data_values.min()) / (zero_data_values.max() - zero_data_values.min())
        q = np.stack([zero_data_values_normalized, peaks_data_values], axis=1)
        
        max_score = 0
        n_clusters = 0
        for n in range(2,min(len(q),20)):
            clusterer = KMeans(n_clusters=n, random_state=10)
            cluster_labels = clusterer.fit_predict(q)
            silhouette_avg = silhouette_score(q, cluster_labels)
            if silhouette_avg > max_score:
                max_score = silhouette_avg
                n_clusters = n

        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(q)
        cluster_centers = clusterer.cluster_centers_
        max_distance_to_center = defaultdict(int)
        for i, label in enumerate(cluster_labels):
            distance = np.sqrt(sum((q[i] - cluster_centers[label]) ** 2))
            if distance > max_distance_to_center[label]:
                max_distance_to_center[label] = distance

        peaks_range = np.linspace(0.1,1,100)
        new_peak = 0
        anomaly_points_values= []
        anomaly_points_indexes = []
        itercont = 0
        while len(anomaly_points_indexes) < min(total, len(q) - 3) and itercont < 1000:
            point_index = random.sample(range(len(q)), 1)[0]
            if q[point_index][0] != 0:
                for peak in peaks_range:
                    valid = True
                    for label in cluster_labels:
                        if np.sqrt((q[point_index][0] - cluster_centers[label][0]) ** 2 + (peak - cluster_centers[label][1]) ** 2 ) <= max_distance_to_center[label]:
                            valid = False
                            break
                    if valid:
                        valid_peaks = np.linspace(peak,1,1000)
                        new_peak = valid_peaks[random.sample(range(1000), 1)[0]]
                        anomaly_points_values.append(new_peak)
                        anomaly_points_indexes.append(point_index)
                        break    
                
                if not valid:
                    itercont += 1  
            else:
                itercont += 1        

        anomalies_indexes = []
        for n, index in enumerate(anomaly_points_indexes):
            start_index = sum(zero_data_values[:index]) + len(zero_data_values[:index])
            end_index = sum(zero_data_values[:index+1]) + len(zero_data_values[:index+1])
            data[end_index-1] = anomaly_points_values[n]
            anomalies_indexes.extend([i for i in range(start_index, end_index)])

        anomalies_indexes = sorted(anomalies_indexes)
        labels = pd.Series(index=data.index, data=0, name='anomaly', dtype=int)
        labels[anomalies_indexes] = 1

        return data, labels


#Anomalia de actividad
class ActivityAnomalyGenerator(AnomalyGenerator):

    def __init__(self, size: float, indexes_to_ignore : list):
        if isinstance(size, float) and not 0 < size <= 1:
            raise ValueError('Size must be a float between 0 and 1')

        if isinstance(size, int) and size <= 0:
            raise ValueError('Size must be a positive integer')

        self.size: int | float = size
        self.indexes_to_ignore = indexes_to_ignore

    def generate(self, data: pd.Series) -> (pd.Series, pd.Series):
        
        data = data.copy()
        total = self.size if isinstance(self.size, int) else int(self.size * len(data))
        
        anomalies_windows_indexes = []
        anomalies_windows_values = []
        itercont = 0
        while len(anomalies_windows_indexes) < min(total, int(len(data)*0.01) - 3) and itercont < 1000:

            try:
                window_size = random.sample( range(5,int(len(data)*0.01) + 1), 1)[0]
            except:
                break
            window = 1 + np.blackman(window_size)
            start = np.random.randint(low=0, high=len(data) - window_size)
            indexes = list(range(start,start + window_size))

            if not empty_union(indexes, self.indexes_to_ignore) or not disjoint(indexes, anomalies_windows_indexes):
                itercont += 1
            else:
                anomalies_windows_indexes.append(indexes)
                anomalies_windows_values.append( list(np.random.uniform(low=np.random.uniform(low=0.01,high=0.1), high=np.random.uniform(low=0.1,high=0.25)) * window ) )

            

        anomalies_windows_indexes = list(itertools.chain(*anomalies_windows_indexes))
        anomalies_windows_values = list(itertools.chain(*anomalies_windows_values))

        data[anomalies_windows_indexes] = anomalies_windows_values

        labels = pd.Series(index=data.index, data=0, name='anomaly', dtype=int)
        labels[anomalies_windows_indexes] = 1

        return data , labels




