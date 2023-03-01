import torch
import numpy as np
import pandas as pd
from dase.anomaly import TrainAnomalyGenerator
from dase.processor import NoiseFilter, MinMaxNormalizer
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from sklearn.metrics import f1_score, precision_score, recall_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Divide datos entre test y validacion para entrenamiento
"""
def split_data(data : pd.Series, test_size : float) -> (pd.Series, pd.Series, float):
    limit = int(len(data)*(1-test_size))
    return data.iloc[:limit], data.iloc[limit:], limit

"""
Clase que genera los datloaders a partir de dataframes con el fin de entrenar el modelo
"""
class TimeSeriesDataloader():
    
    def __init__(self, windows_size, batch_size,shuffle = True,drop_last=True):
        self.windows_size = windows_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def create(self, x : np.array, y : np.array):

        data = []
        labels = []
        for i in range(x.shape[0]-self.windows_size+1):
            data.append(x[i:i+self.windows_size,:])
            labels.append(y[i:i+self.windows_size])

        data = np.array(data)
        labels = np.array(labels)

        data = torch.Tensor(data)
        labels = torch.Tensor(labels)

        data = torch.reshape(data, (data.shape[0], data.shape[1], data.shape[2]))
        labels = torch.reshape(labels, (labels.shape[0], labels.shape[1],1))
        ds = torch.utils.data.TensorDataset(data,labels)
        dataloader = torch.utils.data.DataLoader(ds, batch_size = self.batch_size, shuffle = self.shuffle, drop_last=self.drop_last)
        return dataloader
        

"""
Clase de detector de anomalias basado en LSTM
model = LSTM_AD()
"""
class LSTM_AD(torch.nn.Module):
    

    #Una capa recurrente conectada a una densa
    def __init__(self, input_size=1, output_size=10, num_layers=1, dropout=0.0, predict_th=0.5):
        super(LSTM_AD, self).__init__()
        
        self.input_size = input_size
        self.recurrent_output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.th = predict_th

        self.recurrent = torch.nn.LSTM(input_size, output_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.dense = torch.nn.Linear(output_size,1)
    #
    def setHidden(self, hidden):
        self.hidden = hidden

    def __setHiperParams(self, hiperparams : dict):
        
        self.__init__(self.input_size, hiperparams["recurrent_output_size"], self.num_layers, self.dropout, hiperparams["th"])

        h0 = torch.zeros(1,  self.batch_size, self.recurrent_output_size)
        c0 = torch.zeros(1,  self.batch_size, self.recurrent_output_size)
        hidden = (h0,c0)
        self.setHidden(hidden)

    def forward(self, xb):
        x = self.recurrent(xb, self.hidden)
        return torch.sigmoid(self.dense(x[0]))
    

    """
    Recibe datos de un dataframe con una serie univariada (1 columna)
    Preprocesa los datos (filtra ruido mayor a media, normaliza)
    Agrega anomalias y genera etiquetas
    Optimiza hiperparametros del modelo
    Entrena separando datos en train y val llamando al metodo train
    Implementa early stop y se queda con el modelo de mejor resultado en val
    """
    def fit(self, df : pd.DataFrame):
        

        #Segun el tamaño de los datos determina la ventana, el tamaño del batch y se instancian los dataloaders
        if len(df) <= 200:
            self.windows_size = int(len(df)*0.15)
            self.batch_size = 2
        elif len(df) > 200 and len(df) <= 1000:
            self.windows_size = int(len(df)*0.1)
            self.batch_size = 4
        elif len(df) > 1000 and len(df) <= 5000:
            self.windows_size = int(len(df)*0.05)
            self.batch_size = 16
        else:
            self.windows_size = min(int(len(df)*0.02), 500)
            self.batch_size = 64

        self.fit_dataloader = TimeSeriesDataloader(self.windows_size,self.batch_size)
        self.detect_dataloader = TimeSeriesDataloader(self.windows_size,1, shuffle=False, drop_last=False)

        
        #preprocesamiento de datos
        self.normalizer, self.noise_filter = MinMaxNormalizer(), NoiseFilter(deviation_factor=0.0)
        column = list(df.keys())[0]
        data = df.copy()
        data = self.noise_filter.fitTransform(data,column,column)
        data = self.normalizer.fitTransform(data,column,column)

        #Generacion de anomalias
        generator = TrainAnomalyGenerator()
        anomaly_data, anomalies_indexes = generator(data[column])
        target = anomalies_indexes

        #Segun el tamaño de los datos utiliza mayor o menor proporcion en la optimizacion de hiperparametros
        if len(df) <= 1000:
            opt_data, _, limit = split_data(anomaly_data, 0.0)
        elif len(df) > 1000 and len(df) <= 5000:
            opt_data, _, limit = split_data(anomaly_data, 0.4)
        else:
            opt_data, _, limit = split_data(anomaly_data, 0.6)
        opt_labels = target[:limit]


        #Optimizacion de hiperparametros
        opt_train_data, opt_val_data, limit = split_data(opt_data, 0.33)
        opt_train_labels, opt_val_labels = opt_labels[:limit], opt_labels[limit:]
        opt_train_dataloader = self.fit_dataloader.create(opt_train_data.values.reshape(-1,1), opt_train_labels)
        opt_val_dataloader = self.fit_dataloader.create(opt_val_data.values.reshape(-1,1), opt_val_labels)
        trials = 30
        metric="f1"
        optimizer = LSTMOptimizer(trials, metric)
        best_hiperparams = optimizer.findBestModel(opt_train_dataloader, opt_val_dataloader)
        self.__setHiperParams(best_hiperparams)

        #Entreno con 75% train y 25% val
        train_data, val_data, limit = split_data(anomaly_data, 0.25)
        train_labels, val_labels = target[:limit], target[limit:]

        train_dataloader = self.fit_dataloader.create(train_data.values.reshape(-1,1), train_labels)
        val_dataloader = self.fit_dataloader.create(val_data.values.reshape(-1,1), val_labels)
        
        self.train(train_dataloader, val_dataloader, epochs=50, lr = best_hiperparams["lr"], scheduler_gamma = best_hiperparams["scheduler_gamma"])

        return 

    """
    Implementa el entranamiento de la red neuronal
    """
    def train(self, train_dataloader, val_dataloader,epochs=10, lr=5e-3, patience=10, scheduler_gamma=0.98):


        opt = torch.optim.RMSprop(self.parameters(),lr = lr)
        _ = torch.optim.lr_scheduler.ExponentialLR(opt, scheduler_gamma)

        history_train_loss = []
        history_val_loss = []
        early_stop_count = 0
        best_val_loss = 0
        for epoch in range(epochs):
            train_loss = 0
            val_loss = 0
            for train_batch ,train_targets in train_dataloader:

                preds = self(train_batch)
                train_loss += torch.nn.functional.binary_cross_entropy(preds,train_targets) 
            
            for val_batch, val_targets in val_dataloader:

                preds = self(val_batch)
                val_loss += torch.nn.functional.binary_cross_entropy(preds, val_targets) 

            train_loss /=  len(train_dataloader)
            val_loss /= len(val_dataloader)
            print("EPOCH: {}, train_loss={:.5f}, val_loss={:.5f}".format(epoch, train_loss, val_loss))

            history_train_loss.append(train_loss.clone().detach().numpy())
            history_val_loss.append(val_loss.clone().detach().numpy())

            if epoch == 0:
                best_val_loss = val_loss
                torch.save(self.state_dict(),"best_lstm.pth")
            else:
                if val_loss >= best_val_loss:
                    early_stop_count += 1
                else:
                    early_stop_count = 0
                    best_val_loss = val_loss
                    torch.save(self.state_dict(),"best_lstm.pth")

            if early_stop_count == patience:
                break
  
            opt.zero_grad()
            train_loss.backward()
            opt.step()

        state = torch.load("best_lstm.pth")
        self.load_state_dict(state)
    
        return 


    """
    Recibe datos de un dataframe con una serie univariada (1 columna)
    Preprocesa los datos (periodiza a 5min, filtra ruido mayor a media + 1*std, normaliza) con los parametros aprendidos de train
    Los primeros windows_size - 1 datos de las predicciones valen 0, el resto es detectado por la red
    Devuelve un dataframe booleano con el mismo indice de los datos y el resultado de la prediccion (True anomalias, False dato normal) y dos tresholds nulos (no aplican a este modelo)
    """
    def detect(self, df : pd.DataFrame):

        h0 = torch.zeros(1,  1, self.recurrent_output_size)
        c0 = torch.zeros(1,  1, self.recurrent_output_size)
        hidden = (h0,c0)
        self.setHidden(hidden)

        try:
            column = list(df.keys())[0]
        except:
            pass
        data = df.copy()   
        data = self.noise_filter.transform(data,column,column)
        data = self.normalizer.transform(data,column,column)

        fake_target = np.array([0 for i in range( len(data) )])
        raw_preds = []
        dataloader = self.detect_dataloader.create(data.values.reshape(-1,1), fake_target)
        
        preds = [0 for i in range(self.windows_size - 1)]

        data_preds = self.predict(dataloader)

        preds.extend(data_preds)

        anomaly_th_lower, anomaly_th_upper = None, None
        data["preds"] = preds
        preds = data["preds"] == 1
        data.drop(["preds"],axis=1,inplace=True)

        return preds, anomaly_th_lower, anomaly_th_upper

    """
    Implementa las predicciones de la red neuronal
    """
    def predict(self, dataloader):
        
        preds = []
        for X, _ in dataloader:
            for batch_pred in self(X):
                pred = batch_pred[-1]
                if pred >= self.th:
                    preds.append(1)
                else:
                    preds.append(0)

        return preds


"""
optimiza los hiperparametros del modelo
"""
class LSTMOptimizer():
    def __init__(self, trials : int, metric : str):
       
        assert trials > 0
        self.__trials = trials
        self.__best_score = -9999999
        self.__best_hiperparams = {}
        self.__metric = metric

    def __predict(self, model):
        
        predict = model.predict(self.__val_dataloader)
        target = []
        for X,y_batches in self.__val_dataloader:
            for batch in y_batches:
                target.append(batch[-1].item())

        if self.__metric == "precision":
            return precision_score(target, predict)
        if self.__metric == "recall":
            return recall_score(target, predict)
        return f1_score(target, predict,zero_division=0)

    def __objective(self,trial):
        
        lr = trial.suggest_float("lr",1e-4, 1e-1)
        scheduler_gamma = trial.suggest_float("scheduler_gamma",0.8, 1.0)
        predict_th = trial.suggest_float("th",0.1,0.8)
        recurrent_output_size = trial.suggest_int("recurrent_output_size",1, 10)
       
        dropout = 0.0
        num_layers = 1
        model =  LSTM_AD( input_size = self.__n_features, output_size = recurrent_output_size, num_layers = num_layers, dropout = dropout,predict_th = predict_th)
        h0 = torch.zeros(1,  self.__batch_size, recurrent_output_size)
        c0 = torch.zeros(1,  self.__batch_size, recurrent_output_size)
        hidden = (h0,c0)

        model.setHidden(hidden)
        
        epochs = 20
        patience = 5
        model.train(self.__train_dataloader, self.__val_dataloader,epochs,lr, patience, scheduler_gamma)
        
        score = self.__predict(model)

        if score > self.__best_score:
            self.__best_score = score
            self.__best_hiperparams = {"lr" : lr, "scheduler_gamma" : scheduler_gamma, "th" : predict_th, "recurrent_output_size" : recurrent_output_size}

        return score

    def earlyStop(self, study, trial):
        if self.__best_score == 1.0:
            study.stop()
    
    def findBestModel(self, train_dataloader , val_dataloader):

        self.__train_dataloader = train_dataloader
        self.__val_dataloader = val_dataloader
        
        for x,_ in train_dataloader:
            self.__n_features = x[0].shape[1] 
            self.__batch_size = x.shape[0]

        study = optuna.create_study(pruner = SuccessiveHalvingPruner(),direction = "maximize")
        study.optimize(self.__objective, n_trials=self.__trials, callbacks=[self.earlyStop])
        
        self.__best_score = -999999

        return self.__best_hiperparams



    




    