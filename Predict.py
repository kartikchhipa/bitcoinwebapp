import pickle
import streamlit as st
import pandas as pd
import datetime
import numpy as np
import keras
import xgboost as xgb
import lightgbm as lgb
import sklearn
import plotly.express as px
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step):
        a = dataset[i:(i+time_step), 0]   
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)
def app():
    model_name=st.radio('Choose One Machine Learning Model',('XGBoostRegressor','RandomForestRegressor','LGBMRegressor','DecisionTreeRegressor','LSTM Network'),disabled=False)
    data=pd.read_csv("https://raw.githubusercontent.com/KartikChhipa01/datasets/main/BTC-USD.csv")
    data['Date']=pd.to_datetime(data['Date'])
    X=data[['Date','Adj Close']]
    X_copy=X.copy()
    X_copy=X_copy[X_copy['Date']>'2021-02-19']
    X_copy2=X_copy.copy()
    del X_copy['Date']
    scaler=pickle.load(open('scaler.sav','rb'))
    X_copy=scaler.fit_transform(np.array(X_copy).reshape(-1,1))
    X_copy_train=X_copy[:255]
    X_copy_test=X_copy[255:]

    date=st.date_input('Enter the Date for which you want to predict Bitcoin Price',value=datetime.date(2022,2,20),min_value=datetime.date(2022,2,20),max_value=datetime.date(2022,3,1))
    time_step=20
    pred_days=(date-datetime.date(2022,2,19)).days
    if(model_name=='XGBoostRegressor'):
        last_days=list(X_copy_test[len(X_copy_test)-time_step:].reshape(1,-1))[0].tolist()
        model=xgb.XGBRegressor()
        model.load_model(fname='xgbreg')        
        output_days=[]
        i=0
        while(i<pred_days): 
            if(len(last_days)>time_step):
                yhat=model.predict(np.array(last_days[1:]).reshape(1,-1))
                last_days.extend(yhat.tolist())
                last_days=last_days[1:]
                output_days.extend(yhat.tolist())
                i=i+1
            else:
                yhat = model.predict(np.array(last_days).reshape(1,-1))
                last_days.extend(yhat.tolist())
                output_days.extend(yhat.tolist())
                i=i+1
        last_days_predict_plot=np.zeros((time_step+len(output_days),1))
        last_days_predict_plot[:]=np.nan
        last_days_predict_plot[:time_step]=np.array(X_copy2.iloc[len(X_copy_train)+len(X_copy_test)-time_step:,1]).reshape(-1,1)
        next_days_predict_plot=np.zeros((time_step+len(output_days),1))
        next_days_predict_plot[:]=np.nan
        next_days_predict_plot[time_step:]=scaler.inverse_transform(np.array(output_days).reshape(-1,1))

        df=pd.DataFrame({'Date':pd.date_range(datetime.datetime(2022,2,19)-datetime.timedelta(days=time_step-1), periods=time_step+len(output_days)),
                 'Close Price Last Days':last_days_predict_plot.reshape(1,-1)[0].tolist(),'Next Days Prediction':next_days_predict_plot.reshape(1,-1)[0].tolist()})
        fig=px.line(df,x=df['Date'],y=[df['Close Price Last Days'],df['Next Days Prediction']],labels={'value':'Close Price'})
        fig.update_layout(title_text='Prediction of Future Price (Using DecisionTreeRegressor)',
                        font_size=15,legend_title_text='Close Price')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig)
        st.write(scaler.inverse_transform(np.array(output_days).reshape(1,-1)))
    if(model_name=='RandomForestRegressor'):
        last_days=list(X_copy_test[len(X_copy_test)-time_step:].reshape(1,-1))[0].tolist()
        model=pickle.load(open('randomforestreg.sav','rb'))
        output_days=[]
        i=0
        while(i<pred_days): 
            if(len(last_days)>time_step):
                yhat=model.predict(np.array(last_days[1:]).reshape(1,-1))
                last_days.extend(yhat.tolist())
                last_days=last_days[1:]
                output_days.extend(yhat.tolist())
                i=i+1
            else:
                yhat = model.predict(np.array(last_days).reshape(1,-1))
                last_days.extend(yhat.tolist())
                output_days.extend(yhat.tolist())
                i=i+1
        last_days_predict_plot=np.zeros((time_step+len(output_days),1))
        last_days_predict_plot[:]=np.nan
        last_days_predict_plot[:time_step]=np.array(X_copy2.iloc[len(X_copy_train)+len(X_copy_test)-time_step:,1]).reshape(-1,1)
        next_days_predict_plot=np.zeros((time_step+len(output_days),1))
        next_days_predict_plot[:]=np.nan
        next_days_predict_plot[time_step:]=scaler.inverse_transform(np.array(output_days).reshape(-1,1))

        df=pd.DataFrame({'Date':pd.date_range(datetime.datetime(2022,2,19)-datetime.timedelta(days=time_step-1), periods=time_step+len(output_days)),
                 'Close Price Last Days':last_days_predict_plot.reshape(1,-1)[0].tolist(),'Next Days Prediction':next_days_predict_plot.reshape(1,-1)[0].tolist()})
        fig=px.line(df,x=df['Date'],y=[df['Close Price Last Days'],df['Next Days Prediction']],labels={'value':'Close Price'})
        fig.update_layout(title_text='Prediction of Future Price (Using DecisionTreeRegressor)',
                        font_size=15,legend_title_text='Close Price')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig)
        st.write(scaler.inverse_transform(np.array(output_days).reshape(1,-1)))
    if(model_name=='LGBMRegressor'):
        last_days=list(X_copy_test[len(X_copy_test)-time_step:].reshape(1,-1))[0].tolist()
        model=pickle.load(open('lgbmreg.sav','rb'))
        output_days=[]
        i=0
        while(i<pred_days):
            
            if(len(last_days)>time_step):
                
                yhat=model.predict(np.array(last_days[1:]).reshape(1,-1))
                last_days.extend(yhat.tolist())
                last_days=last_days[1:]
            
                output_days.extend(yhat.tolist())
                i=i+1
                
            else:
                yhat = model.predict(np.array(last_days).reshape(1,-1))
                last_days.extend(yhat.tolist())
                output_days.extend(yhat.tolist())
                i=i+1   
        last_days_predict_plot=np.zeros((time_step+len(output_days),1))
        last_days_predict_plot[:]=np.nan
        last_days_predict_plot[:time_step]=np.array(X_copy2.iloc[len(X_copy_train)+len(X_copy_test)-time_step:,1]).reshape(-1,1)
        next_days_predict_plot=np.zeros((time_step+len(output_days),1))
        next_days_predict_plot[:]=np.nan
        next_days_predict_plot[time_step:]=scaler.inverse_transform(np.array(output_days).reshape(-1,1))

        df=pd.DataFrame({'Date':pd.date_range(datetime.datetime(2022,2,19)-datetime.timedelta(days=time_step-1), periods=time_step+len(output_days)),
                 'Close Price Last Days':last_days_predict_plot.reshape(1,-1)[0].tolist(),'Next Days Prediction':next_days_predict_plot.reshape(1,-1)[0].tolist()})
        fig=px.line(df,x=df['Date'],y=[df['Close Price Last Days'],df['Next Days Prediction']],labels={'value':'Close Price'})
        fig.update_layout(title_text='Prediction of Future Price (Using DecisionTreeRegressor)',
                        font_size=15,legend_title_text='Close Price')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig)
        st.write(scaler.inverse_transform(np.array(output_days).reshape(1,-1)))
    if(model_name=='LSTM Network'):
        last_days=list(X_copy_test[len(X_copy_test)-time_step:].reshape(1,-1))[0].tolist()
        model=keras.models.load_model('LSTM')
        train=np.array(last_days)
        output_days=[]
        i=0
        while(i<pred_days):
            
            if(len(last_days)>time_step):
                train=np.array(last_days[1:]).astype("float32")
                train=train.reshape(1,-1)
                train=train.reshape((1,time_step,1))
                yhat=model.predict(train)
                last_days.extend(yhat[0].tolist())
                last_days=last_days[1:]
            
                output_days.extend(yhat.tolist())
                i=i+1
                
            else:
                train=train.reshape((1,time_step,1))
                yhat = model.predict(np.array(last_days).reshape((1,time_step,1)))
                last_days.extend(yhat[0].tolist())
                output_days.extend(yhat.tolist())
                i=i+1
        last_days_predict_plot=np.zeros((time_step+len(output_days),1))
        last_days_predict_plot[:]=np.nan
        last_days_predict_plot[:time_step]=np.array(X_copy2.iloc[len(X_copy_train)+len(X_copy_test)-time_step:,1]).reshape(-1,1)
        next_days_predict_plot=np.zeros((time_step+len(output_days),1))
        next_days_predict_plot[:]=np.nan
        next_days_predict_plot[time_step:]=scaler.inverse_transform(np.array(output_days).reshape(-1,1))

        df=pd.DataFrame({'Date':pd.date_range(datetime.datetime(2022,2,19)-datetime.timedelta(days=time_step-1), periods=time_step+len(output_days)),
                 'Close Price Last Days':last_days_predict_plot.reshape(1,-1)[0].tolist(),'Next Days Prediction':next_days_predict_plot.reshape(1,-1)[0].tolist()})
        fig=px.line(df,x=df['Date'],y=[df['Close Price Last Days'],df['Next Days Prediction']],labels={'value':'Close Price'})
        fig.update_layout(title_text='Prediction of Future Price (Using DecisionTreeRegressor)'
                        ,font_size=15,legend_title_text='Close Price')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig)
        st.write(scaler.inverse_transform(np.array(output_days).reshape(1,-1)))
    if(model_name=='DecisionTreeRegressor'):
        last_days=list(X_copy_test[len(X_copy_test)-time_step:].reshape(1,-1))[0].tolist()
        model=pickle.load(open('dtreereg.sav','rb'))
        output_days=[]
        i=0
        while(i<pred_days):
            
            if(len(last_days)>time_step):
                
                yhat=model.predict(np.array(last_days[1:]).reshape(1,-1))
                last_days.extend(yhat.tolist())
                last_days=last_days[1:]
            
                output_days.extend(yhat.tolist())
                i=i+1
                
            else:
                yhat = model.predict(np.array(last_days).reshape(1,-1))
                last_days.extend(yhat.tolist())
                output_days.extend(yhat.tolist())
                
                i=i+1
        last_days_predict_plot=np.zeros((time_step+len(output_days),1))
        last_days_predict_plot[:]=np.nan
        last_days_predict_plot[:time_step]=np.array(X_copy2.iloc[len(X_copy_train)+len(X_copy_test)-time_step:,1]).reshape(-1,1)
        next_days_predict_plot=np.zeros((time_step+len(output_days),1))
        next_days_predict_plot[:]=np.nan
        next_days_predict_plot[time_step:]=scaler.inverse_transform(np.array(output_days).reshape(-1,1))

        df=pd.DataFrame({'Date':pd.date_range(datetime.datetime(2022,2,19)-datetime.timedelta(days=time_step-1), periods=time_step+len(output_days)),
                 'Close Price Last Days':last_days_predict_plot.reshape(1,-1)[0].tolist(),'Next Days Prediction':next_days_predict_plot.reshape(1,-1)[0].tolist()})
        fig=px.line(df,x=df['Date'],y=[df['Close Price Last Days'],df['Next Days Prediction']],labels={'value':'Close Price'})
        fig.update_layout(title_text='Prediction of Future Price (Using DecisionTreeRegressor)',
                        font_size=15,legend_title_text='Close Price')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig)
        st.write(scaler.inverse_transform(np.array(output_days).reshape(1,-1)))
