import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error
import pickle

df=pd.read_csv('Delhi_v2.csv')

df['locality']=df['Address'].str.split(',').str[0:3].str.join(' ')

print(df['locality'].value_counts().head(50))

grouped_median=df.groupby('locality')['price'].median()
print(grouped_median)

df['Locality_prices']=df['locality'].map(grouped_median)

df.to_csv('Delhi_processed.csv', index=False)

y=df['price']

x=df[['Locality_prices','area','Bedrooms','Bathrooms']]

print(x.head())

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

regressor=RandomForestRegressor()

regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

print(r2_score(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))

with open('rf_model.pkl', 'wb') as model_file:
    pickle.dump(regressor, model_file)


with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
