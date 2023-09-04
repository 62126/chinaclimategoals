#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


# In[5]:


cols = ["Total Energy Consumption, Agriculture, Forestry, Animal Husbandry and Fishery(10000 tons of SCE)", 
        "Total Energy Consumption, Manufacturing(10000 tons of SCE)", 
        "Total Energy Consumption, Manufacture of Computers, Communication and Other Electronic Equipment(10000 tons of SCE)",
        "Total Energy Consumption, Wholesale, Retail Trade and Hotel, Restaurants(10000 tons of SCE)"]

for col in cols:
    Target_Column = col

    df = pd.read_excel("D:\Jupyter Directory\Dissertation\Dissertation Datasets\Ones I will use\Consumption by Sector.xls")
    df.set_index("Databaseï¼Annual", inplace=True)
    new_df = df.T[["Indicators", Target_Column]].reset_index(drop=True)
    new_df = new_df.rename(columns={"Indicators": "Year"}).sort_values(by="Year")
    new_df["Year"] = new_df["Year"].astype("int")
    new_df.dropna(inplace=True)

    t = new_df["Year"]
    y = new_df[Target_Column]

    plt.figure(figsize=(10, 6))
    plt.plot(t, y, color="dodgerblue", linewidth=2, label="Original Data")
    plt.xlabel("Year")
    plt.ylabel(Target_Column)
    plt.title(f"Plot of {Target_Column}")
    plt.grid(True)
    plt.legend()
    plt.show()


# In[6]:


cols = ["Total Energy Consumption(10000 tons of SCE)",
        "Proportion of Coal(%)",
        "Proportion of Petroleum(%)", 
        "Proportion of Natural Gas(%)",
        "Proportion of Primary  Electricity and  Other Energy(%)",
        "Consumption of Coal(10000 tons)", 
        "Consumption of Coke(10000 tons)",
        "Consumption of Crude Oil(10000 tons)",
        "Consumption of Gasoline(10000 tons)",
        "Consumption of Kerosene(10000 tons)",
        "Consumption of Diesel Oil(10000 tons)",
        "Consumption of Fuel Oil(10000 tons)",
        "Consumption of Natural Gas(100 million cu.m)",
        "Consumption of Electricity(100 million kwh)"]

for col in cols:
    Target_Column = col

    df = pd.read_excel("D:\Jupyter Directory\Dissertation\Dissertation Datasets\Ones I will use\Annual Total Energy Consumption.xls")
    df.set_index("Databaseï¼Annual", inplace=True)
    new_df = df.T[["Indicators", Target_Column]].reset_index(drop=True)
    new_df = new_df.rename(columns={"Indicators": "Year"}).sort_values(by="Year")
    new_df["Year"] = new_df["Year"].astype("int")
    new_df.dropna(inplace=True)

    t = new_df["Year"]
    y = new_df[Target_Column]

    plt.figure(figsize=(10, 6))
    plt.plot(t, y, color="dodgerblue", linewidth=2, label="Original Data")
    plt.xlabel("Year")
    plt.ylabel(Target_Column)
    plt.title(f"Plot of {Target_Column}")
    plt.grid(True)
    plt.legend()
    plt.show()


# In[44]:


from pmdarima import auto_arima

cols = ["Total Energy Consumption, Agriculture, Forestry, Animal Husbandry and Fishery(10000 tons of SCE)", 
        "Total Energy Consumption, Manufacturing(10000 tons of SCE)", 
        "Total Energy Consumption, Manufacture of Computers, Communication and Other Electronic Equipment(10000 tons of SCE)",
        "Total Energy Consumption, Wholesale, Retail Trade and Hotel, Restaurants(10000 tons of SCE)"]

Arima_Results = []

for col in cols:

    Target_Column = col

    df = pd.read_excel("D:\Jupyter Directory\Dissertation\Dissertation Datasets\Ones I will use\Consumption by Sector.xls")
    
    df.set_index("Databaseï¼Annual",inplace = True)

    new_df = df.T[["Indicators",Target_Column]].reset_index(drop = True)

    new_df = new_df.rename(columns={"Indicators": "Year"}).sort_values(by="Year")
    new_df["Year"] = new_df["Year"].astype("int")

    new_df.dropna(inplace = True)

    # Generate synthetic time series data
    t = new_df["Year"]
    y = new_df[Target_Column]

    # Find the best ARIMA model using auto_arima
    stepwise_fit = auto_arima(y, seasonal=False, trace=True)

    # Fit the best ARIMA model
    best_order = stepwise_fit.get_params()["order"]
    model = stepwise_fit.fit(y)

    # Forecast the next 10 data points
    forecast, conf_int = model.predict(n_periods=10, return_conf_int=True)

    # Append the forecasted values to the original data
    extended_t = np.concatenate((t, np.arange(t.iloc[-1]+1, t.iloc[-1]+11)))
    extended_y = np.concatenate((y, forecast))

    Result = pd.DataFrame({"Year":extended_t,Target_Column:extended_y})
    Arima_Results.append(Result)

    # Plot original data and forecasted values
    plt.figure(figsize=(10, 6))

    # Plot original data in a solid line
    plt.plot(t, y, color="dodgerblue", linewidth=2, label="Original Data")

    # Plot forecasted values as a dashed line with shaded confidence interval
    plt.plot(extended_t[-10:], forecast, linestyle="dashed", color="darkorange", label="Forecasted Values")
    plt.fill_between(extended_t[-10:], conf_int[:, 0], conf_int[:, 1], color="darkorange", alpha=0.1)

    # Set labels and title
    plt.xlabel("Year")
    plt.ylabel(Target_Column)
    plt.title(f"ARIMA Forecast (Order: {best_order})")

    # Add grid lines
    plt.grid(True)

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()
    


Final_Arima_Results1 = pd.concat(Arima_Results, axis=1)

Final_Arima_Results1.columns.values[0] = "year"

# Assuming 'df' is your DataFrame
cols_to_drop = [col for col in Final_Arima_Results1.columns if col == "Year"][1:]
Final_Arima_Results1.drop(columns=cols_to_drop, inplace=True)

Final_Arima_Results1



# In[13]:


from pmdarima import auto_arima

cols = ["Total Energy Consumption(10000 tons of SCE)",
        "Proportion of Coal(%)",
        "Proportion of Petroleum(%)", 
        "Proportion of Natural Gas(%)",
        "Proportion of Primary  Electricity and  Other Energy(%)",
        "Consumption of Coal(10000 tons)", 
        "Consumption of Coke(10000 tons)",
        "Consumption of Crude Oil(10000 tons)",
        "Consumption of Gasoline(10000 tons)",
        "Consumption of Kerosene(10000 tons)",
        "Consumption of Diesel Oil(10000 tons)",
        "Consumption of Fuel Oil(10000 tons)",
        "Consumption of Natural Gas(100 million cu.m)",
        "Consumption of Electricity(100 million kwh)"]

Arima_Results = []

for col in cols:

    Target_Column = col

    df = pd.read_excel("D:\Jupyter Directory\Dissertation\Dissertation Datasets\Ones I will use\Annual Total Energy Consumption.xls")

    df.set_index("Databaseï¼Annual",inplace = True)

    new_df = df.T[["Indicators",Target_Column]].reset_index(drop = True)

    new_df = new_df.rename(columns={"Indicators": "Year"}).sort_values(by="Year")
    new_df["Year"] = new_df["Year"].astype("int")

    new_df.dropna(inplace = True)

    # Generate synthetic time series data
    t = new_df["Year"]
    y = new_df[Target_Column]

    # Find the best ARIMA model using auto_arima
    stepwise_fit = auto_arima(y, seasonal=False, trace=True)

    # Fit the best ARIMA model
    best_order = stepwise_fit.get_params()["order"]
    model = stepwise_fit.fit(y)

    # Forecast the next 10 data points
    forecast, conf_int = model.predict(n_periods=10, return_conf_int=True)

    # Append the forecasted values to the original data
    extended_t = np.concatenate((t, np.arange(t.iloc[-1]+1, t.iloc[-1]+11)))
    extended_y = np.concatenate((y, forecast))

    Result = pd.DataFrame({"Year":extended_t,Target_Column:extended_y})
    Arima_Results.append(Result)

    # Plot original data and forecasted values
    plt.figure(figsize=(10, 6))

    # Plot original data in a solid line
    plt.plot(t, y, color="dodgerblue", linewidth=2, label="Original Data")

    # Plot forecasted values as a dashed line with shaded confidence interval
    plt.plot(extended_t[-10:], forecast, linestyle="dashed", color="darkorange", label="Forecasted Values")
    plt.fill_between(extended_t[-10:], conf_int[:, 0], conf_int[:, 1], color="darkorange", alpha=0.1)

    # Set labels and title
    plt.xlabel("Year")
    plt.ylabel(Target_Column)
    plt.title(f"ARIMA Forecast (Order: {best_order})")

    # Add grid lines
    plt.grid(True)

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()
    


Final_Arima_Results2 = pd.concat(Arima_Results, axis=1)

Final_Arima_Results2.columns.values[0] = "year"

# Assuming 'df' is your DataFrame
cols_to_drop = [col for col in Final_Arima_Results2.columns if col == "Year"][1:]
Final_Arima_Results2.drop(columns=cols_to_drop, inplace=True)

Final_Arima_Results2


# In[14]:


from keras.models import Sequential
from keras.layers import LSTM, Dense

cols = ["Total Energy Consumption, Agriculture, Forestry, Animal Husbandry and Fishery(10000 tons of SCE)", 
        "Total Energy Consumption, Manufacturing(10000 tons of SCE)", 
        "Total Energy Consumption, Manufacture of Computers, Communication and Other Electronic Equipment(10000 tons of SCE)",
        "Total Energy Consumption, Wholesale, Retail Trade and Hotel, Restaurants(10000 tons of SCE)"]

LSTM_Results = []

for col in cols:

    # Load data
    target_column = col
    df = pd.read_excel("D:\Jupyter Directory\Dissertation\Dissertation Datasets\Ones I will use\Consumption by Sector.xls")

    # Set the index
    df.set_index("Databaseï¼Annual", inplace=True)

    # Prepare the data
    new_df = df.T[["Indicators", target_column]].reset_index(drop=True)
    new_df = new_df.rename(columns={"Indicators": "Year"}).sort_values(by="Year")
    new_df["Year"] = new_df["Year"].astype("int")
    new_df.dropna(inplace=True)

    # Select the target column
    y = new_df[target_column].values

    # Normalize the data
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))

    # Prepare the data for LSTM
    look_back = 5  # Number of previous time steps to use for prediction
    X = []
    y_lstm = []
    for i in range(len(y_scaled) - look_back):
        X.append(y_scaled[i:(i + look_back)])
        y_lstm.append(y_scaled[i + look_back])
    X, y_lstm = np.array(X), np.array(y_lstm)

    # Reshape the data for LSTM (samples, time steps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train the model
    model.fit(X, y_lstm, epochs=100, batch_size=32)

    # Forecast the next 10 data points
    last_sequence = y_scaled[-look_back:]
    forecast_scaled = []
    for _ in range(10):
        input_data = np.array([last_sequence[-look_back:]])
        input_data = input_data.reshape((1, look_back, 1))
        forecast_scaled.append(model.predict(input_data)[0, 0])
        last_sequence = np.append(last_sequence, forecast_scaled[-1])

    # Inverse transform the forecasted data
    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

    # Append the forecasted values to the original data
    extended_t = np.concatenate((new_df["Year"], np.arange(new_df["Year"].iloc[-1] + 1, new_df["Year"].iloc[-1] + 11)))
    extended_y = np.concatenate((y, forecast))

    Result = pd.DataFrame({"Year": extended_t, target_column: extended_y})

    LSTM_Results.append(Result)

    # Plot original data and forecasted values
    plt.figure(figsize=(10, 6))
    plt.plot(new_df["Year"], y, color="dodgerblue", linewidth=2, label="Original Data")
    plt.plot(extended_t[-10:], forecast, linestyle="dashed", color="darkorange", label="Forecasted Values")
    plt.xlabel("Year")
    plt.ylabel(target_column)
    plt.title("LSTM Forecast")
    plt.grid(True)
    plt.legend()
    plt.show()

Final_LSTM_Results1 = pd.concat(LSTM_Results, axis=1)

Final_LSTM_Results1.columns.values[0] = "year"

# Assuming 'df' is your DataFrame
cols_to_drop = [col for col in Final_LSTM_Results1.columns if col == "Year"][1:]
Final_LSTM_Results1.drop(columns=cols_to_drop, inplace=True)

Final_LSTM_Results1


# In[15]:


from keras.models import Sequential
from keras.layers import LSTM, Dense

cols = ["Total Energy Consumption(10000 tons of SCE)",
        "Proportion of Coal(%)",
        "Proportion of Petroleum(%)", 
        "Proportion of Natural Gas(%)",
        "Proportion of Primary  Electricity and  Other Energy(%)",
        "Consumption of Coal(10000 tons)", 
        "Consumption of Coke(10000 tons)",
        "Consumption of Crude Oil(10000 tons)",
        "Consumption of Gasoline(10000 tons)",
        "Consumption of Kerosene(10000 tons)",
        "Consumption of Diesel Oil(10000 tons)",
        "Consumption of Fuel Oil(10000 tons)",
        "Consumption of Natural Gas(100 million cu.m)",
        "Consumption of Electricity(100 million kwh)"]

LSTM_Results = []

for col in cols:

    # Load data
    target_column = col
    df = pd.read_excel("D:\Jupyter Directory\Dissertation\Dissertation Datasets\Ones I will use\Annual Total Energy Consumption.xls")

    # Set the index
    df.set_index("Databaseï¼Annual", inplace=True)

    # Prepare the data
    new_df = df.T[["Indicators", target_column]].reset_index(drop=True)
    new_df = new_df.rename(columns={"Indicators": "Year"}).sort_values(by="Year")
    new_df["Year"] = new_df["Year"].astype("int")
    new_df.dropna(inplace=True)

    # Select the target column
    y = new_df[target_column].values

    # Normalize the data
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))

    # Prepare the data for LSTM
    look_back = 5  # Number of previous time steps to use for prediction
    X = []
    y_lstm = []
    for i in range(len(y_scaled) - look_back):
        X.append(y_scaled[i:(i + look_back)])
        y_lstm.append(y_scaled[i + look_back])
    X, y_lstm = np.array(X), np.array(y_lstm)

    # Reshape the data for LSTM (samples, time steps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train the model
    model.fit(X, y_lstm, epochs=100, batch_size=32)

    # Forecast the next 10 data points
    last_sequence = y_scaled[-look_back:]
    forecast_scaled = []
    for _ in range(10):
        input_data = np.array([last_sequence[-look_back:]])
        input_data = input_data.reshape((1, look_back, 1))
        forecast_scaled.append(model.predict(input_data)[0, 0])
        last_sequence = np.append(last_sequence, forecast_scaled[-1])

    # Inverse transform the forecasted data
    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

    # Append the forecasted values to the original data
    extended_t = np.concatenate((new_df["Year"], np.arange(new_df["Year"].iloc[-1] + 1, new_df["Year"].iloc[-1] + 11)))
    extended_y = np.concatenate((y, forecast))

    Result = pd.DataFrame({"Year": extended_t, target_column: extended_y})

    LSTM_Results.append(Result)

    # Plot original data and forecasted values
    plt.figure(figsize=(10, 6))
    plt.plot(new_df["Year"], y, color="dodgerblue", linewidth=2, label="Original Data")
    plt.plot(extended_t[-10:], forecast, linestyle="dashed", color="darkorange", label="Forecasted Values")
    plt.xlabel("Year")
    plt.ylabel(target_column)
    plt.title("LSTM Forecast")
    plt.grid(True)
    plt.legend()
    plt.show()
    
Final_LSTM_Results2 = pd.concat(LSTM_Results, axis=1)

Final_LSTM_Results2.columns.values[0] = "year"

# Assuming 'df' is your DataFrame
cols_to_drop = [col for col in Final_LSTM_Results2.columns if col == "Year"][1:]
Final_LSTM_Results2.drop(columns=cols_to_drop, inplace=True)

Final_LSTM_Results2


# In[16]:


from prophet import Prophet

# Define the list of columns to forecast
cols = ["Total Energy Consumption, Agriculture, Forestry, Animal Husbandry and Fishery(10000 tons of SCE)", 
        "Total Energy Consumption, Manufacturing(10000 tons of SCE)", 
        "Total Energy Consumption, Manufacture of Computers, Communication and Other Electronic Equipment(10000 tons of SCE)",
        "Total Energy Consumption, Wholesale, Retail Trade and Hotel, Restaurants(10000 tons of SCE)"]

Prophet_Results = []

for col in cols:
    # Load data
    target_column = col
    df = pd.read_excel("D:\Jupyter Directory\Dissertation\Dissertation Datasets\Ones I will use\Consumption by Sector.xls")

    # Preprocessing: Set the index, select relevant columns, and rename them

    df.set_index("Databaseï¼Annual", inplace=True)

    # Prepare the data
    new_df = df.T[["Indicators", target_column]].reset_index(drop=True)
    new_df["Indicators"] =new_df["Indicators"].astype("int")
    new_df["date"] = pd.to_datetime(new_df["Indicators"].astype(str) + "-12-31")

    new_df = new_df.rename(columns={"date": "ds", target_column: "y"}).sort_values(by="ds")
    new_df.dropna(inplace=True)

    # Create and fit the Prophet model
    model = Prophet()
    model.fit(new_df)

    # Make future dataframe for forecasting
    future = model.make_future_dataframe(periods=10, freq="Y")

    # Forecast the next 10 data points
    forecast = model.predict(future)

    # Append the forecasted values to the original data
    extended_t = future["ds"]
    extended_y = np.concatenate([new_df["y"].values, forecast["yhat"][-10:]])

    Result = pd.DataFrame({"Year": extended_t, target_column: extended_y})

    Result["Year"] = Result["Year"].dt.year

    Prophet_Results.append(Result)

    # Plot original data and forecasted values
    plt.figure(figsize=(10, 6))
    plt.plot(new_df["ds"], new_df["y"], color="dodgerblue", linewidth=2, label="Original Data")
    plt.plot(forecast["ds"][-10:], forecast["yhat"][-10:], linestyle="dashed", color="darkorange", label="Forecasted Values")
    plt.fill_between(forecast["ds"][-10:], forecast["yhat_lower"][-10:], forecast["yhat_upper"][-10:], color="darkorange", alpha=0.1)
    plt.xlabel("Year")
    plt.ylabel(target_column)
    plt.title("Prophet Forecast")
    plt.grid(True)
    plt.legend()
    plt.show()
    
Final_Prophet_Results1 = pd.concat(Prophet_Results, axis=1)

Final_Prophet_Results1.columns.values[0] = "year"

# Assuming 'df' is your DataFrame
cols_to_drop = [col for col in Final_Prophet_Results1.columns if col == "Year"][1:]
Final_Prophet_Results1.drop(columns=cols_to_drop, inplace=True)

Final_Prophet_Results1


# In[17]:


from prophet import Prophet

# Define the list of columns to forecast
cols = ["Total Energy Consumption(10000 tons of SCE)",
        "Proportion of Coal(%)",
        "Proportion of Petroleum(%)", 
        "Proportion of Natural Gas(%)",
        "Proportion of Primary  Electricity and  Other Energy(%)",
        "Consumption of Coal(10000 tons)", 
        "Consumption of Coke(10000 tons)",
        "Consumption of Crude Oil(10000 tons)",
        "Consumption of Gasoline(10000 tons)",
        "Consumption of Kerosene(10000 tons)",
        "Consumption of Diesel Oil(10000 tons)",
        "Consumption of Fuel Oil(10000 tons)",
        "Consumption of Natural Gas(100 million cu.m)",
        "Consumption of Electricity(100 million kwh)"]

Prophet_Results = []

for col in cols:
    # Load data
    target_column = col
    df = pd.read_excel("D:\Jupyter Directory\Dissertation\Dissertation Datasets\Ones I will use\Annual Total Energy Consumption.xls")

    # Preprocessing: Set the index, select relevant columns, and rename them

    df.set_index("Databaseï¼Annual", inplace=True)

    # Prepare the data
    new_df = df.T[["Indicators", target_column]].reset_index(drop=True)
    new_df["Indicators"] =new_df["Indicators"].astype("int")
    new_df["date"] = pd.to_datetime(new_df["Indicators"].astype(str) + "-12-31")

    new_df = new_df.rename(columns={"date": "ds", target_column: "y"}).sort_values(by="ds")
    new_df.dropna(inplace=True)

    # Create and fit the Prophet model
    model = Prophet()
    model.fit(new_df)

    # Make future dataframe for forecasting
    future = model.make_future_dataframe(periods=10, freq="Y")

    # Forecast the next 10 data points
    forecast = model.predict(future)

    # Append the forecasted values to the original data
    extended_t = future["ds"]
    extended_y = np.concatenate([new_df["y"].values, forecast["yhat"][-10:]])

    Result = pd.DataFrame({"Year": extended_t, target_column: extended_y})

    Result["Year"] = Result["Year"].dt.year

    Prophet_Results.append(Result)

    # Plot original data and forecasted values
    plt.figure(figsize=(10, 6))
    plt.plot(new_df["ds"], new_df["y"], color="dodgerblue", linewidth=2, label="Original Data")
    plt.plot(forecast["ds"][-10:], forecast["yhat"][-10:], linestyle="dashed", color="darkorange", label="Forecasted Values")
    plt.fill_between(forecast["ds"][-10:], forecast["yhat_lower"][-10:], forecast["yhat_upper"][-10:], color="darkorange", alpha=0.1)
    plt.xlabel("Year")
    plt.ylabel(target_column)
    plt.title("Prophet Forecast")
    plt.grid(True)
    plt.legend()
    plt.show()
    
Final_Prophet_Results2= pd.concat(Prophet_Results, axis=1)

Final_Prophet_Results2.columns.values[0] = "year"

# Assuming 'df' is your DataFrame
cols_to_drop = [col for col in Final_Prophet_Results2.columns if col == "Year"][1:]
Final_Prophet_Results2.drop(columns=cols_to_drop, inplace=True)

Final_Prophet_Results2


# In[18]:


# Calculate the average for each corresponding cell in the three dataframes

Average1 = (Final_Arima_Results1 + Final_LSTM_Results1 + Final_Prophet_Results1) / 3

print("Average1 DataFrame:")
Average1


# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt

# Convert 'Year' column to string
# Remove the last two characters from the 'year' column
Average1["year"] = Average1["year"].astype(str).str[:-2]

# Set seaborn style
sns.set(style="whitegrid")



# Iterate through each column and create separate plots
for col in Average1.columns[1:]:
    # Set the plot size
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=Average1, x="year", y=col, marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Values")
    ax.set_title(f"Average Plot for {col}")
    ax.yaxis.grid(True)  # Show only horizontal gridlines
    ax.xaxis.grid(False)  # Turn off vertical gridlines
    
    # Rotate x-axis tick labels by 45 degrees
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Customize plot appearance
    sns.despine()  # Remove spines (axis lines)
    ax.tick_params(axis="both", which="both", length=0)  # Remove tick marks
    ax.set_facecolor("#f0f0f0")  # Set plot background color
    ax.grid(color="white", linestyle="-", linewidth=0.5)  # Adjust gridlines
    
    plt.tight_layout()  # Improve spacing between plots
    plt.show()


# In[20]:


# Calculate the average for each corresponding cell
Average2 = (Final_Arima_Results2 + Final_LSTM_Results2 + Final_Prophet_Results2) / 3

print("Average2 Dataframe:")
Average2


# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt

# Convert 'Year' column to string
# Remove the last two characters from the 'year' column
Average2["year"] = Average2["year"].astype(str).str[:-2]

# Set seaborn style
sns.set(style="whitegrid")



# Iterate through each column and create separate plots
for col in Average2.columns[1:]:
    # Set the plot size
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=Average2, x="year", y=col, marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Values")
    ax.set_title(f"Average Plot for {col}")
    ax.yaxis.grid(True)  # Show only horizontal gridlines
    ax.xaxis.grid(False)  # Turn off vertical gridlines
    
    # Rotate x-axis tick labels by 45 degrees
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Customize plot appearance
    sns.despine()  # Remove spines (axis lines)
    ax.tick_params(axis="both", which="both", length=0)  # Remove tick marks
    ax.set_facecolor("#f0f0f0")  # Set plot background color
    ax.grid(color="white", linestyle="-", linewidth=0.5)  # Adjust gridlines
    
    plt.tight_layout()  # Improve spacing between plots
    plt.show()


# ---

# In[33]:


from sklearn.metrics import mean_squared_error
import numpy as np

# Assuming you have calculated your Average2 DataFrame
# And you have individual forecast DataFrames: Final_Arima_Results2, Final_LSTM_Results2, Final_Prophet_Results2
cols = ["Total Energy Consumption(10000 tons of SCE)",
        "Proportion of Coal(%)",
        "Proportion of Petroleum(%)", 
        "Proportion of Natural Gas(%)",
        "Proportion of Primary  Electricity and  Other Energy(%)",
        "Consumption of Coal(10000 tons)", 
        "Consumption of Coke(10000 tons)",
        "Consumption of Crude Oil(10000 tons)",
        "Consumption of Gasoline(10000 tons)",
        "Consumption of Kerosene(10000 tons)",
        "Consumption of Diesel Oil(10000 tons)",
        "Consumption of Fuel Oil(10000 tons)",
        "Consumption of Natural Gas(100 million cu.m)",
        "Consumption of Electricity(100 million kwh)"]

individual_forecasts = [Final_Arima_Results2, Final_LSTM_Results2, Final_Prophet_Results2]
forecast_names = ["ARIMA", "LSTM", "Prophet"]

for col in cols:
    forecast_columns = [forecast[col] for forecast in individual_forecasts]
    average_forecast_col = Average2[col]
    
    rmse_values = []
    
    for i, forecast_col in enumerate(forecast_columns):
        # Drop rows with NaN from both forecast_col and average_forecast_col
        valid_indices = ~np.isnan(forecast_col) & ~np.isnan(average_forecast_col)
        forecast_col_valid = forecast_col[valid_indices]
        average_forecast_col_valid = average_forecast_col[valid_indices]
        
        # Calculate the RMSE for the current forecast method
        rmse = np.sqrt(mean_squared_error(forecast_col_valid, average_forecast_col_valid))
        rmse_values.append(rmse)
        
        print(f"RMSE between Average and {forecast_names[i]} forecast for '{col}': {rmse:.2f}")


# In[34]:


from sklearn.metrics import mean_squared_error
import numpy as np

# Assuming you have calculated your Average2 DataFrame
# And you have individual forecast DataFrames: Final_Arima_Results2, Final_LSTM_Results2, Final_Prophet_Results2
cols = ["Total Energy Consumption, Agriculture, Forestry, Animal Husbandry and Fishery(10000 tons of SCE)", 
        "Total Energy Consumption, Manufacturing(10000 tons of SCE)", 
        "Total Energy Consumption, Manufacture of Computers, Communication and Other Electronic Equipment(10000 tons of SCE)",
        "Total Energy Consumption, Wholesale, Retail Trade and Hotel, Restaurants(10000 tons of SCE)"]

individual_forecasts = [Final_Arima_Results1, Final_LSTM_Results1, Final_Prophet_Results1]
forecast_names = ["ARIMA", "LSTM", "Prophet"]

for col in cols:  # Assuming 'cols' contains the forecasted column names
    # Select the corresponding columns from individual forecasts and the average forecast
    forecast_columns = [forecast[col] for forecast in individual_forecasts]
    average_forecast_col = Average1[col]
    
    rmse_values = []
    
    for i, forecast_col in enumerate(forecast_columns):
        # Drop rows with NaN from both forecast_col and average_forecast_col
        valid_indices = ~np.isnan(forecast_col) & ~np.isnan(average_forecast_col)
        forecast_col_valid = forecast_col[valid_indices]
        average_forecast_col_valid = average_forecast_col[valid_indices]
        
        # Calculate the RMSE for the current forecast method
        rmse = np.sqrt(mean_squared_error(forecast_col_valid, average_forecast_col_valid))
        rmse_values.append(rmse)
        
        print(f"RMSE between Average and {forecast_names[i]} forecast for '{col}': {rmse:.2f}")


# In[ ]:




