from abc import ABC, abstractmethod
import pandas as pd
import yfinance as yf
import requests
import numpy as np
import matplotlib.pyplot as plt



class strategy(ABC):
    
    def __init__(self, name):

        """
        General abstract class for Quantitative Investment Strategies

        Attributes:
        - name: Name of the strategy
        - _benchmarks: Benchmarks of the strategy for performance calculation
        - _uls_universe: Dataframe containing the timeseries for all underlyings contained in our strategy on all trading days
        - _weights_df: dataframe containing timeseries of weights for our strategy on all trading days
        - _pv_df : dataframe containing timeseries of strategy value on all trading days 
        - _reb_weights_df: dataframe containing timeseries of weights for our strategy on rebalancing days
        - _reb_underlyings_df : dataframe containing timeseries of underlying prices for our strategy on rebalancing days
        - _reb_pv_df : dataframe containing timeseries of the value of our strategy on rebalancing days

        Warning : 
        - When using a function for the calculation of the weights, the first argument of the function MUST be
                  the dataframe containing the timeseries for the underlyings, so it must be _uls_universe
        - This function rebalancing frequency parameter must be named "reb_freq"
        """
        self.name = name
        self.__benchmarks = None

        # Data : Benchmarks and Underlyings
        self.__selection_df = None
        self.__uls_universe = None
        self.__weights_df = None

        # Calculated Dataframes
        self.__pv_df = None
        self.__spis_df = None
        self.__reb_weights_df = None 
        self.__reb_underlyings_df = None
        self.__reb_pv_df = None
        self.__reb_spis_df = None
        
        #Performance attributes
        self.__annualized_return = None
        self.__total_return = None
        self.__annualized_vol = None
        self.__sharpe_ratio = None
        self.__performance_table = None

    @property
    def performance_table(self):
        return self.__performance_table
    
    @performance_table.setter
    def performance_table(self, benchmarks):
        self.__performance_table = benchmarks

    @property
    def annualized_return(self):
        return self.__annualized_return
    
    @annualized_return.setter
    def annualized_return(self, benchmarks):
        self.__annualized_return = benchmarks

    @property
    def selection_df(self):
        return self.__selection_df
    
    @selection_df.setter
    def selection_df(self, benchmarks):
        self.__selection_df = benchmarks

    @property
    def total_return(self):
        return self.__total_return
    
    @total_return.setter
    def total_return(self, benchmarks):
        self.__total_return = benchmarks

    @property
    def annualized_vol(self):
        return self.__annualized_vol
    
    @annualized_vol.setter
    def annualized_vol(self, benchmarks):
        self.__annualized_vol = benchmarks

    @property
    def sharpe_ratio(self):
        return self.__sharpe_ratio
    
    @sharpe_ratio.setter
    def sharpe_ratio(self, benchmarks):
        self.__sharpe_ratio = benchmarks


    @property
    def benchmarks(self):
        return self.__benchmarks
    
    @benchmarks.setter
    def benchmarks(self, benchmarks):
        self.__benchmarks = benchmarks
    
    @property
    def pv_df(self):
        return self.__pv_df 
    
    @pv_df.setter
    def pv_df(self, benchmarks):
        self.__pv_df = benchmarks

    @property
    def spis_df(self):
        return self.__spis_df 
    
    @spis_df.setter
    def spis_df(self, benchmarks):
        self.__spis_df = benchmarks

    @property
    def reb_spis_df(self):
        return self.__reb_spis_df 
    
    @reb_spis_df.setter
    def reb_spis_df(self, benchmarks):
        self.__reb_spis_df = benchmarks

    @property
    def reb_pv_df(self):
        return self.__reb_pv_df
    
    @reb_pv_df.setter
    def reb_pv_df(self, benchmarks):
        self.__reb_pv_df = benchmarks

    @property
    def ULs_universe(self):
        return self.__uls_universe
    
    @ULs_universe.setter
    def ULs_universe(self, Uls):
        self.__uls_universe = Uls

    @property
    def weights(self):
        return self.__weights_df
    
    @weights.setter
    def weights(self, Uls):
        self.__weights_df = Uls

    @property
    def reb_weights_df(self):
        return self.__reb_weights_df
    
    @reb_weights_df.setter
    def reb_weights_df(self, Uls):
        self.__reb_weights_df = Uls

    @property
    def reb_underlyings_df(self):
        return self.__reb_underlyings_df
    
    @reb_underlyings_df.setter
    def reb_underlyings_df(self, Uls):
        self.__reb_underlyings_df = Uls




    @abstractmethod
    def performance_calculation(self):
        pass

    @abstractmethod
    def to_excel(self): 
        pass
    
    @abstractmethod
    def implement_strat(self, func, **funcArgs):
        pass

    @abstractmethod
    def plot_graph(self,graph_name):
        pass

    def dl_data_from_excel(self, path, decimal_separator = '.'):
        return pd.read_excel(path, header=0,index_col=0,skiprows=0, decimal = decimal_separator)

    def dl_data(self, ticker, start_date, end_date): 
        print("Data downloading...")
        try : 
            dl_data = yf.download(ticker, start = start_date, end = end_date)
            if dl_data.empty : 
                print(f"No data found for {ticker} with yahoo finance. Attempt with Capital IQ...")
                #MISS IMPLEMENTATION OF CAPITAL IQ API Here
                    
                if dl_data.empty : 
                    print(f"No data found for {ticker} neither with Yahoo nor Capital IQ")
                    return pd.DataFrame()
            else : 
                print("Data download : DONE")
                return dl_data['Close']
        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
        except ValueError as e:
            print(f"Invalid parameters: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def add_data(self, ticker = None, start_date = None, end_date = None, path = None, decimal_separator = '.'):
        if path == None :
            new_data = self.dl_data(ticker, start_date, end_date)
        else :

            new_data = self.dl_data_from_excel(path = path, decimal_separator = decimal_separator)
            if new_data.empty : 
                print("No data downloaded")
                return
            else : 

                self.ULs_universe = new_data
                return
            
        if new_data.empty : 
            print("No data downloaded")
            return
        
        if self.ULs_universe is None:
            self.ULs_universe = pd.DataFrame(new_data)
            self.ULs_universe.columns = [ticker]
        else:
            self.ULs_universe[ticker] = new_data
    
    def add_benchmark(self, benchmark_ticker, start_date = None, end_date = None, path = None):
        if path == None :
            new_data = self.dl_data(benchmark_ticker, start_date, end_date)
        else :
            new_data = self.dl_data_from_excel(path = path)
            if new_data.empty : 
                print("No data downloaded")
                return
            else : 

                print(new_data)

                self.benchmarks = new_data
        if new_data.empty : 
            print("No data downloaded")
            return
        
        if self.benchmarks is None:
            self.benchmarks = pd.DataFrame(new_data)
            self.benchmarks.columns = [benchmark_ticker]
        else:
            self.benchmarks[benchmark_ticker] = new_data

    @property
    def get_benchmark_names(self):
        if self.benchmarks == None :
            return None
        else : 
            return list(self.benchmarks.columns)

def calculate_rebalancing_dates(trading_days, reb_freq):
    """
    Calculate the rebalancing dates based on the trading days and rebalancing frequency.

    Parameters:
    - trading_days: DatetimeIndex of trading days
    - reb_freq: Frequency of rebalancing (e.g., '1M', '3M', etc.)

    Returns:
    - rebalancing_dates: List of rebalancing dates adjusted to the next trading day if there is a mismatch
    """
    reb_dates = []
    first_date = trading_days[0]
    i = 1
    current_date = first_date
    
    while current_date <= trading_days[-1]:
        if current_date in trading_days:
            reb_dates.append(current_date)
        else:
            # If the rebalancing date is not a trading day, adjust to the nearest trading day
            next_trading_day = trading_days[trading_days > current_date][0]
            reb_dates.append(next_trading_day)
        
        # Move to the next rebalancing date based on frequency
        current_date =  first_date + pd.offsets.DateOffset(months=i*int(reb_freq[:-1])) if reb_freq.endswith('M') else pd.offsets.DateOffset(days=i*int(reb_freq[:-1]))
        i += 1
    return pd.DatetimeIndex(reb_dates)

def calc_log_vol(pv_series, trading_days = 252):
    log_returns = np.log(pv_series / pv_series.shift(1)).dropna()
    daily_log_vol = log_returns.std()
    annualized_vol = daily_log_vol * np.sqrt(trading_days)
    return annualized_vol


class momentum(strategy):

    def __init__(self, strategy_name):
        super().__init__(strategy_name)



    # Fonction suivante à améliorer
    def add_selection_df(self,Ul_data = True, path = None):
        print("Downloading selection Dataframe...")
        if Ul_data == True :
            self.selection_df = self.ULs_universe
        else : 
            self.selection_df = pd.read_excel(path, header=0,index_col=0,skiprows=0)
        print("Selection dataframe download : DONE")
        
    
    def calc_weights(self, func, **funcArgs):
        print("Weights calculation...")
        if self.ULs_universe is None or self.selection_df is None :
            print("Weights calculation impossible : No underlying data downloaded")
            return None
        else :
            weights_df = func(self.selection_df, **funcArgs)
            weights_df.columns = self.ULs_universe.columns
            return weights_df






    def extract_timeseries_for_rebalancing(self, reb_freq):
        
        '''
        Extract the timeseries for the values and weights of the underlying stocks for each rebalancing date.

        Parameters:
        - uls_universe: DataFrame containing the underlying stock values with trading days as index
        - weights: DataFrame containing the weights of the underlying stocks with trading days as index
        - reb_freq: Frequency of rebalancing (e.g., '1M', '3M', etc.)

        Returns:
        - reb_values: DataFrame containing the values of the underlying stocks for each selected rebalancing date
        - reb_weights: DataFrame containing the weights of the underlying stocks for each selected rebalancing date
        '''


        trading_days = self.ULs_universe.index
        rebalancing_dates = calculate_rebalancing_dates(trading_days, reb_freq)

        reb_stock_prices = self.ULs_universe.loc[rebalancing_dates]
        reb_weights = self.weights.loc[rebalancing_dates]

        return reb_stock_prices, reb_weights
    
    

    def calc_spis_pv(self):
        print("SPIs and Portfolio value calculation on rebalancing date...")
        if self.reb_weights_df is None :
            print("Warning : Weight rebalancing dataframe is empty")
            return None
        
        if self.reb_underlyings_df is None :
            print("Warning : Underlying rebalancing dataframe is empty")
            return None
        
        data = []
        reb_weights = self.reb_weights_df
        reb_stock_prices = self.reb_underlyings_df

        first_spis = {column : 100*weight/stock_price for column, weight, stock_price in zip(reb_stock_prices.columns, reb_weights.iloc[0], reb_stock_prices.iloc[0])}
        first_spis['index'] = reb_stock_prices.index[0] 
        first_spis['Portfolio Value'] = 100
        data.append(first_spis)
    

        for i in range(1,len(reb_stock_prices)) :
            #previous_date = reb_stock_prices.index[i-1]
            current_date = reb_stock_prices.index[i]
            previous_spis = pd.Series({col : data[-1][col] for col in reb_stock_prices.columns})
            current_prices = reb_stock_prices.loc[current_date]
            pv = (previous_spis*current_prices).sum() # portfolio value at time t   

            #New Spis calculation
            spis = {column : weight * pv / stock_price for column, weight, stock_price in zip(reb_stock_prices.columns, reb_weights.iloc[i], reb_stock_prices.iloc[i])}
            spis['index'] = current_date
            spis['Portfolio Value'] = pv
            data.append(spis)


        df_reb_spis = pd.DataFrame(data).set_index('index')


        print("SPIs and Portfolio value calculation on rebalancing date : DONE")
        return df_reb_spis


    def implement_strat(self,func, risk_free_rate = 0, **funcArgs ):
        """
        Implementation of the strategy :
        - Daily weights calculation
        - Daily SPIs calculation
        - Daily strategy value calculation

        Parameters:
        - func: function returning the dataframe containing daily 
        - funcArgs : dictionnary containing the parameters for the function func
        """

        print("_________________Strategy implementation_________________")
        print("(1/3) weights calculation...")

        self.weights = self.calc_weights(func, **funcArgs)
        #make sure weights indexes are same as the others df
        
        
        self.ULs_universe = self.ULs_universe.loc[self.weights.index]

        self.reb_underlyings_df, self.reb_weights_df = self.extract_timeseries_for_rebalancing(reb_freq = funcArgs['reb_freq'])
        self.reb_spis_df = self.calc_spis_pv()
        self.reb_pv_df = self.reb_spis_df['Portfolio Value']

        #Extend timeseries to all trading days 
        daily_Spis = pd.DataFrame(np.nan, index = self.ULs_universe.index, columns = self.ULs_universe.columns)
        #daily_pv = pd.Series(np.nan, index = self.ULs_universe.index, name = "Portfolio Value")

        daily_Spis = daily_Spis.combine_first(self.reb_spis_df[self.ULs_universe.columns])
        daily_Spis.fillna(axis = 0, method = 'ffill', inplace = True)
        self.spis_df = daily_Spis

        print("(2/3) Strategy value calculation...")

        daily_pv = (self.spis_df*self.ULs_universe).sum(axis = 1)
        self.pv_df = daily_pv

        print("(3/4) Calculating performance metrics...")

        self.performance_calculation(risk_free_rate = risk_free_rate)

        print("(4/4) Strategy implementation : Done")

        

    

    def performance_calculation(self, risk_free_rate = 0):
        print("Performance calculation...")
        total_return = (self.pv_df[-1]-self.pv_df[0])/self.pv_df[0]
        nb_years = (pd.to_datetime(self.pv_df.index[-1]) - pd.to_datetime(self.pv_df.index[0])).days/365.25
        annualized_return = total_return/nb_years
        annual_vol = calc_log_vol(self.pv_df)
        sharpe_ratio = (annualized_return-risk_free_rate)/annual_vol


        self.annualized_return = annualized_return
        self.total_return = total_return
        self.annualized_vol = annual_vol
        self.sharpe_ratio = sharpe_ratio
        #Miss Benchmark performance comparison 
        #for key in self.benchmarks
        #benchmark_performance = {}
        data = {"total_return" :total_return,"annualized_return" : annualized_return,  "annual_vol" : annual_vol, "sharpe_ratio" : sharpe_ratio}
        performance = pd.DataFrame(data, index = [self.name])
        self.performance_table = performance
        print("Performance calculation : DONE")


    def plot_graph(self,graph_name):
        print("Plotting graph...")
        ax = self.pv_df.plot(title=graph_name, figsize=(10, 6),grid = True) 
        ax.set_xlabel('Date')
        ax.set_ylabel('Strategy Value')
        plt.show()
        print("Plotting graph : DONE")

    def to_excel(self, path, dl_data = False): 
        print("Transfering data to Excel...")
        try :
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                self.pv_df.to_excel(writer, sheet_name = "Portfolio Value")
                self.weights.to_excel(writer, sheet_name = "Weights")
                self.spis_df.to_excel(writer, sheet_name = "SPIs")
                self.performance_table.to_excel(writer, sheet_name = "Performance")
                if dl_data == True : 
                    self.ULs_universe.to_excel(writer, sheet_name = "Data")   
            print("Data transfer to Excel : DONE")
        except Exception as e:
            print(f"Data transfer to Excel Error \n Details: {e}")



def Long_Short_strat_rules(uls_universe, reb_freq = '1M', nb_long = 5):
    """
    Calculate the strategy weights based on the rules provided.
    
    Parameters:
    - uls_universe: DataFrame containing the underlying asset prices.
    - reb_freq: Frequency of rebalancing (default is daily '1D').
    
    Returns:
    - weights_df: DataFrame containing the calculated weights.
    """
    # Calculate daily returns
    
    daily_returns = np.log(uls_universe/uls_universe.shift(1)).dropna()

    # Calculate rolling 3-day average returns
    rolling_avg_returns = daily_returns.rolling(window=20).mean().dropna()


    reb_indexes = calculate_rebalancing_dates(rolling_avg_returns.index,reb_freq=reb_freq)


    weights_df = pd.DataFrame(index=rolling_avg_returns.index, columns=uls_universe.columns).fillna(0)


    for date in reb_indexes:
        # Get the average returns for the current date
        avg_returns = rolling_avg_returns.loc[date]

        # Select top 5 and bottom 5 assets
        top_5_assets = avg_returns.nlargest(5).index
        bottom_5_assets = avg_returns.nsmallest(5).index

        # Assign weights
        weights_df.loc[date, top_5_assets] = 1/nb_long
        weights_df.loc[date, bottom_5_assets] = -1/nb_long

    mask_all_zeros = (weights_df == 0).all(axis=1)

    #print("Second weights")
    #print(weights_df.loc[reb_indexes])
    weights_df = weights_df.where(~mask_all_zeros).ffill()

    return weights_df

def Long_Only_strat_rules(uls_universe, reb_freq = '1M', nb_long = 5):
    """
    Calculate the strategy weights based on the rules provided.
    
    Parameters:
    - uls_universe: DataFrame containing the underlying asset prices.
    - reb_freq: Frequency of rebalancing (default is daily '1D').
    
    Returns:
    - weights_df: DataFrame containing the calculated weights.
    """
    # Calculate daily returns

    daily_returns = np.log(uls_universe/uls_universe.shift(1)).dropna()

    # Calculate rolling 3-day average returns
    rolling_avg_returns = daily_returns.rolling(window=20).mean().dropna()


    reb_indexes = calculate_rebalancing_dates(rolling_avg_returns.index,reb_freq=reb_freq)


    weights_df = pd.DataFrame(index=rolling_avg_returns.index, columns=uls_universe.columns).fillna(0)


    for date in reb_indexes:
        # Get the average returns for the current date
        avg_returns = rolling_avg_returns.loc[date]

        # Select top 5 and bottom 5 assets
        top_5_assets = avg_returns.nlargest(5).index

        # Assign weights
        weights_df.loc[date, top_5_assets] = 1/nb_long

    mask_all_zeros = (weights_df == 0).all(axis=1)

    #print("Second weights")
    #print(weights_df.loc[reb_indexes])
    weights_df = weights_df.where(~mask_all_zeros).ffill()

    return weights_df





#"dataset_stock_original.xlsx"
#"data_Forex_strat.xlsx"
path = "dataset_stock_original.xlsx"
path2 = "strat_Stat.xlsx"
#df = pd.read_excel(path, header=0,index_col=0,skiprows=0)
#df2 = Fx_Carry_strat_rules(df, reb_freq = '1M')
#with pd.ExcelWriter(path2, engine='openpyxl') as writer:
#                df2.to_excel(writer, sheet_name = "Portfolio Value")

parameters = {'reb_freq' : '1M', "nb_long" : 5}

test_strat = momentum("Long Only Test")
test_strat.add_data(path=path, decimal_separator=',')
test_strat.add_selection_df()

test_strat.implement_strat(Long_Only_strat_rules,risk_free_rate=0, **parameters)
test_strat.plot_graph("test")
test_strat.to_excel(path = path2, dl_data = True)


#test_strat



#c = ('Close', 'AAPL')
'''
test_strat.add_data("AAPL", "2023-01-01", "2023-05-01")
test_strat.add_data("MSFT", "2023-01-04", "2023-05-01")
test_strat.add_benchmark("TSLA", "2023-01-04", "2023-05-01")
test_strat.add_benchmark("", "2023-01-04", "2023-05-01")

b = test_strat.benchmarks
a = test_strat.ULs_universe
#a = a[c]
print(type(a))
print(a)
print(type(b))
print(b)
'''


del test_strat

        
        