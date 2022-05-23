from typing import Optional
import pandas as pd
from stock_simulator import StockSimulator
from get_parameters import event_mapping_dict, Wraken_macro, Wraken_micro

# index_initial_price = 1000
# comp_initial_price = 100
# comp_event = 'new_product'
# event = 'tech_blossom'

# company_name = ['Wraken', 'Surgo']

# for company in company_name:
#     if company == 'Wraken':
#         comp_initial_price = 100
#     if
#     index_initial_price = 1000
    
#     comp_event = 'new_product'
#     event = 'tech_blossom'
#     simulator = StockSimulator(index_initial_price, comp_initial_price, event_mapping_dict, Wraken_macro, Wraken_micro, event, comp_event)
#     index_simulation, first_order_simulation, comp_simulation, third_order_simulation = simulator.end_loop_simulation()
#     current_comp_price = simulator.comp_price_output()
#     current

"""
daily stock growth -> stock_simulator
1. stock ticker
2. import from stock simulator
3. weight for each stock
stock_tickerA = None,
        stock_tickerB = None,
        stock_tickerC = None,
        stock_tickerD = None,
        stock_tickerE = None,
        stock_tickerF = None,
        stock_tickerG = None,
        stock_tickerH = None,
        stock_tickerI = None,
        stock_tickerJ = None,
        stock_tickerK = None,
        stock_tickerL = None,
        stock_tickerM = None
self.user_dictionary contains ticker and return for stocks.
"""

class UserInterface:
    """
    UserInterface is a class containing features to describe an user's portfolio movement.
    The default starting value for a portfolio of any new users is $10,000.
    The change in value depends on the net change of all holds within the portfolio.
    """
    def __init__(
        self,
        stock_current_price_dict,
        portfolio_starting_value: Optional[float] = 0,
        cash_starting_value: Optional[float] = 10000
    ):
        self.stock_current_price_dict = stock_current_price_dict
        self.return_dictionary = self._set_initial_return_dictionary()
        self.value_dictionary = self._set_initial_value_dictionary
        self.stock_ticker_list = []
        self.stock_weight_list = []
        self.portfolio_net_return_percentage = "{:.2%}".format(0)
        self.portfolio_net_value = portfolio_starting_value
        self.cash_value = cash_starting_value

        self.stock_initial_price = {}
        for ticker in self.stock_ticker_list:
            self.stock_initial_price[ticker] = 0

    def _set_initial_return_dictionary(self):
        ticker_list = self.stock_ticker_list
        return_rate = [0 for _ in range(len(self.stock_ticker_list))]
        return_dictionary = {ticker_list[index]: return_rate[index] for index in range(len(ticker_list))}
        return return_dictionary
    
    def _set_initial_value_dictionary(self):
        ticker_list = self.stock_ticker_list
        value = [0 for _ in range(len(self.stock_ticker_list))]
        value_dictionary = {ticker_list[index]: value[index] for index in range(len(ticker_list))}
        return value_dictionary
    
    def return_change(self):
        for ticker in self.stock_ticker_list:
            current_price = self.stock_current_price_dict[ticker]
            initial_price = self.stock_initial_price[ticker]
            stock_return = (current_price - initial_price)/initial_price
            stock_return_rate = "{:.2%}".format(stock_return)
            self.return_dictionary[ticker] = stock_return_rate

    def per_stock_value_change(self, ticker):
        stock_return = self.return_dictionary[ticker]
        weight = self.value_dictionary[ticker]/self.portfolio_net_value
        self.stock_weight_list.append(weight)
        stock_adjusted_return = weight * stock_return
        percentage_return = "{:.2%}".format(stock_adjusted_return)
        stock_value_change = stock_adjusted_return * self.value_dictionary[ticker]
        return stock_adjusted_return, stock_value_change, percentage_return
    
    def portfolio_return_value_change(self):
        portfolio_net_return = 0
        for ticker in self.stock_ticker_list:
            stock_adjusted_return, stock_value_change, percentage_return = self.per_stock_value_change()
            portfolio_net_return += stock_adjusted_return
            self.value_dictionary[ticker] += stock_value_change
            self.return_dictionary[ticker] += percentage_return
        portfolio_net_return_percentage = "{:.2%}".format(portfolio_net_return)
        self.portfolio_net_return_percentage = portfolio_net_return_percentage

    def portfolio_value_trade(self):
        buy_or_sell = input('Buy or Sell')
        ticker = input("Ticker")
        share_number = input("Total Share")
        price = input("Price")
        net_value = share_number * price
        if ticker not in self.stock_ticker_list:
            self.stock_initial_price[ticker] = price
            self.value_dictionary[ticker] = net_value
            self.stock_ticker_list.append(ticker)
            if buy_or_sell == 'BUY':
                if self.cash_value - net_value >= 0:
                    self.portfolio_net_value += net_value
                    self.cash_value -= net_value
                else:
                    print('Warning! You don\' have enough money \in your cash account.')
            if buy_or_sell == 'SELL':
                if self.value_dictionary[ticker] - net_value >= 0:
                    self.portfolio_net_value -= net_value
                    self.cash_value += net_value
                else:
                    print('Warning! Value exceeding for portfolio\'s current value.')
        else:
            if buy_or_sell == 'BUY':
                if self.cash_value - net_value >= 0:
                    self.portfolio_net_value += net_value
                    self.cash_value -= net_value
                    self.value_dictionary[ticker] += net_value
                else:
                    print('Warning! You don\' have enough money \in your cash account.')
            if buy_or_sell == 'SELL':
                if self.value_dictionary[ticker] - net_value >= 0:
                    self.portfolio_net_value -= net_value
                    self.cash_value += net_value
                    self.value_dictionary[ticker] -= net_value
                else:
                    print('Warning! Value exceeding for portfolio\'s current value.')

    
    def portfolio_value_output(self):
        print(self.portfolio_net_value)

    def cash_balance_output(self):
        print(self.cash_value)
    
    def portfolio_return_percentage(self):
        print(self.portfolio_net_return_percentage)
    
    def portfolio_dash_board(self):
        return_list = list(self.return_dictionary.values())
        value_list = list(self.value_dictionary.values())
        dash_board = pd.DataFrame(
            {'weight': self.stock_weight_list,
             'return_rate': return_list,
             'value': value_list
            })
        dash_board.set_index(self.stock_ticker_list, inplace=True)
        return dash_board
        

        


