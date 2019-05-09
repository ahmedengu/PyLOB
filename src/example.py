import json
import math
import operator
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=np.RankWarning)
np.seterr(divide='ignore', invalid='ignore')


def ema(series, period):
    values = np.zeros(len(series))
    period = 2.0 / (period + 1)
    for i, val in enumerate(series):
        values[i] = val if i == 0 else period * val + (1 - period) * values[i - 1]
    return values


def sma(series, period):
    values = np.zeros(len(series))
    for i, val in enumerate(series):
        series_slice = series[:i + 1][-min(i + 1, period):]
        values[i] = sum(series_slice) / min(i + 1, period)
    return values


def change(series):
    values = np.zeros(len(series))
    for i, val in enumerate(series):
        values[i] = 0 if i == 0 else val - series[i - 1]
    return values


def linreg(series, period, offset):
    values = np.zeros(len(series))
    for i, val in enumerate(series):
        series_slice = series[:i + 1][-min(i + 1, period):]
        coefs = np.polyfit([i for i in range(len(series_slice))], series_slice, 1)
        slope = coefs[0]
        intercept = coefs[1]
        values[i] = intercept + slope * (period - 1 - offset)
    return values


def cci(series, period):
    values = np.zeros(len(series))
    for i, val in enumerate(series):
        series_slice = series[:i + 1][-min(i + 1, period):]
        current_sma = sma(series_slice, period)[-1]
        values[i] = (val - current_sma) / (0.015 * sum([abs(x - current_sma) for x in series_slice]) / period)
    return values


def ohlc4(close_prices, open_prices, high_prices, low_prices):
    values = np.zeros(len(close_prices))
    for i, val in enumerate(close_prices):
        values[i] = ((close_prices[i] + open_prices[i] + high_prices[i] + low_prices[i]) / 4)
    return values


# call the pinescript code every X minute and make sure that the call happen only when there is an update
def apply_strategy():
    global ticks, bars, bars_len, open_orders_count, last_time, tape_len, order_size, current_active, active_order
    if len(lob.tape) > tape_len and pd.Timedelta(lob.time, unit='ms') - pd.Timedelta(last_time,
                                                                                     unit='ms') >= pd.Timedelta(period):
        ticks = pd.DataFrame(list(lob.tape))
        ticks['time'] = pd.to_datetime(ticks['time'], unit='ms')
        ticks.set_index('time', inplace=True)
        bars = pd.DataFrame(ticks['price'].resample(period).ohlc())
        bars.columns = ['open', 'high', 'low', 'close']
        v = pd.DataFrame(ticks['qty'].resample(period).sum())
        v.columns = ['qty']
        bars['qty'] = v['qty']
        bars.dropna(inplace=True)
        if len(bars) > bars_len and bars_len > 2:
            src_close = bars['close'].values
            ohlc4_val = ohlc4(bars['close'].values, bars['open'].values, bars['high'].values, bars['low'].values)

            ema_val = ema(ohlc4_val, ema_period)

            long_condition = src_close[-1] > ema_val[-1]
            short_condition = src_close[-1] < ema_val[-1]
            # print("Buy: " + str(long_condition) + ", Sell: " + str(short_condition))

            if long_condition and current_active != 1:
                cancel_active_order(active_order)
                close_filled_orders()
                open_orders_count += order_size
                order_data.update({'side': 'bid',
                                   'price': src_close[-1],
                                   'qty': order_size,
                                   'tid': str(open_orders_count) + '_buy',
                                   'timestamp': bars.index[-1].value // int(1e6)})
                trades, active_order = lob.processOrder(order_data, False, False)
                append_filled_trades(trades)
                current_active = 1

            if short_condition and current_active != -1:
                cancel_active_order(active_order)
                close_filled_orders()
                open_orders_count += order_size
                order_data.update({'side': 'ask',
                                   'price': src_close[-1],
                                   'qty': order_size,
                                   'tid': str(open_orders_count) + '_sell',
                                   'timestamp': bars.index[-1].value // int(1e6)})
                trades, active_order = lob.processOrder(order_data, False, False)
                append_filled_trades(trades)
                current_active = -1

        bars_len = len(bars)
        last_time = lob.time
        tape_len = len(lob.tape)


def close_filled_orders():
    global filled_orders, close_orders_count
    if len(filled_orders):
        for i, trade in enumerate(filled_orders):
            if 'buy' in trade['party1'][0] or 'sell' in trade['party1'][0]:
                order_data.update({'side': trade['party2'][1],
                                   'price': bars['close'].values[-1],
                                   'qty': trade['qty'],
                                   'tid': trade['party1'][0].split('_')[0] + '_close_' + str(i),
                                   'timestamp': bars.index[-1].value // int(1e6)})
            else:
                order_data.update({'side': trade['party1'][1],
                                   'price': bars['close'].values[-1],
                                   'qty': trade['qty'],
                                   'tid': trade['party2'][0].split('_')[0] + '_close_' + str(i),
                                   'timestamp': bars.index[-1].value // int(1e6)})
            try:
                lob.processOrder(order_data, False, False)
                close_orders_count += order_data['qty']
            except Exception as e:
                print(e)
                pass
        filled_orders = []


def cancel_active_order(active_order):
    try:
        lob.cancelOrder(active_order.side, active_order.idNum)
    except Exception as e:
        print(e)
        pass


def append_filled_trades(trades):
    global filled_orders
    if len(trades):
        for trade in trades:
            if 'sell' in trade['party1'][0] or 'buy' in trade['party1'][0] or \
                    'sell' in trade['party2'][0] or 'buy' in trade['party2'][0]:
                filled_orders.append(trade)


if __name__ == '__main__':

    from PyLOB.orderbook import OrderBook

    filename = 'BITFINEX_SPOT_BTC_USD_2018-10-01&2018-10-01.json'

    with open(filename) as f:
        dataset = json.load(f)

    lob = OrderBook()
    open_orders_count = 0
    close_orders_count = 0
    bars_len = 0
    last_time = 0
    tape_len = 0
    period = '1Min'
    order_size = 1
    current_active = 0  # -1 sell, 1 buy , 0 none
    filled_orders = []
    active_order = {}
    ema_period = 11
    capital = 10000

    for i, order in enumerate(dataset):
        order_data = {'type': 'limit', 'timestamp': pd.to_datetime(order['time_coinapi']).value // int(1e6)}
        for j, ask in enumerate(order['asks']):
            order_data.update({'side': 'ask',
                               'price': ask['price'],
                               'qty': ask['size'],
                               'tid': str(i) + '_' + str(j)})
            trades, orderInBook = lob.processOrder(order_data, False, False)
            append_filled_trades(trades)

        for j, bid in enumerate(order['bids']):
            order_data.update({'side': 'bid',
                               'price': bid['price'],
                               'qty': bid['size'],
                               'tid': str(i) + '_' + str(j)})
            trades, orderInBook = lob.processOrder(order_data, False, False)
            append_filled_trades(trades)

        apply_strategy()

    cancel_active_order(active_order)
    close_filled_orders()

    print(ticks)
    print(lob)
    print(bars)
    fills = ticks[operator.or_(ticks['party1'].apply(lambda x: 'sell' in x[0] or 'buy' in x[0] or 'close' in x[0]),
                               ticks['party2'].apply(lambda x: 'sell' in x[0] or 'buy' in x[0] or 'close' in x[0]))]
    fills['type'] = 'open'
    fills['type'].loc[operator.or_(fills['party1'].apply(lambda x: 'close' in x[0]),
                                   fills['party2'].apply(lambda x: 'close' in x[0]))] = 'close'

    capital_list = []
    for index,fill in fills.iterrows():
        if fill['type'] == 'open':
            capital -= fill['qty'] * fill['price']
        else:
            capital += fill['qty'] * fill['price']

        capital_list.append(capital)

    fills['initial capital: 10000'] = capital_list

    print(fills)
    open_only = fills.loc[fills['type'] == 'open']
    print('Placed open quantity:', open_orders_count)
    print('Filled open quantity:', open_only['qty'].sum())
    print('Placed close quantity:', open_orders_count)
    print('Filled close quantity:', fills['qty'].sum() - open_only['qty'].sum())
    print('All filled orders quantity:', fills['qty'].sum())

    bars.to_csv('bars.csv')
    ticks.to_csv('ticks.csv')
    fills.to_csv('fills.csv')
